
import os
import json
import time
import re
import asyncio
import textwrap
import yaml
from typing import Dict, AsyncGenerator, List, Tuple, Any, Optional
from pathlib import Path
from config_utils import ConfigManager
from aibot_logger import ChatLogger
from handler_base import BaseModelHandler, UsageTracker, classify_error
import logging

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama, llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LocalLlamaModel:

    def __init__(self):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python 라이브러리가 안됩니다. 확인하시길...")

        from config_utils import ConfigManager
        from pathlib import Path

        config_manager = ConfigManager()
        paths_config = config_manager.get_paths_config()

        model_path_from_config = paths_config.get('local_llama_model_path')

        if model_path_from_config:
            self.model_path = str(Path(model_path_from_config).resolve())
        else:
            self.model_path = "../models/meta-llama-3.1-8b-instruct-q4_k_m.gguf"

        if not os.path.exists(self.model_path):
            logger.warning(f"기본 경로에서 모델을 찾을 수 없습니다: {self.model_path}")

            alternative_paths = [
                "./meta-llama-3.1-8b-instruct-q4_k_m.gguf",
                "../models/llama-3.1-8b-instruct-q4_k_m.gguf",
                "./llama-3.1-8b-instruct-q4_k_m.gguf",
                "meta-llama-3.1-8b-instruct-q4_k_m.gguf"
            ]

            found_path = None
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    found_path = alt_path
                    break

            if found_path:
                self.model_path = found_path
                logger.info(f"대안 경로에서 모델 파일 발견: {self.model_path}")
            else:
                available_files = []
                try:
                    files = [f for f in os.listdir("./") if f.endswith('.gguf')]
                    available_files.extend([f"./{f}" for f in files])
                except:
                    pass

                error_msg = f"모델 파일을 찾을 수 없습니다.\n확인한 경로:\n- {self.model_path}\n"
                error_msg += "\n".join([f"- {path}" for path in alternative_paths])
                if available_files:
                    error_msg += f"\n\n사용 가능한 .gguf 파일들:\n"
                    error_msg += "\n".join([f"- {f}" for f in available_files])

                raise FileNotFoundError(error_msg)

        context_size = int(os.getenv("LLAMA_CONTEXT_SIZE", "16384"))

        logger.info(f"로컬 Llama 모델 파일 로드 중: {self.model_path}")
        logger.info(f"설정: 컨텍스트={context_size}")

        try:
            logger.info("GPU 우선 모델 로딩 시도...")

            try:
                logger.info("1단계: 풀 GPU 모드로 시도...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=context_size,
                    n_threads=8,
                    n_gpu_layers=35,
                    n_batch=512,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    logits_all=False,
                    embedding=False
                )
                logger.info(f"✅ 풀 GPU 모드로 로컬 Llama 모델 로드 완료")
                logger.info(f"GPU 레이어: 35개, 컨텍스트: {context_size}")
                return

            except Exception as gpu_error:
                logger.warning(f"풀 GPU 모드 실패: {str(gpu_error)}")

            try:
                logger.info("2단계: 적당한 GPU 레이어로 시도...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=context_size,
                    n_threads=8,
                    n_gpu_layers=20,
                    n_batch=512,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    logits_all=False,
                    embedding=False
                )
                logger.info(f"✅ 적당한 GPU 모드로 로컬 Llama 모델 로드 완료")
                logger.info(f"GPU 레이어: 20개, 컨텍스트: {context_size}")
                return

            except Exception as limited_gpu_error:
                logger.warning(f"적당한 GPU 모드 실패: {str(limited_gpu_error)}")

            try:
                logger.info("3단계: 최소 GPU 레이어로 시도...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=context_size,
                    n_threads=8,
                    n_gpu_layers=10,
                    n_batch=256,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    logits_all=False,
                    embedding=False
                )
                logger.info(f"✅ 최소 GPU 모드로 로컬 Llama 모델 로드 완료")
                logger.info(f"GPU 레이어: 10개, 컨텍스트: {context_size}")
                return

            except Exception as min_gpu_error:
                logger.warning(f"최소 GPU 모드 실패: {str(min_gpu_error)}")

            try:
                logger.info("4단계: CPU 전용 모드로 fallback...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=context_size,
                    n_threads=8,
                    n_gpu_layers=0,
                    n_batch=512,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    logits_all=False,
                    embedding=False
                )
                logger.warning(f"⚠️ CPU 전용 모드로 로컬 Llama 모델 로드 완료 (GPU 사용 실패)")
                logger.warning(f"💡 GPU 드라이버나 CUDA 설정을 확인해보세요")
                return

            except Exception as cpu_error:
                logger.error(f"CPU 전용 모드도 실패: {str(cpu_error)}")

            try:
                logger.info("5단계: 최소 설정으로 마지막 시도...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=4,
                    n_gpu_layers=0,
                    n_batch=256,
                    verbose=True,
                    use_mmap=False,
                    use_mlock=False
                )
                logger.warning(f"⚠️ 최소 설정으로 로컬 Llama 모델 로드 완료")
                return

            except Exception as minimal_error:
                logger.error(f"최소 설정 모드도 실패: {str(minimal_error)}")
                raise Exception(f"모든 로딩 방법 실패. 마지막 오류: {str(minimal_error)}")

        except Exception as e:
            logger.error(f"❌ 로컬 모델 로드 오류: {str(e)}")
            raise

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01, top_p: float = 0.9) -> Dict:
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.1,
            top_k=40,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False
        )

        return {
            "generation": response["choices"][0]["text"],
            "prompt_token_count": len(self.llm.tokenize(prompt.encode('utf-8'))),
            "generation_token_count": len(self.llm.tokenize(response["choices"][0]["text"].encode('utf-8'))),
            "stop_reason": "stop"
        }

    def generate_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01, top_p: float = 0.9):
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.1,
            top_k=10,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False,
            stream=True
        ):
            yield chunk["choices"][0]["text"]

    def generate_complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01, top_p: float = 0.9) -> str:
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.1,
            top_k=10,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False,
            stream=False
        )

        return response["choices"][0]["text"]

class LlamaHandler(BaseModelHandler):
    registry_key = "llama"  # handler_registry.HANDLER_CLASSES 키와 일치
    is_local = True          # 로컬 llama-cpp 또는 Bedrock (둘 다 외부 OpenAI 호환 endpoint 는 아님)

    def build_agent_prompt(self, messages, options=None):
        """OpenAI messages -> Llama-3.1 chat template. agent_complete용."""
        parts = ["<|begin_of_text|>"]
        for m in messages:
            parts.append("<|start_header_id|>%s<|end_header_id|>\n%s<|eot_id|>" % (m.get("role","user"), m.get("content","")))
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        return "".join(parts)

    def _agent_generate(self, prompt):
        if not getattr(self, "local_model", None):
            raise RuntimeError("llama local_model 미로드")
        return self.local_model.generate_complete(prompt, 2048, 0.3, 0.9)

    # Bedrock 경유 시 streaming 불가능할 수 있음 — 실제 운영에서 issue 발생 시 False 로
    # supports_streaming = False

    def __init__(self, rag_system=None, use_kg: bool = True, intent_analyzer=None):
        super().__init__(rag_system, use_kg, intent_analyzer)

        # reload_system_prompt 이 [prompts] 의 올바른 키('llama')를 읽게 한다.
        # 미설정 시 base 폴백(gpt_oss)으로 새 카트리지 프롬프트를 놓친다.
        self.handler_type = "llama"

        self.config_manager = ConfigManager()

        self.model_opt = os.getenv("model_opt", "on").lower()
        self.use_local_model = self.model_opt == "on"

        # 실제 서빙 GGUF 파일명에서 라벨 파생 (gemma/gpt-oss 핸들러와 동일 규칙)
        try:
            _gguf_stem = Path(self.config_manager.get_paths_config().get('local_llama_model_path', '')).stem
        except Exception:
            _gguf_stem = ''
        self.model_name = _gguf_stem or 'llama-3.1-8b'
        self.agent_model_name = self.model_name

        self.local_model = None
        self.bedrock_runtime = None

        self.aws_region = "us-east-1"
        self.model_id = "us.meta.llama3-1-8b-instruct-v1:0"

        raw_prompt_path = self.config_manager.get_prompt_config().get('llama')
        self.system_prompt = self._clean_prompt_content(self.load_prompt_from_file(raw_prompt_path))

        self.available = False
        self._initialize_model()

        if self.use_local_model and self.local_model:
            self.logger = ChatLogger("local-llama")
        elif self.bedrock_runtime:
            self.logger = ChatLogger("aws-bedrock-llama")
        else:
            self.logger = ChatLogger("llama-unavailable")

    # _clean_prompt_content: BaseModelHandler default 사용.

    def reload_system_prompt(self) -> str:
        """base 리로드 후 __init__(285줄)과 동일하게 _clean_prompt_content 재적용.

        llama 만 __init__ 에서 base 프롬프트를 정리하므로, 정리를 안 하는 base 리로드를
        그대로 쓰면 리로드 후 프롬프트가 __init__ 결과와 어긋난다.
        """
        path = super().reload_system_prompt()
        if path:
            self.system_prompt = self._clean_prompt_content(self.system_prompt)
        return path

    def _perform_translation(self, query: str, mode: str, preserved: dict = None) -> str:
        try:
            if not self.available:
                logger.warning("Llama 모델이 사용 불가능하여 사전 번역 사용")
                return self._enhanced_dictionary_translate(query)

            # 번역 시스템 프롬프트는 카트리지가 정의 — [prompts] translation_summary/normal/cve.
            # 다른 핸들러(gemma·gpt_oss·qwen·openai)는 이미 이 통로를 쓴다. llama 만 인라인이라
            # 도메인 탈색이 이 경로를 비껴갔었다 (보안 실험실 문구·CVE 규칙이 코드에 박혀 있었다).
            from config_utils import load_translation_prompt
            system_instructions = load_translation_prompt(mode)
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n{system_instructions}\n<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n{query}\n<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )

            if self.use_local_model and self.local_model:
                translation = self.local_model.generate_complete(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.1
                )
            else:
                request_body = {
                    "prompt": prompt,
                    "max_gen_len": 100,
                    "temperature": 0.1,
                    "top_p": 0.9
                }

                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json"
                )
                response_body = json.loads(response['body'].read())
                translation = response_body.get('generation', '')

            print(f"========= 정리 전 번역 내용 =========\n{translation}\n===============================")

            translation = translation.strip()
            translation = re.sub(r'^(Keywords:|키워드:|English:|Translation:)', '', translation).strip()
            translation = translation.split('\n')[0].strip()
            print(f"========= 정리 후 번역 내용 =========\n{translation}\n===============================")

            if mode == 'CVE':
                return translation
            else:
                if translation:
                    original_lower = query.lower()
                    translation_lower = translation.lower()

                    # 번역이 핵심 동사를 빠뜨렸을 때 되살리는 폴백. 도메인 중립 동사만 남긴다
                    # (원래는 공격·침입 같은 보안 도메인 어휘가 박혀 있었다).
                    critical_mappings = [
                        (['로그인', '접속'], 'login'),
                        (['실패한', '실패'], 'failure'),
                        (['차단한', '차단'], 'block'),
                        (['탐지한', '탐지'], 'detect')
                    ]

                    for korean_terms, english_term in critical_mappings:
                        if any(term in original_lower for term in korean_terms):
                            if english_term not in translation_lower:
                                translation = f"{english_term} {translation}"
                                logger.warning(f"누락된 핵심 키워드 '{english_term}' 추가됨")

                    logger.info(f"키워드 번역: '{query}' → '{translation}'")
                    return translation
                else:
                    return self._enhanced_dictionary_translate(query)

        except Exception as e:
            logger.error(f"LLaMA 번역 오류: {e}")
            return self._enhanced_dictionary_translate(query)

    def _initialize_model(self):
        if self.use_local_model:
            try:
                self.local_model = LocalLlamaModel()
                self.available = True
                logger.info("✅ 로컬 Llama 모델 초기화 완료")
                return

            except Exception as e:
                logger.error(f"❌ 로컬 모델 초기화 실패: {str(e)}")
                self.local_model = None
                self.use_local_model = False
                logger.warning("AWS Bedrock으로 전환")

        if not self.use_local_model:
            logger.info("🌐 AWS Bedrock 초기화...")
            try:
                self.bedrock_runtime = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.aws_region
                )
                self.available = True
                logger.info(f"✅ AWS Bedrock 초기화 완료")
            except Exception as e:
                logger.error(f"❌ AWS Bedrock 오류: {str(e)}")
                logger.warning("💡 로컬 모델과 AWS Bedrock 모두 사용할 수 없습니다.")
                logger.warning("   - 로컬 모델: 모델 파일 확인 필요")
                logger.warning("   - AWS Bedrock: AWS 자격 증명 확인 필요")
                self.available = False

    def create_clean_prompt(self, query: str, context: str = "", chat_history: str = "", intent: str = "", response_format: bool = False, request: dict = None) -> str:

        from datetime import datetime, timedelta
        now = datetime.now()
        current_date = now.strftime("%Y%m%d")
        current_readable = now.strftime("%Y년 %m월 %d일 (%A)")

        system_prompt_with_date = self.system_prompt.replace(
            "current date (20250629)",
            f"current date ({current_date})"
        ).replace(
            "current date is 20250629",
            f"current date is {current_date}"
        )

        if intent and intent.upper() == "QNA":
            system_prompt_with_date = self._apply_response_format(system_prompt_with_date, response_format)

        prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt_with_date}"

        prompt += f"\n\n[CURRENT DATE & CALCULATION GUIDE]"
        prompt += f"\n📅 Current date: {current_date} ({current_readable})"
        prompt += f"\n"
        prompt += f"\n⚠️ IMPORTANT: When user asks for 'recent X days' or 'last X days', calculate BACKWARDS from today:"
        prompt += f"\n- Recent 7 days = {(now - timedelta(days=6)).strftime('%Y%m%d')} to {current_date}"
        prompt += f"\n- Recent 30 days = {(now - timedelta(days=29)).strftime('%Y%m%d')} to {current_date}"
        prompt += f"\n- Yesterday = {(now - timedelta(days=1)).strftime('%Y%m%d')}"
        prompt += f"\n- Last week = {(now - timedelta(days=6)).strftime('%Y%m%d')} to {current_date}"
        prompt += f"\n"
        prompt += f"\n🚫 Do NOT use future dates unless explicitly asked for future periods!"

        if context.strip():
            prompt += f"\n\n🎯 Reference documents:\n{context}"

        if chat_history.strip():
            prompt += f"\n\n🕒 Previous conversation:\n{chat_history}"

        user_content = query

        if request and request.get("app_context") and intent and intent.upper() in ['ACTION', 'PLAN']:
            user_content += f"\n\n## 시스템 컨텍스트\n현재 설치된 앱: {request['app_context']}"

        if re.search(r'[가-힣]', query):
            if self.config_manager.get_option_config().get('translation') == 'True':
                translated_question, _ = self.translate_query_for_search(query, intent)
                if translated_question and translated_question != query:
                    user_content += f"\n(English: {translated_question})"

        prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        return prompt

    async def generate_complete(self, request, user_id=None, channel_id=None, connection=None, thread_ts=None,
                          context=None, source_files=None, related_concepts=None, concepts=None, intent_info=None) -> Dict:
        start_time = time.time()
        self._usage_tracker = UsageTracker(model=self.model_name, user_id=user_id or "")

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            history = request.get("history", [])
            response_format = request.get("response_format", False)
            print(f"[DEBUG] history length: {len(history) if history else 0}")

        else:
            query = str(request)
            question_type = None
            history = None
            response_format = False

        if not self.available:
            return {
                "answer": "Llama 모델을 사용할 수 없습니다.",
                "intent": None,
                "sources": [],
                "response_time": 0,
                "question_type": question_type
            }

        chat_session = self.chat_session


        if context is None:
            if self.rag_system:
                try:
                    result = await self.get_related_documents(request)
                    if len(result) >= 6:
                        context, source_files, related_concepts, concepts, intent_info, reference_summary = result
                    elif len(result) >= 5:
                        context, source_files, related_concepts, concepts, intent_info = result
                    else:
                        context, source_files, related_concepts, concepts = result

                    self.last_sources = source_files

                    if related_concepts and self.use_kg:
                        concept_context = "\n\n관련 개념:\n"
                        for concept_info in related_concepts[:3]:
                            concept = concept_info.get('concept')
                            concept_context += f"- {concept}\n"
                        context += concept_context
                except Exception as e:
                    logger.error(f"RAG 검색 중 오류: {e}")
                    context, source_files, related_concepts, concepts = "", [], [], []

            selected_intent = question_type or (intent_info.get('primary_intent', '') if intent_info else '')
        elif context != "":
            self.last_sources = source_files if source_files else []

            if related_concepts and self.use_kg:
                concept_context = "\n\n관련 개념:\n"
                for concept_info in related_concepts[:3]:
                    concept = concept_info.get('concept')
                    concept_context += f"- {concept}\n"
                context += concept_context

            selected_intent = question_type or (intent_info.get('primary_intent', '') if intent_info else '')

        original_prompt = self.system_prompt


        if self.config_manager.get_option_config().get('user_intent_prompt') == 'True' and selected_intent:
            logger.info(f"의도 분석 개별 프롬프트 사용: {selected_intent}")

            intent_upper = selected_intent.upper()

            if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('post_action'))
                print(f"🔧 Llama POST-ACTION 프롬프트 로드 완료")
            elif intent_upper == 'ACTION':
                self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('action'))
            elif intent_upper == 'PLAN':
                self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('plan'))
            else:
                qna_intent = self._safe_analyze_qna_subtype(query)
                if qna_intent.upper() == 'CVE':
                    self.system_prompt = self._clean_prompt_content(self.load_prompt_from_file(self.config_manager.get_prompt_config().get('qna_cve')))
                else:
                    self.system_prompt = self._clean_prompt_content(self.load_prompt_from_file(self.config_manager.get_prompt_config().get('llama')))

        history_context = ""
        if history and isinstance(history, list):
            num_history = 10
            print(f"🔍 [Llama] History 처리 시작: {len(history)}개 메시지 중 최근 {num_history}개 처리")

            for i, msg in enumerate(history[-num_history:]):
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    if role == "user":
                        history_context += f"<human>: {content}\n\n"

                    elif role == "assistant":
                        history_context += f"<assistant>: {str(content)}\n\n"

            print(f"🔥 [Llama] History Context 생성 완료: {len(history_context)} chars")

        formatted_prompt = self.create_clean_prompt(query, context, history_context, selected_intent.upper(), response_format, request)

        chat_session.add_message("user", query)

        try:
            if self.use_local_model and self.local_model:
                full_response = self.local_model.generate_complete(
                    prompt=formatted_prompt,
                    max_tokens=1024,
                    temperature=0.0,
                    top_p = 1.0
                )

                # usage 집계 — 인프로세스 create_completion(str 반환)은 usage 를 안 주므로
                # 토크나이저로 실측해 채운다. 안 채우면 이 핸들러만 0/0 (핸들러 비대칭).
                try:
                    self._usage_tracker.track({"usage": {
                        "prompt_tokens": len(self.local_model.llm.tokenize(formatted_prompt.encode('utf-8'))),
                        "completion_tokens": len(self.local_model.llm.tokenize(full_response.encode('utf-8'))),
                    }})
                except Exception:
                    pass  # 집계 실패가 응답을 막으면 안 된다

                if (request.get("type", "").upper() == "ACTION" and
                    ('"action": "clarify"' in full_response or '"action":"clarify"' in full_response)):
                    original_intent = request.get("type", "ACTION")
                    print(f"🔄 Clarify 응답 감지 (Llama Local): {original_intent} 의도 유지")

                    try:
                        clarify_data = json.loads(full_response)
                        if isinstance(clarify_data, dict) and "message" in clarify_data:
                            full_response = clarify_data["message"]
                            print(f"📝 Clarify 메시지 추출 (Llama Local): {full_response}")

                            if "original_intent" in clarify_data:
                                request["original_intent"] = clarify_data["original_intent"]
                                print(f"💾 원본 의도 저장 (Llama Local): {clarify_data['original_intent']}")
                    except json.JSONDecodeError:
                        match = re.search(r'"message":\s*"([^"]*)"', full_response)
                        if match:
                            full_response = match.group(1)
                            print(f"📝 Clarify 메시지 추출 (정규식, Llama Local): {full_response}")

            else:
                request_body = {
                    "prompt": formatted_prompt,
                    "max_gen_len": 1024,
                    "temperature": 0.01,
                    "top_p": 0.9
                }

                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json"
                )
                response_body = json.loads(response['body'].read())
                full_response = response_body.get('generation', '')

                # usage 집계 — Bedrock llama 응답의 자체 카운트를 그대로 옮긴다
                self._usage_tracker.track({"usage": {
                    "prompt_tokens": response_body.get("prompt_token_count", 0),
                    "completion_tokens": response_body.get("generation_token_count", 0),
                }})

            if (request.get("type", "").upper() == "ACTION" and
                ('"action": "clarify"' in full_response or '"action":"clarify"' in full_response)):
                original_intent = request.get("type", "ACTION")
                print(f"🔄 Clarify 응답 감지 (Llama): {original_intent} 의도 유지")

                try:
                    clarify_data = json.loads(full_response)
                    if isinstance(clarify_data, dict) and "message" in clarify_data:
                        full_response = clarify_data["message"]
                        print(f"📝 Clarify 메시지 추출 (Llama): {full_response}")
                except json.JSONDecodeError:
                    match = re.search(r'"message":\s*"([^"]*)"', full_response)
                    if match:
                        full_response = match.group(1)
                        print(f"📝 Clarify 메시지 추출 (정규식, Llama): {full_response}")

            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                full_response = self._remove_markdown_formatting(full_response)

            chat_session.add_message("assistant", full_response)

            end_time = time.time()

            if connection and user_id:
                self.logger.log(user_id, query, full_response, f"{end_time - start_time:.2f}",
                                self.last_sources, concepts, getattr(self, 'last_related_concepts', []), connection)

            self.last_response_time = end_time - start_time
            self.last_concepts = concepts if concepts else []

            request_type = request.get("type", "").upper()
            selected_intent_upper = selected_intent.upper() if selected_intent else ""
            is_post_action = (request_type in ["POST_ACTION", "POST-ACTION"] or
                            selected_intent_upper in ["POST_ACTION", "POST-ACTION"])

            if is_post_action:
                try:

                    import json
                    json_response = json.loads(full_response)






                    if 'msg' in json_response:
                        full_response = json_response['msg']
                        print(f"📝 POST_ACTION msg 추출: {full_response}")
                    else:
                        print(f"📝 POST_ACTION JSON에 msg 필드 없음")

                except json.JSONDecodeError:
                    print(f"📝 POST_ACTION 응답이 JSON이 아님, 원본 사용")

            final_response = self._fix_unicode_response(full_response)

            if (self.config_manager.get_option_config().get('show_metadata') and not is_post_action):
                detected_intent = selected_intent or (intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown')
                if detected_intent != 'unknown':
                    intent_text = f"🎯 **의도:** **{detected_intent.upper()}**"
                    final_response = re.sub(r"🎯 \*\*의도:\*\* \*\*.*?\*\*", intent_text, final_response)

                    if intent_text not in final_response:
                        final_response += f"\n\n{intent_text}"

                if source_files:
                    refs_text = f"📚 **참조 문서** ({len(source_files)}개):" + ''.join([f"\n{i+1}. `{src}`" for i, src in enumerate(source_files)])
                else:
                    refs_text = "📚 **참조 문서:** 없음"

                final_response = re.sub(r"📚 \*\*참조 문서\*\*.*?(?=(\n⏱️|\n🎯|\Z))", refs_text, final_response, flags=re.DOTALL)

                if refs_text not in final_response:
                    final_response += f"\n\n{refs_text}"

                time_text = f"⏱️ 답변 시간: {end_time - start_time:.2f}초"
                final_response = re.sub(r"⏱️ 답변 시간: .*?초", time_text, final_response)

                if time_text not in final_response:
                    final_response += f"\n\n{time_text}"

            self.system_prompt = original_prompt

            return {
                "answer": final_response,
                "intent": (selected_intent or intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown').upper(),
                "intent_info": intent_info,
                "sources": source_files,
                "response_time": end_time - start_time,
                "question_type": question_type,
                "usage": self._usage_tracker.to_dict(),
            }

        except Exception as e:
            self.system_prompt = original_prompt

            logger.error(f"Llama generate_complete [{type(e).__name__}]: {e}")
            return {
                "answer": classify_error(e),
                "intent": None,
                "sources": [],
                "response_time": time.time() - start_time,
                "question_type": question_type
            }

    async def generate_stream(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None,
                        context=None, source_files=None, related_concepts=None, concepts=None, intent_info=None) -> AsyncGenerator[str, None]:
        start_time = time.time()

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            history = request.get("history", [])
            response_format = request.get("response_format", False)

        else:
            query = str(request)
            question_type = None
            response_format = False

        if not self.available:
            yield "Llama 모델을 사용할 수 없습니다."
            return

        chat_session = None
        if hasattr(self.rag_system, 'llm_handler') and user_id:
            if hasattr(self.rag_system.llm_handler, 'get_chat_session'):
                chat_session = self.rag_system.llm_handler.get_chat_session(user_id, channel_id)

        if not chat_session:
            chat_session = self.chat_session

        if context is None:
            if self.rag_system:
                try:
                    result = await self.get_related_documents(request)
                    if len(result) >= 6:
                        context, source_files, related_concepts, concepts, intent_info, reference_summary = result
                    elif len(result) >= 5:
                        context, source_files, related_concepts, concepts, intent_info = result
                    else:
                        context, source_files, related_concepts, concepts = result
                    self.last_sources = source_files

                    if context:
                        logger.info(f"참고 문서: {len(source_files)}개")
                        logger.info(f"참고 문서 목록: {source_files}")

                    if related_concepts and self.use_kg:
                        concept_context = "\n\n관련 개념:\n"
                        for concept_info in related_concepts[:3]:
                            concept = concept_info.get('concept')
                            concept_context += f"- {concept}\n"
                        context += concept_context
                except Exception as e:
                    logger.error(f"RAG 검색 중 오류: {e}")
                    context, source_files, related_concepts, concepts = "", [], [], []
        else:
            self.last_sources = source_files if source_files else []

            if related_concepts and self.use_kg:
                concept_context = "\n\n관련 개념:\n"
                for concept_info in related_concepts[:3]:
                    concept = concept_info.get('concept')
                    concept_context += f"- {concept}\n"
                context += concept_context

        chat_history = chat_session.format_chat_history_for_llama(max_tokens=1000)

        formatted_prompt = self.create_clean_prompt(query, context, chat_history, question_type.upper() if question_type else "", response_format, request)

        chat_session.add_message("user", query)

        try:
            full_response = ""

            if self.use_local_model and self.local_model:
                logger.info("🎮 로컬 Llama 모델 사용 중...")

                loop = asyncio.get_event_loop()
                full_response = await loop.run_in_executor(
                    None,
                    lambda: self.local_model.generate_complete(
                        prompt=formatted_prompt,
                        max_tokens=1024,
                        temperature=0.0,
                        top_p = 1.0
                    )
                )
                yield full_response

            else:
                logger.info("🌐 AWS Bedrock 사용 중...")
                request_body = {
                    "prompt": formatted_prompt,
                    "max_gen_len": 1024,
                    "temperature": 0.0,
                    "top_p": 1.0
                }

                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json"
                )
                response_body = json.loads(response['body'].read())
                full_response = response_body.get('generation', '')

                if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                    full_response = self._remove_markdown_formatting(full_response)

                yield full_response

            chat_session.add_message("assistant", full_response)

            request_type_stream = request.get("type", "").upper()
            question_type_upper = question_type.upper() if question_type else ""
            is_post_action_stream = (request_type_stream in ["POST_ACTION", "POST-ACTION"] or
                                   question_type_upper in ["POST_ACTION", "POST-ACTION"])

            if (self.config_manager.get_option_config().get('show_metadata') and not is_post_action_stream):
                detected_intent = question_type or (intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown')
                if detected_intent != 'unknown':
                    yield f"\n\n🎯 **의도:** **{detected_intent.upper()}**"

                if source_files:
                    yield f"\n\n📚 **참조 문서** ({len(source_files)}개):"
                    for i, src in enumerate(source_files):
                        if isinstance(src, str):
                            yield f"\n{i+1}. `{src}`"
                        elif isinstance(src, dict):
                            file_name = src.get('file', '')
                            score = src.get('score', 0.0)
                            yield f"\n{i+1}. `{file_name}` (유사도: {score:.3f})"
                        elif isinstance(src, tuple) and len(src) >= 3:
                            yield f"\n{i+1}. `{src[0]}` (유사도: {src[2]:.3f})"
                else:
                    yield f"\n\n📚 **참조 문서:** 없음"

                end_time = time.time()
                yield f"\n\n⏱️ 답변 시간: {end_time - start_time:.2f}초"

            self.last_response_time = end_time - start_time if 'end_time' in locals() else 0
            self.last_concepts = concepts if concepts else []

        except Exception as e:
            logger.error(f"Llama generate_stream [{type(e).__name__}]: {e}")
            yield classify_error(e)

    def _fix_unicode_response(self, response: str) -> str:
        try:
            import json
            parsed = json.loads(response)

            def fix_msg_field(obj):
                if isinstance(obj, dict) and 'msg' in obj:

                    msg = obj['msg']
                    if '\\u' in msg:
                        import codecs
                        msg = codecs.decode(msg, 'unicode_escape')
                    elif 'í' in msg or 'ì' in msg:
                        msg = msg.encode('latin-1').decode('utf-8')
                    obj['msg'] = msg
                return obj

            if isinstance(parsed, list):
                parsed = [fix_msg_field(item) for item in parsed]
            elif isinstance(parsed, dict):
                parsed = fix_msg_field(parsed)

            return json.dumps(parsed, ensure_ascii=False, indent=2)

        except Exception:
            return response

    async def generate_complete_error(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> str:

        if not self.available:
            return "Llama 모델을 사용할 수 없습니다."

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
        else:
            query = str(request)
            question_type = None

        concepts = []

        try:
            if self.use_local_model and self.local_model:
                full_response = self.local_model.generate_complete(
                    prompt=query,
                    max_tokens=1024,
                    temperature=0.01
                )
            else:
                request_body = {
                    "prompt": query,
                    "max_gen_len": 1024,
                    "temperature": 0,
                    "top_p": 1.0
                }

                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json"
                )
                response_body = json.loads(response['body'].read())
                full_response = response_body.get('generation', '')

            return full_response

        except Exception as e:
            logger.error(f"Llama generate_complete_error [{type(e).__name__}]: {e}")
            return classify_error(e)

    # process_question_error: BaseModelHandler default 사용.

    async def get_complete_answer(self, request, user_id=None, channel_id=None, source='API', thread_ts=None):
        return await self.generate_complete(request, user_id, channel_id, source, thread_ts)

    # process_question: BaseModelHandler default 사용 (generate_stream wrap).

    def _apply_response_format(self, system_prompt: str, response_format: bool = False) -> str:

        if response_format is True:
            format_instruction = """

                FORMAT ENHANCEMENT RULES:
                - Use emoji whenever possible to make responses more engaging
                - Use markdown formatting for better readability (bold, italic, tables, code blocks)
                - Use rich text formatting including **bold**, *italic*, `code`, and tables
                - Make responses visually appealing and easy to read
            """

        else:
            format_instruction = """

                PLAIN TEXT RESTRICTIONS:
                CLEAN TEXT MODE - NO VISUAL EMPHASIS:
                - NEVER use **bold**, *italic*, or visual emphasis markdown
                - NEVER use emojis or special unicode characters
                - NEVER use # headers or complex formatting
                - Allow simple numbered lists (1. 2. 3.) with proper indentation
                - Allow sub-items with indentation (   - sub-item)
                - Allow basic text structure and spacing
                - For emphasis: use CAPITAL LETTERS or descriptive words
                - Keep language policy intact (use 해요체 for Korean)
                - Focus on clear, structured content without visual distractions
            """


        return system_prompt + format_instruction

    def _remove_markdown_formatting(self, text: str) -> str:
        import re

        try:
            if '```query' in text and text.count('```') % 2 != 0:
                return text  

            query_blocks = []
            def save_query_block(match):
                query_blocks.append(match.group(0))
                return f"__QUERY_BLOCK_{len(query_blocks)-1}__"

            text = re.sub(r'```query[\s\S]*?```', save_query_block, text)

            def replace_code_block(match):
                code_content = match.group(1)
                return code_content.strip()

            text = re.sub(r'```(?:[a-zA-Z]*\n)?([\s\S]*?)```', replace_code_block, text)

            for i, block in enumerate(query_blocks):
                text = text.replace(f"__QUERY_BLOCK_{i}__", block)

            if not query_blocks:
                text = re.sub(r'`([^`]+)`', r'\1', text)

            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'__(.+?)__', r'\1', text)
            text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
            text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', text)

            text = re.sub(r'^#{1,6}\s+(.*)$', r'\1', text, flags=re.MULTILINE)

            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

            emoji_pattern = re.compile("["
                                    "\U0001F600-\U0001F64F"
                                    "\U0001F300-\U0001F5FF"
                                    "\U0001F680-\U0001F6FF"
                                    "\U0001F1E0-\U0001F1FF"
                                    "\U00002600-\U000026FF"
                                    "\U00002700-\U000027BF"
                                    "\U0001F900-\U0001F9FF"
                                    "\U0001FA70-\U0001FAFF"
                                    "\U00002328"
                                    "\U000023CF"
                                    "]+", flags=re.UNICODE)
            text = emoji_pattern.sub('', text)

            lines = text.split('\n')
            clean_lines = []
            in_table = False

            for line in lines:
                if re.match(r'^[\s\|]*[-:]{3,}[\s\|]*[-:\s\|]+', line) or re.match(r'^\s*\|?\s*[-:]+\s*\|\s*[-:]+', line):
                    in_table = True
                    continue
                elif '|' in line and in_table:
                    cells = [cell.strip() for cell in line.split('|')]
                    cells = [c for c in cells if c]
                    if cells:
                        clean_lines.append('  '.join(cells))
                else:
                    if '|' not in line:
                        in_table = False
                    clean_lines.append(line)

            text = '\n'.join(clean_lines)

            lines = text.split('\n')
            processed_lines = []
            for line in lines:
                if re.match(r'^[\s]*[-*+]\s+', line):
                    content = re.sub(r'^[\s]*[-*+]\s+', '', line)
                    processed_lines.append(f'- {content}')
                else:
                    processed_lines.append(line)
            text = '\n'.join(processed_lines)

            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
            text = text.strip()

            return text

        except Exception as e:
            logger.warning(f"마크다운 제거 중 오류: {e}")
            return text

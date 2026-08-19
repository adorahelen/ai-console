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
from handler_base import BaseModelHandler, UsageTracker, sanitize_history, classify_error
import logging

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama, llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LocalQwenModel:
    def __init__(self):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python 라이브러리가 필요합니다.")

        from config_utils import ConfigManager
        from pathlib import Path

        config_manager = ConfigManager()
        paths_config = config_manager.get_paths_config()

        model_path_from_config = paths_config.get('qwen_model_path')

        if model_path_from_config:
            self.model_path = str(Path(model_path_from_config).resolve())
        else:
            self.model_path = "/data/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Qwen 모델 파일을 찾을 수 없습니다: {self.model_path}")

        context_size = int(os.getenv("LLAMA_CONTEXT_SIZE", "32768"))

        logger.info(f"🚀 로컬 Qwen3.5 모델 로드 중: {self.model_path}")
        logger.info(f"설정: 컨텍스트={context_size}")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=context_size,
                n_threads=8,
                n_gpu_layers=-1,
                n_batch=2048,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                logits_all=False,
                embedding=False,
                mul_mat_q=True,
            )
            logger.info(f"✅ Qwen3.5 모델 로드 완료")

        except Exception as e:
            logger.error(f"❌ Qwen3.5 모델 로드 실패: {str(e)}")
            raise

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01, top_p: float = 0.9) -> Dict:
        self.llm.reset()
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.1,
            top_k=40,
            stop=[
                "<|im_end|>",
                "<|endoftext|>",
            ],
            echo=False
        )

        return {
            "generation": response["choices"][0]["text"],
            "prompt_token_count": len(self.llm.tokenize(prompt.encode('utf-8'))),
            "generation_token_count": len(self.llm.tokenize(response["choices"][0]["text"].encode('utf-8'))),
            "stop_reason": "stop"
        }

    def generate_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01, top_p: float = 0.9, repeat_penalty: float = 1.1, top_k: int = 40):
        self.llm.reset()
        buffer = ""
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stop=[
                "<|im_end|>",
                "<|endoftext|>",
            ],
            echo=False,
            stream=True
        ):
            token = chunk["choices"][0]["text"]
            buffer += token

            if '\n' in buffer:
                lines = buffer.split('\n')
                for i in range(len(lines) - 1):
                    yield lines[i] + '\n'
                buffer = lines[-1]
            elif any(char in buffer for char in ['. ', '! ', '? ', ': ']):
                yield buffer
                buffer = ""
            elif len(buffer) > 50 and ' ' in buffer:
                last_space = buffer.rfind(' ')
                yield buffer[:last_space + 1]
                buffer = buffer[last_space + 1:]

        if buffer:
            yield buffer

    def generate_complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01, top_p: float = 0.9) -> str:
        try:
            self.llm.reset()
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.1,
                top_k=40,
                stop=[
                    "<|im_end|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )
            result = response["choices"][0]["text"].strip()

            # ChatML 태그 정리
            unwanted_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
            for token in unwanted_tokens:
                result = result.replace(token, "")

            # <think>...</think> 블록 제거 (완성/미완성 모두)
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            result = re.sub(r'<think>.*', '', result, flags=re.DOTALL).strip()

            return result.strip()

        except Exception as e:
            logger.error(f"Qwen 생성 오류: {e}")
            return ""


class QwenHandler(BaseModelHandler):
    registry_key = "qwen"  # handler_registry.HANDLER_CLASSES 키와 일치
    is_local = True         # 로컬 GGUF / llama-server

    def build_agent_prompt(self, messages, options=None):
        """OpenAI messages -> Qwen chat template. agent_complete(위저드/completions2)용."""
        parts = []
        for m in messages:
            parts.append("<|im_start|>%s\n%s<|im_end|>" % (m.get("role","user"), m.get("content","")))
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _agent_generate(self, prompt):
        if not getattr(self, "local_model", None):
            raise RuntimeError("qwen local_model 미로드")
        return self.local_model.generate_complete(prompt, 2048, 0.3, 0.9)


    def __init__(self, rag_system=None, use_kg: bool = True, intent_analyzer=None):
        super().__init__(rag_system, use_kg, intent_analyzer)

        self.config_manager = ConfigManager()

        self.handler_type = "qwen"
        self.model_name = 'qwen3.5-9b'
        self.local_model = None
        self.available = False

        prompt_path = self.config_manager.get_prompt_config().get('qwen')
        if not prompt_path:
            prompt_path = self.config_manager.get_prompt_config().get('gpt_oss')

        self.system_prompt = self.load_prompt_from_file(prompt_path)

        self.logger = ChatLogger(self.model_name)

        self._initialize_model()

    def _initialize_model(self):
        try:
            self.local_model = LocalQwenModel()
            self.available = True
            logger.info("✅ 로컬 Qwen3.5 모델 초기화 완료")
        except Exception as e:
            logger.error(f"❌ Qwen3.5 모델 초기화 실패: {str(e)}")
            self.local_model = None
            self.available = False

    def _perform_translation(self, query: str, mode: str, preserved: dict = None) -> str:
        """Qwen3.5 모델을 사용한 번역 (별도 번역 LLM 불필요)"""
        try:
            if not self.available:
                logger.warning("Qwen 모델이 사용 불가능하여 사전 번역 사용")
                return self._enhanced_dictionary_translate(query)

            print(f"🔍 번역 모드: {mode} (Qwen3.5)")
            # 번역 시스템 프롬프트는 카트리지가 정의 — [prompts] translation_* (공유 파일)
            from config_utils import load_translation_prompt
            system_instructions = load_translation_prompt(mode)
            prompt = (
                f"<|im_start|>system\n{system_instructions}<|im_end|>\n"
                f"<|im_start|>user\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n</think>\n"
            )

            translation = self.local_model.generate_complete(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )

            def _restore_for_log(text):
                if preserved:
                    for key, token in preserved.items():
                        text = text.replace(key, token)
                return text

            print(f"========= 정리 전 번역 내용 =========\n{_restore_for_log(translation)}\n===============================")

            translation = translation.strip()
            translation = re.sub(r'^(Keywords:|키워드:|English:|Translation:)', '', translation).strip()
            translation = translation.split('\n')[0].strip()
            print(f"========= 정리 후 번역 내용 =========\n{_restore_for_log(translation)}\n===============================")

            if mode == 'CVE':
                return translation
            else:
                if translation:
                    original_lower = query.lower()
                    translation_lower = translation.lower()

                    critical_mappings = [
                        (['공격한', '공격하는', '공격'], 'attack'),
                        (['로그인', '접속'], 'login'),
                        (['실패한', '실패'], 'failure'),
                        (['침입한', '침입'], 'intrusion'),
                        (['차단한', '차단'], 'block'),
                        (['탐지한', '탐지'], 'detect')
                    ]

                    for korean_terms, english_term in critical_mappings:
                        if any(term in original_lower for term in korean_terms):
                            if english_term not in translation_lower:
                                translation = f"{english_term} {translation}"
                                logger.warning(f"누락된 핵심 키워드 '{english_term}' 추가됨")

                    logger.info(f"키워드 번역: '{_restore_for_log(query)}' → '{_restore_for_log(translation)}'")

                    return translation
                else:
                    return self._enhanced_dictionary_translate(query)

        except Exception as e:
            logger.error(f"Qwen 번역 오류: {e}")
            return self._enhanced_dictionary_translate(query)

    async def generate_complete(self, request, user_id=None, channel_id=None, connection=None, thread_ts=None,
                          context=None, source_files=None, related_concepts=None, concepts=None, intent_info=None) -> Dict:
        start_time = time.time()
        self._usage_tracker = UsageTracker(model="qwen3.5-9b", user_id=user_id or "")

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            history = request.get("history", [])
            response_format = request.get("response_format", False)

            query_results = request.get("query_results", request.get("results", None))
            previous_query = request.get("previous_query", None)

            if question_type and question_type.upper() in ["POST_ACTION", "POST-ACTION"]:
                print(f"🔍 POST_ACTION 요청 분석:")
                print(f"   - query: {query}")
                print(f"   - query_results: {str(query_results)[:200] if query_results else 'None'}")
                print(f"   - previous_query: {previous_query}")
                print(f"   - history 개수: {len(history) if history else 0}")

        else:
            query = str(request)
            question_type = None
            history = None
            response_format = False

        if not self.available:
            return {
                "answer": "Qwen3.5 모델을 사용할 수 없습니다.",
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
                        reference_summary = None
                    else:
                        context, source_files, related_concepts, concepts = result
                        intent_info = None
                        reference_summary = None

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

        self._select_intent_prompt(selected_intent)  # complete: PLAYBOOK 분기 포함

        history_context = sanitize_history(history, current_query=query)

        chat_session.add_message("user", query)

        formatted_prompt = self.create_clean_prompt(
            query=query,
            context=context,
            chat_history=history_context,
            intent=selected_intent.upper() if selected_intent else "",
            response_format=response_format,
            question_type=question_type,
            request=request
        )

        # 프롬프트 로깅
        import time as time_module
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/qwen/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write(f"Method: generate_complete\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE QWEN PROMPT STRUCTURE\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"📝 Formatted Prompt:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{formatted_prompt}\n")
                f.write("\n" + "=" * 60 + "\n\n")
                f.write(f"System Prompt Length: {len(self.system_prompt)} chars\n")
                f.write(f"Context Length: {len(context)} chars\n")
                f.write(f"History Context Length: {len(history_context)} chars\n")
                f.write(f"Full Prompt Length: {len(formatted_prompt)} chars\n")
                f.write(f"\nRAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")
                f.write(f"Related Concepts: {len(related_concepts)}\n")
                for k, concept in enumerate(related_concepts[:5]):
                    concept_name = concept.get('concept', 'Unknown') if isinstance(concept, dict) else str(concept)
                    f.write(f"  {k+1}. {concept_name}\n")
        except Exception as log_e:
            print(f"⚠️ Qwen 로그 저장 실패 (generate_complete): {log_e}")

        try:
            # 토큰 사용량 분석
            def count_tokens(text):
                return len(self.local_model.llm.tokenize(text.encode('utf-8')))

            print(f"📊 토큰 사용량 분석:")
            system_tokens = count_tokens(self.system_prompt)
            context_tokens = count_tokens(context)
            history_tokens = count_tokens(history_context)
            query_tokens = count_tokens(query)
            total_prompt_tokens = count_tokens(formatted_prompt)

            print(f"  - 시스템 프롬프트: {system_tokens:,} 토큰")
            print(f"  - RAG 컨텍스트: {context_tokens:,} 토큰")
            print(f"  - 대화 히스토리: {history_tokens:,} 토큰")
            print(f"  - 사용자 질문: {query_tokens:,} 토큰")
            print(f"  - 전체 프롬프트: {total_prompt_tokens:,} 토큰")
            print(f"  - 컨텍스트 한계: 32,768 토큰")

            if total_prompt_tokens > 32768:
                print(f"❌ 토큰 초과! ({total_prompt_tokens:,} > 32,768)")
                return {
                    "answer": f"프롬프트가 너무 깁니다. ({total_prompt_tokens:,} 토큰 > 32,768 한계)",
                    "intent": None,
                    "sources": [],
                    "response_time": 0,
                    "question_type": question_type
                }

            start_model_time = time.time()
            print(f"🚀 Qwen3.5 응답 생성 시작 (generate_complete)...")
            self.local_model.llm.reset()
            response = self.local_model.llm.create_completion(
                prompt=formatted_prompt,
                max_tokens=8192,
                temperature=0.1,
                top_p=0.8,
                repeat_penalty=1.2,
                top_k=20,
                stop=[
                    "<|im_end|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )
            end_model_time = time.time()
            print(f"✅ Qwen3.5 응답 생성 완료 {end_model_time - start_model_time:.2f}초 (generate_complete)")

            full_response = response["choices"][0]["text"].strip()

            # <think>...</think> 블록 제거
            full_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

            print(f"📝 전체 응답 (generate_complete): {full_response}")

            # Clarify 응답 처리
            if (request.get("type", "").upper() == "ACTION" and
                ('"action": "clarify"' in full_response or '"action":"clarify"' in full_response)):
                original_intent = request.get("type", "ACTION")
                print(f"🔄 Clarify 응답 감지 (Qwen): {original_intent} 의도 유지")

                try:
                    clarify_data = json.loads(full_response)
                    if isinstance(clarify_data, dict) and "message" in clarify_data:
                        full_response = clarify_data["message"]
                except json.JSONDecodeError:
                    match = re.search(r'"message":\s*"([^"]*)"', full_response)
                    if match:
                        full_response = match.group(1)

            # ChatML 태그 정리
            final_response = self._get_clean_message(full_response)
            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                final_response = self._remove_markdown_formatting(final_response)

            chat_session.add_message("assistant", final_response)

            end_time = time.time()

            if connection and user_id:
                self.logger.log(user_id, query, final_response, f"{end_time - start_time:.2f}",
                                self.last_sources, concepts, getattr(self, 'last_related_concepts', []), connection)

            self.last_response_time = end_time - start_time
            self.last_concepts = concepts if concepts else []

            request_type = request.get("type", "").upper()
            selected_intent_upper = selected_intent.upper() if selected_intent else ""
            is_post_action = (request_type in ["POST_ACTION", "POST-ACTION"] or
                            selected_intent_upper in ["POST_ACTION", "POST-ACTION"])

            processed_response = self._fix_unicode_response(final_response)

            show_metadata_config = self.config_manager.get_option_config().get('show_metadata')

            if (show_metadata_config and not is_post_action):
                detected_intent = selected_intent or (intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown')
                if detected_intent != 'unknown':
                    intent_text = f"🎯 **의도:** **{detected_intent.upper()}**"
                    if intent_text not in processed_response:
                        processed_response += f"\n\n{intent_text}"

                if source_files:
                    refs_text = f"📚 **참조 문서** ({len(source_files)}개):"
                    refs_text += ''.join([f"\n{i+1}. `{src}`" for i, src in enumerate(source_files)])
                else:
                    refs_text = "📚 **참조 문서:** 없음"

                if refs_text not in processed_response:
                    processed_response += f"\n\n{refs_text}"

                time_text = f"⏱️ 답변 시간: {end_time - start_time:.2f}초"
                if time_text not in processed_response:
                    processed_response += f"\n\n{time_text}"

            self.system_prompt = original_prompt

            return {
                "answer": processed_response,
                "intent": (selected_intent or intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown').upper(),
                "intent_info": intent_info,
                "sources": source_files,
                "response_time": end_time - start_time,
                "question_type": question_type,
                "usage": self._usage_tracker.to_dict(),
            }

        except Exception as e:
            self.system_prompt = original_prompt

            logger.error(f"Qwen generate_complete [{type(e).__name__}]: {e}")
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
            response_format = request.get("response_format", False)
            history = request.get("history", [])
        else:
            query = str(request)
            question_type = None
            response_format = False

        if not self.available:
            yield "Qwen3.5 모델을 사용할 수 없습니다."
            return

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
        else:
            self.last_sources = source_files if source_files else []

            if related_concepts and self.use_kg:
                concept_context = "\n\n관련 개념:\n"
                for concept_info in related_concepts[:3]:
                    concept = concept_info.get('concept')
                    concept_context += f"- {concept}\n"
                context += concept_context

        original_prompt = self.system_prompt

        selected_intent = question_type or (intent_info.get('primary_intent', '') if intent_info else '')

        self._select_intent_prompt(selected_intent, allow_playbook=False)  # stream: PLAYBOOK 분기 없음(기존 동작 보존)

        chat_session.add_message("user", query)

        formatted_prompt = self.create_clean_prompt(query, context, "", selected_intent.upper() if selected_intent else "", response_format, question_type, request)

        # 프롬프트 로깅
        import time as time_module
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/qwen/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE QWEN PROMPT STRUCTURE\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"📝 Formatted Prompt:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{formatted_prompt}\n")
                f.write("\n" + "=" * 60 + "\n\n")
                f.write(f"System Prompt Length: {len(self.system_prompt)} chars\n")
                f.write(f"Context Length: {len(context)} chars\n")
                f.write(f"Full Prompt Length: {len(formatted_prompt)} chars\n")
                f.write(f"\nRAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")
        except Exception as log_e:
            print(f"⚠️ Qwen 로그 저장 실패: {log_e}")

        try:
            full_response = ""

            for chunk in self.local_model.generate_stream(
                prompt=formatted_prompt,
                max_tokens=2048,
                temperature=0.1,
                top_p=0.8,
                repeat_penalty=1.2,
                top_k=20
            ):
                if chunk:
                    full_response += chunk

                    # <think> 블록이 끝나기 전까지는 출력하지 않음
                    if '<think>' in full_response and '</think>' not in full_response:
                        continue

                    # <think> 블록이 끝났으면 제거 후 출력
                    if '</think>' in full_response and not hasattr(self, '_think_stripped'):
                        self._think_stripped = True
                        full_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
                        if full_response:
                            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                                yield self._remove_markdown_formatting(full_response)
                            else:
                                yield full_response
                        continue

                    if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                        yield self._remove_markdown_formatting(chunk)
                    else:
                        yield chunk

            # cleanup
            if hasattr(self, '_think_stripped'):
                del self._think_stripped

            chat_session.add_message("assistant", full_response)

            show_metadata_stream = self.config_manager.get_option_config().get('show_metadata')
            request_type_stream = request.get("type", "").upper() if isinstance(request, dict) else ""
            question_type_upper = question_type.upper() if question_type else ""

            is_post_action_stream = (request_type_stream in ["POST_ACTION", "POST-ACTION"] or
                                   question_type_upper in ["POST_ACTION", "POST-ACTION"])

            if (show_metadata_stream and not is_post_action_stream):
                if intent_info:
                    detected_intent = intent_info.get('primary_intent', 'unknown')
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
                else:
                    yield f"\n\n📚 **참조 문서:** 없음"

                end_time = time.time()
                yield f"\n\n⏱️ 답변 시간: {end_time - start_time:.2f}초"

            self.last_response_time = time.time() - start_time
            self.last_concepts = concepts if concepts else []

            self.system_prompt = original_prompt

        except Exception as e:
            self.system_prompt = original_prompt

            error_message = f"Qwen3.5 스트리밍 생성 중 오류: {str(e)}"
            logger.error(error_message)
            yield error_message

    def _get_clean_message(self, response: str) -> str:
        """ChatML 태그에서 순수 응답 추출"""
        clean = response

        # <think>...</think> 블록 제거
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL)

        # ChatML 태그 정리
        chatml_tags = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        for tag in chatml_tags:
            clean = clean.replace(tag, "")

        # 역할 태그 정리 (응답에 포함될 수 있음)
        clean = re.sub(r'^(system|user|assistant)\s*\n?', '', clean.strip())

        return clean.strip()

    # _clean_prompt_content: BaseModelHandler default 사용 (superset 토큰 strip).

    def _select_intent_prompt(self, selected_intent, allow_playbook: bool = True) -> None:
        """selected_intent 에 따라 self.system_prompt 를 의도별 프롬프트로 교체.

        generate_complete / generate_stream 가 공유 (전엔 두 메서드에 같은 블록이 복붙돼
        있었음). complete 만 PLAYBOOK 분기 포함(allow_playbook=True), stream 은 미포함
        (allow_playbook=False) — 기존 동작 보존용 파라미터.
        """
        if not (self.config_manager.get_option_config().get('user_intent_prompt') == 'True'
                and selected_intent):
            return

        intent_upper = selected_intent.upper()
        prompt_cfg = self.config_manager.get_prompt_config()

        if intent_upper in ['POST_ACTION', 'POST-ACTION']:
            post_action_path = prompt_cfg.get('post_action')
            if post_action_path:
                gpt_prompt = self.load_prompt_from_file(post_action_path)
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
        elif intent_upper == 'ACTION':
            gpt_prompt = self.load_prompt_from_file(prompt_cfg.get('action_gpt'))
            self.system_prompt = self._clean_prompt_content(gpt_prompt)
        elif intent_upper == 'PLAN':
            gpt_prompt = self.load_prompt_from_file(prompt_cfg.get('plan_gpt'))
            self.system_prompt = self._clean_prompt_content(gpt_prompt)
        elif allow_playbook and intent_upper == 'PLAYBOOK':
            gpt_prompt = self.load_prompt_from_file(prompt_cfg.get('playbook'))
            self.system_prompt = self._clean_prompt_content(gpt_prompt)
        else:
            gpt_prompt = self.load_prompt_from_file(prompt_cfg.get('qna_gpt'))
            self.system_prompt = self._clean_prompt_content(gpt_prompt)

    def create_clean_prompt(self, query: str, context: str = "", chat_history: str = "", intent: str = "", response_format: bool = True, question_type: str = None, request: dict = None) -> str:
        from datetime import datetime, timedelta
        now = datetime.now()
        current_date = now.strftime("%Y%m%d")
        current_readable = now.strftime("%Y년 %m월 %d일 (%A)")

        system_content = self.system_prompt.replace(
            "current date (20250629)",
            f"current date ({current_date})"
        ).replace(
            "current date is 20250629",
            f"current date is {current_date}"
        )

        if question_type and question_type.upper() == "QNA":
            system_content = self._apply_response_format(system_content, response_format)

        system_content += "\n\n[RESPONSE STYLE]"
        system_content += "\nDo not show your thinking process or intermediate steps."
        system_content += "\nDo not use phrases like 'Let me think...', 'First, I need to...', 'Step 1:', etc."
        system_content += "\nProvide the final answer directly without showing how you arrived at it."
        system_content += "\nKeep the answer comprehensive but skip the reasoning chain."
        system_content += "\nDo NOT output <think> tags or any thinking blocks."

        system_content += f"\n\n[CURRENT DATE & CALCULATION GUIDE]"
        system_content += f"\n현재 날짜: {current_date} ({current_readable})"
        system_content += f"\n"
        system_content += f"\n⚠️ IMPORTANT: When user asks for 'recent X days' or 'last X days', calculate BACKWARDS from today:"
        system_content += f"\n- Recent 7 days = {(now - timedelta(days=6)).strftime('%Y%m%d')} to {current_date}"
        system_content += f"\n- Recent 30 days = {(now - timedelta(days=29)).strftime('%Y%m%d')} to {current_date}"
        system_content += f"\n- Yesterday = {(now - timedelta(days=1)).strftime('%Y%m%d')}"
        system_content += f"\n- Last week = {(now - timedelta(days=6)).strftime('%Y%m%d')} to {current_date}"
        system_content += f"\n"
        system_content += f"\n🚫 Do NOT use future dates unless explicitly asked for future periods!"

        if chat_history.strip():
            system_content += """

[CONVERSATION HISTORY CONTEXT]
⚠️ CRITICAL: The user is continuing from previous conversation.
- When user says "마지막" (last), "이전" (previous), "위에서" (above), refer to YOUR previous responses
- When user mentions "그룹" (group), "항목" (item), check what you listed in previous answers
- Prioritize conversation context over keyword-based document search
- If user refers to something from previous messages, DO NOT search for new information unless explicitly asked
"""

        user_content = f"질문: {query}"

        if request and request.get("app_context") and question_type and question_type.upper() in ['ACTION', 'PLAN']:
            user_content += f"\n\n## 시스템 컨텍스트\n현재 설치된 앱: {request['app_context']}"

        if context.strip():
            user_content += f"\n\n관련 문서:\n{context}"

        if chat_history.strip():
            user_content += f"\n\n이전 대화:\n{chat_history}"

        is_post_action_check = (question_type and question_type.upper() in ["POST_ACTION", "POST-ACTION"])

        if re.search(r'[가-힣]', query) and not is_post_action_check:
            if self.config_manager.get_option_config().get('translation') == 'True':
                # 캐시된 번역 결과 재사용 (RAG 검색에서 이미 번역됨)
                translated_question = self.translation_cache.get(query)
                if not translated_question:
                    translated_question, _ = self.translate_query_for_search(query, intent)
                if translated_question and translated_question != query:
                    user_content += f"\n(English: {translated_question})"
                    print(f"🔍 번역 추가됨 (캐시): {translated_question}")
        elif is_post_action_check:
            print(f"🔍 POST_ACTION이므로 번역 건너뜀")

        # ChatML 포맷
        prompt = f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""

        return prompt

    # process_question: BaseModelHandler default 사용 (generate_stream wrap).
    # process_question_error: BaseModelHandler default 사용.

    async def generate_complete_error(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> str:
        try:
            if not self.available:
                return "Qwen3.5 모델을 사용할 수 없습니다."

            if isinstance(request, dict):
                query = request.get("question", "")
                system_prompt = request.get("system", "")
                user_prompt = request.get("user", "")
            else:
                query = str(request)
                system_prompt = ""
                user_prompt = query

            if system_prompt and user_prompt:
                formatted_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
            else:
                formatted_prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

            response = self.local_model.llm.create_completion(
                prompt=formatted_prompt,
                max_tokens=4096,
                temperature=0.1,
                stop=[
                    "<|im_end|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )

            full_response = response["choices"][0]["text"].strip()

            # <think> 블록 제거
            full_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

            # ChatML 태그 정리
            unwanted_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
            for token in unwanted_tokens:
                full_response = full_response.replace(token, "")

            return full_response.strip()

        except Exception as e:
            error_message = f"Qwen 에러 생성 중 오류: {str(e)}"
            logger.error(error_message)
            return error_message

    def _fix_unicode_response(self, response: str) -> str:
        try:
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

    def _apply_response_format(self, system_prompt: str, response_format: bool = False) -> str:
        if response_format is True:
            format_instruction = """

                🎨 ENHANCED FORMAT MODE - MANDATORY RULES:
                - 🔥 MUST use emojis in every response! Add relevant emojis to section headers and key points
                - 📝 MUST use **bold** for important terms and concepts
                - ✨ MUST use markdown formatting: headers (#), bullet points (•), code blocks (```)
                - 🎯 Start your response with a relevant emoji
                - 💡 Use emojis to categorize information (📊 for data, ⚡ for commands, 🔍 for examples)
                - Make the response visually engaging and colorful with emojis and formatting
            """
        else:
            format_instruction = """

                PLAIN TEXT RESTRICTIONS:
                CLEAN TEXT MODE - NO VISUAL EMPHASIS BUT PRESERVE STRUCTURE:
                - NEVER use **bold**, *italic*, or visual emphasis markdown
                - NEVER use emojis or special unicode characters
                - NEVER use # headers or complex formatting
                - MUST use proper indentation for sub-items:
                  Example format:
                  1) Main command
                     - Sub-explanation with 5 spaces indentation
                     - Another sub-explanation with 5 spaces indentation
                  2) Next command
                     - Sub-explanation with 5 spaces indentation
                - ALWAYS indent sub-items with exactly 5 spaces before the dash
                - For emphasis: use CAPITAL LETTERS or descriptive words
                - Keep language policy intact (use 해요체 for Korean)
                - Focus on clear, structured content with proper indentation
            """

        return system_prompt + format_instruction

    def _remove_markdown_formatting(self, text: str) -> str:
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

            text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

            return text

        except Exception as e:
            logger.error(f"마크다운 제거 오류: {e}")
            return text

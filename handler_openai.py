
import os
import json
import time
import re
import asyncio
import yaml
import textwrap
from typing import Dict, AsyncGenerator, List, Tuple, Any, Optional
from openai import OpenAI
from config_utils import ConfigManager
from aibot_logger import ChatLogger
from handler_base import BaseModelHandler, UsageTracker, sanitize_history, api_call_with_retry, api_call_with_retry_async, classify_error
import logging

logger = logging.getLogger(__name__)

class OpenAIHandler(BaseModelHandler):
    registry_key = "gpt"  # handler_registry.HANDLER_CLASSES 키와 일치
    error_response_mode = "stream"  # generate_stream_error 사용 (process_question_error base default)

    def __init__(self, rag_system=None, use_kg: bool = True, intent_analyzer=None):
        super().__init__(rag_system, use_kg, intent_analyzer)
        
        self.config_manager = ConfigManager()
    
        model_config = self.config_manager.get_model_config()
        openai_config = self.config_manager.get_openai_config()
        self.openai_api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
        # OpenAI 호환 백엔드(예: Furiosa 평가 endpoint) 지정 시 base_url override
        self.openai_base_url = openai_config.get('base_url') or None
        self.model = model_config.get('model_name', 'gpt-4o')
        self.temperature = model_config.get('temperature', 0.01)
        self.max_tokens = model_config.get('max_tokens', 1024)
        self.max_completion_tokens = model_config.get('max_completion_tokens', 1024)
        self.reasoning_effort = model_config.get('reasoning_effort', None)
        self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('gpt'))
        self.logger = ChatLogger(self.model)
        # 번역용 모델 — 기본은 경량 gpt-4o-mini. base_url override 시(Furiosa 등) gpt-4o-mini 가 없으므로 self.model 로 폴백.
        self._translate_model = self.model if self.openai_base_url else "gpt-4o-mini"

        if self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
                self.available = True
                _backend = self.openai_base_url or "api.openai.com"
                logger.info(f"OpenAI 모델 초기화 완료: model={self.model} backend={_backend}")
            except Exception as e:
                logger.error(f"OpenAI 클라이언트 초기화 오류: {str(e)}")
                self.client = None
                self.available = False
        else:
            logger.warning("OpenAI API 키가 설정되지 않았습니다.")
            self.client = None
            self.available = False

    def validate_api_key(self, timeout: float = 5.0) -> bool:
        if not self.client:
            return False
        try:
            self.client.with_options(timeout=timeout).models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI API key validation failed: {e}")
            return False

    def _perform_translation(self, query: str, mode: str, preserved: dict = None) -> str:
        try:
            if not self.available:
                return ""

            # 번역 시스템 프롬프트는 카트리지가 정의 — [prompts] translation_* (공유 파일)
            from config_utils import load_translation_prompt
            _mode = 'SUMMARY' if mode == 'SUMMARY' else 'NORMAL'
            prompt = load_translation_prompt(_mode)
            # 번역은 경량 모델 사용 (속도 최적화). 서브클래스(예: Furiosa)는 self._translate_model 을 override 해 자기 모델로 번역.
            # reasoning model (gpt-oss/gpt-5) 사용 시 reasoning_effort='low' 강제 — 번역엔 깊은 사고 불필요, reasoning 토큰이 응답시간 지배하는 걸 방지.
            _trans_params = {
                "model": self._translate_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"{query}"}
                ],
                "temperature": 0.1,
                "max_completion_tokens": 100,
            }
            if self._translate_model.startswith(("openai/gpt-oss", "gpt-oss", "gpt-5")):
                _trans_params["reasoning_effort"] = "low"
            response = api_call_with_retry(lambda: self.client.chat.completions.create(**_trans_params))
            
            translation = response.choices[0].message.content.strip()
            return translation if translation else ""
            
        except Exception as e:
            logger.error(f"OpenAI 번역 오류: {e}")
            return ""
    
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
    
    async def generate_stream(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None,
                        context=None, source_files=None, related_concepts=None, concepts=None, intent_info=None) -> AsyncGenerator[str, None]:
        start_handler_time = time.time()
        if not self.available:
            yield "OpenAI API가 구성되지 않았습니다."
            return

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            history = request.get("history", [])
            response_format = request.get("response_format", False)
            print(f"[DEBUG] OpenAI history length: {len(history) if history else 0}")

        else:
            query = str(request)
            question_type = None
            history = None
            response_format = False

        # ── PII 마스킹: RAG·프롬프트 조립 전에 단방향 치환 (security-review.md S-6) ──
        # 이 핸들러는 외부 API 로 나가므로 마스킹 없이는 원문 PII 가 외부 사업자에게 전달된다.
        query = self._pii_mask_input(request, query)

        chat_session = self.chat_session
        
        if context is None:
            if self.rag_system:
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
                
                if not context:
                    logger.info("관련 문서를 찾을 수 없습니다.")
                else:
                    logger.info(f"참고 문서: {source_files}")
        
        if related_concepts and self.use_kg:
            concept_context = "\n\n관련 개념 정보:\n"
            for i, concept_info in enumerate(related_concepts[:3]):
                concept = concept_info.get('concept')
                node = concept_info.get('node')
                if node:
                    concept_context += f"- {concept}"
                    if hasattr(node, 'name_ko') and node.name_ko:
                        concept_context += f" (한글: {node.name_ko})"
                    concept_context += "\n"
            
            if len(concept_context) > 30:
                context += concept_context

        if question_type and question_type.upper() == "QNA":
            formatted_system_prompt = self._apply_response_format(self.system_prompt, response_format)
        else:
            formatted_system_prompt = self.system_prompt

        # Slack 전용: 별도 Slack 프롬프트 사용 (기존 프롬프트 대체)
        request_context = request.get("context") if isinstance(request, dict) else None
        is_slack = request_context == "Slack"
        if is_slack:
            slack_qna_path = self.config_manager.get_prompt_config().get('slack')
            if slack_qna_path:
                slack_system_prompt = self.load_prompt_from_file(slack_qna_path)
                if slack_system_prompt:
                    formatted_system_prompt = slack_system_prompt

        # locale 지시문 주입
        locale = request.get("locale", "") if isinstance(request, dict) else ""
        if locale:
            formatted_system_prompt += f"\n\n[LANGUAGE RULE]\nYou MUST respond in the language corresponding to locale '{locale}'. This overrides all other language rules."

        messages = [{"role": "system", "content": formatted_system_prompt}]

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if locale:
            user_prompt = f"[IMPORTANT: You MUST respond in locale '{locale}']\n\nQuestion: {query}"
        else:
            user_prompt = f"Question: {query}"

        if isinstance(request, dict) and request.get("app_context") and question_type and question_type.upper() in ['ACTION', 'PLAN']:
            user_prompt += f"\n\n## 시스템 컨텍스트\n현재 설치된 앱: {request['app_context']}"

        if context:
            user_prompt += f"\n\nCurrent time is {current_time}.\n\n관련 문서:{context}"

        history_context = sanitize_history(history, current_query=query)
        if history_context:
            user_prompt += f"\n\n=== 이전 대화 내역 ===\n{history_context}=== 대화 내역 완료 ===\n"

        # locale 지시를 user prompt 끝에도 추가 (RAG 컨텍스트에 묻히지 않도록)
        if locale:
            user_prompt += f"\n\n[REMINDER: Respond in locale '{locale}' only]"

        # 이미지가 포함된 경우 Vision API 형식으로 변환
        images_complete = request.get("images", []) if isinstance(request, dict) else []
        if images_complete:
            user_content = [{"type": "text", "text": user_prompt}]
            for img in images_complete:
                if img.get("base64") and img.get("mimetype"):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mimetype']};base64,{img['base64']}"
                        }
                    })
            messages.append({"role": "user", "content": user_content})
            print(f"📷 [OpenAI] 이미지 {len(images_complete)}개를 Vision API 형식으로 포함")
        else:
            messages.append({"role": "user", "content": user_prompt})
        chat_session.add_message("user", query)

        import time as time_module
        import json
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/gpt/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        selected_intent = question_type.upper()
        history = request.get("history", [])

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE GPT MESSAGES ARRAY\n")
                f.write("=" * 80 + "\n\n")
                
                for i, message in enumerate(messages):
                    f.write(f"📨 Message {i+1} - Role: {message['role'].upper()}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{message['content']}\n")
                    f.write("\n" + "=" * 60 + "\n\n")
                
                f.write(f"Total Messages: {len(messages)}\n")
                f.write(f"System Prompt Length: {len(messages[0]['content']) if messages else 0} chars\n")
                f.write(f"User Prompt Length: {len(user_prompt)} chars\n")
                f.write(f"History Context Length: {len(history_context) if 'history_context' in locals() else 0} chars\n")
                
                f.write(f"\nPrompt File Used: ")
                if selected_intent:
                    intent_upper = selected_intent.upper()
                    if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                        f.write("post_action")
                    elif intent_upper == 'ACTION':
                        f.write("action_gpt")
                    elif intent_upper == 'PLAN':
                        f.write("plan_gpt")
                    else:
                        f.write("qna_gpt")
                else:
                    f.write("default gpt")
                f.write("\n")
                
                f.write(f"RAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")
                    
                f.write(f"History Processed: {'Yes' if history else 'No'}\n")
                if history:
                    f.write(f"History Length: {len(history)} messages\n")
                    
            print(f"💾 전체 GPT 메시지 로그 저장됨: {filename}")
        except Exception as e:
            print(f"❌ 전체 메시지 로그 저장 실패: {e}")

        try:
            start_model_time = time.time()
            prep_elapsed = start_model_time - start_handler_time
            print(f"🚀 GPT 응답 생성 시작 (generate_stream, model={self.model}) | 전처리: {prep_elapsed:.2f}초")

            effective_model = (request.get("model_override") if isinstance(request, dict) else None) or self.model
            create_params = {
                "model": effective_model,
                "messages": messages,
                "temperature": 1 if self.model == "gpt-5-mini" or self.model.startswith("gpt-5.4") else 0.1,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if self.model == "gpt-5-mini":
                create_params["reasoning_effort"] = "minimal"
            elif self.model.startswith("gpt-5.4") and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort
            elif self.model.startswith(("openai/gpt-oss", "gpt-oss")) and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort

            stream = await api_call_with_retry_async(lambda: self.client.chat.completions.create(**create_params))

            full_response = ""
            chunk_count = 0
            stream_usage = None
            # PII 복원(스트리밍) — 토큰이 청크 경계에서 쪼개지지 않게 꼬리 버퍼를 쓴다 (handler_base)
            _pii_emit = self._pii_stream_restorer()

            for chunk in stream:
                chunk_count += 1

                # 마지막 chunk에 usage 정보
                if hasattr(chunk, 'usage') and chunk.usage:
                    stream_usage = chunk.usage

                if (hasattr(chunk, 'choices') and chunk.choices and
                    hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta and
                    chunk.choices[0].delta.content is not None):

                    content = chunk.choices[0].delta.content
                    yield _pii_emit(content)
                    full_response += content

            _tail = _pii_emit("", final=True)
            if _tail:
                yield _tail

            end_model_time = time.time()
            elapsed = end_model_time - start_model_time
            total_elapsed = end_model_time - start_handler_time
            if stream_usage:
                self._log_token_cost(stream_usage, elapsed, "stream", prep_elapsed=prep_elapsed, total_elapsed=total_elapsed)
            else:
                print(f"✅ GPT 응답 생성 완료 {elapsed:.2f}초 (generate_stream, model={self.model})")

            chat_session.add_message("assistant", full_response)
            
            request_type_stream = request.get("type", "").upper()
            question_type_upper = question_type.upper() if question_type else ""
            show_metadata_stream = self.config_manager.get_option_config().get('show_metadata')
            print(f"🔍 스트리밍 메타데이터 조건: request_type='{request_type_stream}', question_type='{question_type_upper}', show_metadata={show_metadata_stream}")

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
                        elif isinstance(src, tuple) and len(src) >= 3:
                            yield f"\n{i+1}. `{src[0]}` (유사도: {src[2]:.3f})"
                else:
                    yield f"\n\n📚 **참조 문서:** 없음"
                
                end_time = time.time()
                yield f"\n\n⏱️ 답변 시간: {end_time - start_time:.2f}초"
            
            self.last_response_time = end_time - start_time if 'end_time' in locals() else 0
            self.last_concepts = concepts if concepts else []
            self.last_related_concepts = related_concepts if related_concepts else []
            
        except Exception as e:
            logger.error(f"OpenAI generate_stream [{type(e).__name__}]: {e}")
            yield classify_error(e)

    async def generate_stream_error(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> AsyncGenerator[str, None]:
        if not self.available:
            yield "OpenAI API가 구성되지 않았습니다."
            return
        
        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
        else:
            query = str(request)
            question_type = None
        
        try:
            messages = [{"role": "system", "content": request.get("system", "")}]
            messages.append({"role": "user", "content": request.get("user", "")})
            
            effective_model = (request.get("model_override") if isinstance(request, dict) else None) or self.model
            create_params = {
                "model": effective_model,
                "messages": messages,
                "temperature": 1 if self.model == "gpt-5-mini" or self.model.startswith("gpt-5.4") else 0,
                "stream": True,
            }
            if self.model == "gpt-5-mini":
                create_params["reasoning_effort"] = "minimal"
            elif self.model.startswith("gpt-5.4") and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort
            elif self.model.startswith(("openai/gpt-oss", "gpt-oss")) and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort

            stream = await api_call_with_retry_async(lambda: self.client.chat.completions.create(**create_params))

            full_response = ""
            _pii_emit = self._pii_stream_restorer()
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield _pii_emit(content)
                    full_response += content
            _tail = _pii_emit("", final=True)
            if _tail:
                yield _tail
            
        except Exception as e:
            logger.error(f"OpenAI generate_stream_raw [{type(e).__name__}]: {e}")
            yield classify_error(e)

    async def generate_complete(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None,
                          context=None, source_files=None, related_concepts=None, concepts=None, intent_info=None) -> Dict:
        print(f"🚀 [DEBUG] generate_complete 호출됨!")
        print(f"🚀 [DEBUG] request type: {request.get('type') if isinstance(request, dict) else 'not dict'}")
        start_time = time.time()
        self._usage_tracker = UsageTracker(model=self.model, user_id=user_id or "")
        
        if not self.available:
            return {
                "answer": "OpenAI API가 구성되지 않았습니다.",
                "intent": "error",
                "sources": [],
                "response_time": 0,
                "question_type": None
            }
        
        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            history = request.get("history", [])
            response_format = request.get("response_format", False)
            print(f"[DEBUG] OpenAI history length: {len(history) if history else 0}")

        else:
            query = str(request)
            response_format = False
            question_type = None
            history = None

        # ── PII 마스킹: RAG·프롬프트 조립 전에 단방향 치환 (security-review.md S-6) ──
        query = self._pii_mask_input(request, query)

        chat_session = self.chat_session

        if context is None:
            if question_type and question_type.upper() == "TITLE":
                context, source_files, related_concepts, concepts = "", [], [], []
                intent_info = None
            elif self.rag_system:
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
                        concept_context = "\n\n관련 개념 정보:\n"
                        for i, concept_info in enumerate(related_concepts[:3]):
                            concept = concept_info.get('concept')
                            node = concept_info.get('node')
                            if node:
                                concept_context += f"- {concept}"
                                if hasattr(node, 'name_ko') and node.name_ko:
                                    concept_context += f" (한글: {node.name_ko})"
                                concept_context += "\n"

                        if len(concept_context) > 30:
                            context += concept_context
                except Exception as e:
                    logger.error(f"RAG 검색 중 오류: {e}")
                    context, source_files, related_concepts, concepts = "", [], [], []
        elif context != "":  
            self.last_sources = source_files if source_files else []

            if related_concepts and self.use_kg:
                concept_context = "\n\n관련 개념 정보:\n"
                for i, concept_info in enumerate(related_concepts[:3]):
                    concept = concept_info.get('concept')
                    node = concept_info.get('node')
                    if node:
                        concept_context += f"- {concept}"
                        if hasattr(node, 'name_ko') and node.name_ko:
                            concept_context += f" (한글: {node.name_ko})"
                        concept_context += "\n"

                if len(concept_context) > 30:
                    context += concept_context

        original_prompt = self.system_prompt

        selected_intent = question_type or (intent_info.get('primary_intent', '') if intent_info else '')

        intent_condition = self.config_manager.get_option_config().get('user_intent_prompt', '').lower() == 'true'
        selected_intent_condition = bool(selected_intent)


        if intent_condition and selected_intent_condition:
            logger.info(f"의도 분석 개별 프롬프트 사용: {selected_intent}")
            intent_upper = selected_intent.upper()

            if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                    self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('post_action'))
                    print(f"🔧 OpenAI 일반 POST-ACTION 프롬프트 로드 완료")
            elif intent_upper == 'ACTION':
                self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('action_gpt'))
            elif intent_upper == 'PLAN':
                print(f"🔒 [OpenAI] PLAN 응답 생성 - 워크플로우 비활성화됨")
                self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('plan_gpt'))
            elif intent_upper == 'TITLE':
                self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('title'))
            else:
                self.system_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('qna_gpt'))

        if question_type and question_type.upper() == "QNA":
            formatted_system_prompt = self._apply_response_format(self.system_prompt, response_format)
        else:
            formatted_system_prompt = self.system_prompt

        # Slack 전용: 별도 Slack 프롬프트 사용 (기존 프롬프트 대체)
        request_context = request.get("context") if isinstance(request, dict) else None
        is_slack = request_context == "Slack"
        if is_slack:
            slack_qna_path = self.config_manager.get_prompt_config().get('slack')
            if slack_qna_path:
                slack_system_prompt = self.load_prompt_from_file(slack_qna_path)
                if slack_system_prompt:
                    formatted_system_prompt = slack_system_prompt

        # locale 지시문 주입
        locale = request.get("locale", "") if isinstance(request, dict) else ""
        if locale:
            formatted_system_prompt += f"\n\n[LANGUAGE RULE]\nYou MUST respond in the language corresponding to locale '{locale}'. This overrides all other language rules."

        # TITLE은 간결한 제목 요약이므로 YAML에서 프롬프트만 직접 로드 (date_info 없이)
        if question_type and question_type.upper() == "TITLE":
            title_prompt_path = self.config_manager.get_prompt_config().get('title')
            try:
                with open(title_prompt_path, 'r', encoding='utf-8') as f:
                    title_system_prompt = yaml.safe_load(f).get('prompt', '').strip()
            except Exception:
                title_system_prompt = formatted_system_prompt
            if locale:
                title_system_prompt += f"\n\n[CRITICAL LANGUAGE RULE]\nYou MUST write the title in the language of locale '{locale}'. This is the highest priority rule."
            user_prompt = f"Question: {query}"
            if locale:
                user_prompt += f"\n\n(Respond in locale '{locale}' only)"
            messages = [{"role": "system", "content": title_system_prompt}]
            messages.append({"role": "user", "content": user_prompt})

            logger.info(f"[TITLE] OpenAI 제목 요약 호출 - model: {self.model}, locale: {locale}")
            effective_model = (request.get("model_override") if isinstance(request, dict) else None) or self.model
            create_params = {
                "model": effective_model,
                "messages": messages,
                "temperature": 1 if self.model == "gpt-5-mini" or self.model.startswith("gpt-5.4") else 0.01,
                "max_completion_tokens": 1024,
            }
            if self.model == "gpt-5-mini":
                create_params["reasoning_effort"] = "minimal"
            elif self.model.startswith("gpt-5.4") and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort
            elif self.model.startswith(("openai/gpt-oss", "gpt-oss")) and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort
            response = await api_call_with_retry_async(lambda: self.client.chat.completions.create(**create_params))
            self._usage_tracker.track_openai_response(response)
            raw_content = response.choices[0].message.content
            answer = raw_content.strip() if raw_content else ""
            logger.info(f"[TITLE] 응답: '{answer}'")
            self.system_prompt = original_prompt
            return {
                "answer": self._pii_unmask(answer),
                "sources": source_files if source_files else [],
                "intent": "TITLE",
                "question_type": "TITLE",
                "usage": self._usage_tracker.to_dict(),
            }

        messages = [{"role": "system", "content": formatted_system_prompt}]

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if locale:
            user_prompt = f"[IMPORTANT: You MUST respond in locale '{locale}']\n\nQuestion: {query}"
        else:
            user_prompt = f"Question: {query}"

        if isinstance(request, dict) and request.get("app_context") and question_type and question_type.upper() in ['ACTION', 'PLAN']:
            user_prompt += f"\n\n## 시스템 컨텍스트\n현재 설치된 앱: {request['app_context']}"

        if context:
            user_prompt += f"\n\nCurrent time is {current_time}.\n\n관련 문서:{context}"

        history_context = sanitize_history(history, current_query=query)
        if history_context:
            user_prompt += f"\n\n=== 이전 대화 내역 ===\n{history_context}=== 대화 내역 완료 ===\n"

        # locale 지시를 user prompt 끝에도 추가 (RAG 컨텍스트에 묻히지 않도록)
        if locale:
            user_prompt += f"\n\n[REMINDER: Respond in locale '{locale}' only]"

        # 이미지가 포함된 경우 Vision API 형식으로 변환
        images = request.get("images", []) if isinstance(request, dict) else []
        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            for img in images:
                if img.get("base64") and img.get("mimetype"):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mimetype']};base64,{img['base64']}"
                        }
                    })
            messages.append({"role": "user", "content": user_content})
            print(f"📷 [OpenAI] 이미지 {len(images)}개를 Vision API 형식으로 포함")
        else:
            messages.append({"role": "user", "content": user_prompt})
        chat_session.add_message("user", query)

        import time as time_module
        import json
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/gpt/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE GPT MESSAGES ARRAY\n")
                f.write("=" * 80 + "\n\n")
                
                for i, message in enumerate(messages):
                    f.write(f"📨 Message {i+1} - Role: {message['role'].upper()}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{message['content']}\n")
                    f.write("\n" + "=" * 60 + "\n\n")
                
                f.write(f"Total Messages: {len(messages)}\n")
                f.write(f"System Prompt Length: {len(messages[0]['content']) if messages else 0} chars\n")
                f.write(f"User Prompt Length: {len(user_prompt)} chars\n")
                f.write(f"History Context Length: {len(history_context) if 'history_context' in locals() else 0} chars\n")
                
                f.write(f"\nPrompt File Used: ")
                if selected_intent:
                    intent_upper = selected_intent.upper()
                    if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                        f.write("post_action")
                    elif intent_upper == 'ACTION':
                        f.write("action_gpt")
                    elif intent_upper == 'PLAN':
                        f.write("plan_gpt")
                    else:
                        f.write("qna_gpt")
                else:
                    f.write("default gpt")
                f.write("\n")
                
                f.write(f"RAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")
                    
                f.write(f"History Processed: {'Yes' if history else 'No'}\n")
                if history:
                    f.write(f"History Length: {len(history)} messages\n")
                    
            print(f"💾 전체 GPT 메시지 로그 저장됨: {filename}")
        except Exception as e:
            print(f"❌ 전체 메시지 로그 저장 실패: {e}")
            
        try:
            start_model_time = time.time()
            print(f"🚀 GPT 응답 생성 시작 (generate_complete, model={self.model})...")
            effective_model = (request.get("model_override") if isinstance(request, dict) else None) or self.model
            create_params = {
                "model": effective_model,
                "messages": messages,
                "temperature": 1 if self.model == "gpt-5-mini" or self.model.startswith("gpt-5.4") else 0.01,
                "stream": False,
            }
            if self.model == "gpt-5-mini":
                create_params["reasoning_effort"] = "minimal"
            elif self.model.startswith("gpt-5.4") and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort
            elif self.model.startswith(("openai/gpt-oss", "gpt-oss")) and self.reasoning_effort:
                create_params["reasoning_effort"] = self.reasoning_effort

            response = await api_call_with_retry_async(lambda: self.client.chat.completions.create(**create_params))
            end_model_time = time.time()
            elapsed = end_model_time - start_model_time
            if hasattr(response, 'usage') and response.usage:
                self._log_token_cost(response.usage, elapsed, "complete")
                self._usage_tracker.track_openai_response(response)
            else:
                print(f"✅ GPT 응답 생성 완료 {elapsed:.2f}초 (generate_complete, model={self.model})")

            full_response = response.choices[0].message.content
            end_time = time.time()
            
            if connection and user_id:
                self.logger.log(user_id, query, full_response, f"{end_time - start_time:.2f}", 
                                self.last_sources, concepts, related_concepts, connection)
            
            chat_session.add_message("assistant", full_response)
            
            self.last_response_time = end_time - start_time
            self.last_concepts = concepts if concepts else []
            self.last_related_concepts = related_concepts if related_concepts else []
            
            if (request.get("type", "").upper() == "ACTION" and
                ('"action": "clarify"' in full_response or '"action":"clarify"' in full_response)):
                original_intent = request.get("type", "ACTION")
                print(f"🔄 Clarify 응답 감지: {original_intent} 의도 유지")

                try:
                    clarify_data = json.loads(full_response)
                    if isinstance(clarify_data, dict) and "message" in clarify_data:
                        full_response = clarify_data["message"]
                        print(f"📝 Clarify 메시지 추출: {full_response}")

                        if "original_intent" in clarify_data:
                            request["original_intent"] = clarify_data["original_intent"]
                            print(f"💾 원본 의도 저장: {clarify_data['original_intent']}")
                except json.JSONDecodeError:
                    match = re.search(r'"message":\s*"([^"]*)"', full_response)
                    if match:
                        full_response = match.group(1)
                        print(f"📝 Clarify 메시지 추출 (정규식): {full_response}")

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

            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False and not is_slack:
                full_response = self._remove_markdown_formatting(full_response)

            final_response = self._fix_unicode_response(full_response)

            show_metadata_config = self.config_manager.get_option_config().get('show_metadata')
            print(f"🔍 메타데이터 조건 체크: request_type='{request_type}', selected_intent='{selected_intent_upper}', show_metadata={show_metadata_config}")

            if (show_metadata_config and not is_post_action):
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
            
            if is_post_action:
                try:
                    import json
                    json_response = json.loads(full_response)

                    if 'updated_queue' in json_response and json_response['updated_queue']:
                        updated_queue = json_response['updated_queue']
                        analysis = json_response.get('analysis', '분석 완료')

                        print(f"🔄 워크플로우 큐: {len(updated_queue)}개 단계")

                        if updated_queue:
                            next_step = updated_queue[0]
                            remaining_queue = updated_queue[1:]  

                            print(f"🎯 다음 단계 실행: {next_step}")
                            print(f"📋 남은 큐: {len(remaining_queue)}개")

                            next_action = {
                                "msg": next_step,
                                "category": "action",
                                "remaining_queue": remaining_queue
                            }

                            full_response = json.dumps([next_action], ensure_ascii=False, indent=2)
                            print(f"🔍 [다음 ACTION] :\n{full_response}")


                    elif 'msg' in json_response:
                        full_response = json_response['msg']
                        print(f"📝 POST_ACTION msg 추출: {full_response}")
                    else:
                        print(f"📝 POST_ACTION JSON에 msg 필드 없음")

                except json.JSONDecodeError:
                    print(f"📝 POST_ACTION 응답이 JSON이 아님, 원본 사용")

            self.system_prompt = original_prompt

            detected_intent = selected_intent or (intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown')

            return {
                # PII 복원: 마스킹했다면 응답의 <ENTITY_n> 토큰을 원본값으로 (handler_base)
                "answer": self._pii_unmask(final_response),
                "intent": detected_intent.upper() if detected_intent else "GENERAL",
                "intent_info": intent_info,
                "sources": source_files,
                "response_time": end_time - start_time,
                "question_type": question_type,
                "usage": self._usage_tracker.to_dict(),
            }
            
        except Exception as e:
            self.system_prompt = original_prompt

            logger.error(f"OpenAI generate_complete [{type(e).__name__}]: {e}")
            return {
                "answer": classify_error(e),
                "intent": "error",
                "sources": [],
                "response_time": time.time() - start_time,
                "question_type": question_type
            }

    # process_question: BaseModelHandler default 사용 (generate_stream wrap).
    # process_question_error: BaseModelHandler default 사용 (generate_stream_error
    # 보유 시 streaming wrap). 기존 동작과 동일.

    async def get_complete_answer(self, request, user_id=None, channel_id=None, source='API', thread_ts=None):
        try:
            result = await self.generate_complete(request, user_id, channel_id, source, thread_ts)
            
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "intent": result.get("intent", "unknown")
            }
            
        except Exception as e:
            logger.error(f"OpenAI get_complete_answer [{type(e).__name__}]: {e}")

            return {
                "answer": classify_error(e),
                "sources": [],
                "intent": "error"
            }

    def get_search_status(self):
        try:
            return {
                "rag_system_available": True if hasattr(self, 'rag_system') and self.rag_system else False,
                "knowledge_graph_enabled": self.use_kg,  
                "search_config": {
                    "search_strategy": getattr(self, 'search_strategy', 'basic'),
                    "cache_enabled": True
                },
                "total_documents": 0,
                "cache_size": 0
            }
        except Exception as e:
            print(f"OpenAI 검색 상태 확인 오류: {str(e)}")
            return {
                "rag_system_available": False,
                "knowledge_graph_enabled": self.use_kg,  
                "search_config": {"search_strategy": "unknown", "cache_enabled": False},
                "total_documents": 0,
                "cache_size": 0
            }
    
    # 모델별 토큰 단가 (USD per 1M tokens)
    TOKEN_PRICING = {
        "gpt-5.4-mini": {"input": 0.30,  "output": 1.25},
        "gpt-5.4-nano": {"input": 0.10,  "output": 0.40},
        "gpt-5.4":      {"input": 2.50,  "output": 15.00},
        "gpt-5-mini":   {"input": 0.25,  "output": 2.00},
    }

    def _log_token_cost(self, usage, elapsed: float, mode: str, prep_elapsed: float = 0, total_elapsed: float = 0):
        """토큰 사용량 및 예상 비용 로그 출력"""
        input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(usage, 'completion_tokens', 0) or 0
        total_tokens = input_tokens + output_tokens

        # 시간 정보
        time_info = f"GPT {elapsed:.2f}초"
        if prep_elapsed > 0:
            time_info = f"전처리 {prep_elapsed:.2f}초 + GPT {elapsed:.2f}초"
        if total_elapsed > 0:
            time_info += f" = 총 {total_elapsed:.2f}초"

        # 모델명 매칭
        pricing = None
        for key, price in self.TOKEN_PRICING.items():
            if self.model.startswith(key):
                pricing = price
                break

        if pricing:
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
            print(f"✅ GPT 응답 완료 [{time_info}] ({mode}, model={self.model}) "
                  f"| 토큰: in={input_tokens:,} out={output_tokens:,} total={total_tokens:,} "
                  f"| 비용: ${total_cost:.4f} (in=${input_cost:.4f} + out=${output_cost:.4f})")
        else:
            print(f"✅ GPT 응답 완료 [{time_info}] ({mode}, model={self.model}) "
                  f"| 토큰: in={input_tokens:,} out={output_tokens:,} total={total_tokens:,}")

    def _apply_response_format(self, system_prompt: str, response_format: bool = False) -> str:
        print(f"🔥 [DEBUG] _apply_response_format called with response_format: {response_format}")

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
                - Keep language policy intact (follow the Language Rule in the system prompt)
                - Focus on clear, structured content with proper indentation
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
                if re.match(r'^(\s*)[-*+]\s+', line):
                    match = re.match(r'^(\s*)[-*+]\s+(.*)', line)
                    if match:
                        indent = match.group(1)
                        content = match.group(2)
                        processed_lines.append(f'{indent}- {content}')
                    else:
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)
            text = '\n'.join(processed_lines)

            lines = text.split('\n')
            clean_lines = []
            prev_empty = False
            for line in lines:
                if line.strip() == '':  
                    if not prev_empty:  
                        clean_lines.append('')
                    prev_empty = True
                else: 
                    clean_lines.append(line)
                    prev_empty = False
            text = '\n'.join(clean_lines).strip()

            return text

        except Exception as e:
            logger.warning(f"마크다운 제거 중 오류: {e}")
            return text

    # ── Agent (completions2) — OpenAI/호환 백엔드: messages passthrough ──
    # length enum → 토큰 캡 매핑 (추론 모델은 reasoning_tokens 차감 → 약 4x 버퍼)
    _AGENT_LENGTH_TOKEN_MAP = {"low": (512, 2048), "medium": (2048, 8192), "high": (8192, 32768)}
    _AGENT_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
    _AGENT_NEW_TOKENS_PREFIXES = ("gpt-5", "gpt-4.1", "o1", "o3", "o4", "chatgpt-")

    async def agent_complete(self, messages: List[Dict], options: Dict = None) -> Dict[str, Any]:
        """OpenAI 호환 백엔드용 override. messages 를 그대로 전달하고 length/reasoning_effort 적용.
        options: {"length", "reasoning_effort", "model_override"}. 잘못된 값은 ValueError(→400)."""
        if not self.available or self.client is None:
            raise RuntimeError(f"{self.registry_key} 모델을 사용할 수 없습니다. (핸들러 미로드)")
        options = options or {}
        # ── PII 치환: 외부 API 로 나가기 직전 (security-review.md S-6) ──
        # 온프레미스 콘솔에서 가장 아픈 경로 — 마스킹하지 않으면 원문 PII 가 외부 사업자로 나간다.
        messages = self._pii_mask_messages(messages)
        effective_model = options.get("model_override") or self.model
        low_model = effective_model.lower()

        # locale 강제 — api/ai/chats 와 동일 의도. system 메시지로 주입(앞에 prepend).
        locale = options.get("locale")
        if locale:
            messages = [{"role": "system",
                         "content": (f"You MUST respond in the language corresponding to locale "
                                     f"'{locale}'. This overrides all other language rules.")}] + list(messages)

        create_params = {
            "model": effective_model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }

        # length: 없으면 medium
        length = (options.get("length") or "medium").lower()
        if length not in self._AGENT_LENGTH_TOKEN_MAP:
            raise ValueError("length는 low/medium/high 중 하나여야 합니다")
        is_reasoning = low_model.startswith(self._AGENT_REASONING_PREFIXES) and "chat-latest" not in low_model
        non_reason_cap, reason_cap = self._AGENT_LENGTH_TOKEN_MAP[length]
        token_key = "max_completion_tokens" if low_model.startswith(self._AGENT_NEW_TOKENS_PREFIXES) else "max_tokens"
        create_params[token_key] = reason_cap if is_reasoning else non_reason_cap

        # reasoning_effort: 없으면 핸들러 config 기본값
        req_reasoning = options.get("reasoning_effort")
        if req_reasoning is None:
            req_reasoning = self.reasoning_effort
        if req_reasoning is not None:
            if req_reasoning not in ("minimal", "low", "medium", "high"):
                raise ValueError("reasoning_effort는 minimal/low/medium/high 중 하나여야 합니다")
            if is_reasoning:
                create_params["reasoning_effort"] = req_reasoning
            else:
                print(f"⚠️  [completions2] reasoning_effort 미지원 모델({effective_model}) — 옵션 무시")

        try:
            response = await asyncio.to_thread(self.client.chat.completions.create, **create_params)
        except Exception as openai_err:
            raise RuntimeError(f"OpenAI 호출 실패 (model={effective_model}): {openai_err}")

        return {
            # 마스킹했다면 응답의 <ENTITY_n> 토큰을 원본값으로 되돌린다 (restore=False 면 그대로 둔다).
            "content": self._pii_unmask(response.choices[0].message.content or ""),
            "model": effective_model,
            "usage": response.usage.model_dump() if response.usage else {},
        }
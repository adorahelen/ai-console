"""Claude on AWS Bedrock 핸들러.

handler_openai 의 구조(intent 분기 / POST_ACTION / Clarify / 메타데이터)를 그대로 따르고
모델 호출 표면만 boto3 bedrock-runtime + Anthropic Messages API 포맷으로 교체.

자격증명: config.ini [bedrock] 의 aws_access_key_id/secret/session_token 셋 다 채워져 있으면
explicit 으로 전달, 비어 있으면 boto3 default credential chain 사용.

응답 캐시 키는 DEFAULT_MODEL 로 통일되므로 라우팅 모델명(model_id) 신경 안 써도 됨.
"""
import os
import json
import time
import re
import asyncio
import yaml
from typing import Any, Dict, AsyncGenerator
from config_utils import ConfigManager
from aibot_logger import ChatLogger
from handler_base import BaseModelHandler, UsageTracker, sanitize_history, classify_error
import logging

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    BotoCoreError = ClientError = Exception

# Anthropic Messages on Bedrock 의 API 버전 (모델 ID 와 별개, 고정값).
ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"


def _content_to_text(content) -> str:
    """OpenAI 호환 content(str 또는 parts 배열)를 평문으로. 배열이면 text 파트만 취한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


class ClaudeHandler(BaseModelHandler):
    registry_key = "claude"
    is_local = False
    error_response_mode = "stream"  # generate_stream_error 사용 (OpenAI 와 동일)

    def __init__(self, rag_system=None, use_kg: bool = True, intent_analyzer=None):
        super().__init__(rag_system, use_kg, intent_analyzer)

        self.config_manager = ConfigManager()
        bedrock_cfg = self.config_manager.get_bedrock_config()

        self.aws_region = bedrock_cfg.get('aws_region') or 'us-east-1'
        self.model = bedrock_cfg.get('claude_model_id') or 'anthropic.claude-opus-4-7-20260101-v1:0'
        # 번역 전용 경량 모델 — 비면 본 모델로 폴백
        self._translate_model = bedrock_cfg.get('claude_translate_model_id') or self.model
        self.max_tokens = bedrock_cfg.get('claude_max_tokens', 4096)
        self.temperature = bedrock_cfg.get('claude_temperature', 0.01)

        self.system_prompt = self.load_prompt_from_file(
            self.config_manager.get_prompt_config().get('claude')
        )
        self.logger = ChatLogger(self.model)

        self.bedrock_runtime = None
        self.available = False
        self._init_bedrock_client(bedrock_cfg)

    def _init_bedrock_client(self, bedrock_cfg: Dict):
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 가 설치되지 않아 Claude(Bedrock) 핸들러를 초기화할 수 없습니다.")
            return
        try:
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": self.aws_region,
            }
            ak = bedrock_cfg.get('aws_access_key_id')
            sk = bedrock_cfg.get('aws_secret_access_key')
            if ak and sk:
                client_kwargs["aws_access_key_id"] = ak
                client_kwargs["aws_secret_access_key"] = sk
                st = bedrock_cfg.get('aws_session_token')
                if st:
                    client_kwargs["aws_session_token"] = st
                cred_src = "config.ini[bedrock] explicit"
            else:
                cred_src = "boto3 default chain (env / ~/.aws / IAM role)"
            self.bedrock_runtime = boto3.client(**client_kwargs)
            self.available = True
            logger.info(f"Claude(Bedrock) 초기화 완료: model={self.model} region={self.aws_region} creds={cred_src}")
        except (BotoCoreError, ClientError, Exception) as e:
            logger.error(f"Claude(Bedrock) 초기화 실패: {type(e).__name__}: {e}")
            self.bedrock_runtime = None
            self.available = False

    # ── 메시지 빌더 ───────────────────────────────────────────────
    def _build_messages(self, user_prompt: str, images: list = None):
        """Anthropic Messages API 형식 content blocks 구성.

        이미지 있으면 [{"type":"image","source":{"type":"base64","media_type":..,"data":..}},
                       {"type":"text","text":..}], 없으면 plain string.
        """
        if not images:
            return [{"role": "user", "content": user_prompt}]
        content = []
        for img in images:
            if img.get("base64") and img.get("mimetype"):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["mimetype"],
                        "data": img["base64"],
                    },
                })
        content.append({"type": "text", "text": user_prompt})
        return [{"role": "user", "content": content}]

    async def agent_complete(self, messages: list, options: Dict = None) -> Dict[str, Any]:
        """completions2/위저드용 균일 진입점 — OpenAI 스타일 messages → Anthropic 포맷 호출.
        반환: {"content", "model", "usage"}. (로컬 핸들러의 build_agent_prompt 경로 대신
        API 핸들러는 messages 를 그대로 전달한다 — OpenAIHandler.agent_complete 와 동일 패턴.)"""
        if not self.available or self.bedrock_runtime is None:
            raise RuntimeError("claude 모델을 사용할 수 없습니다. (핸들러 미로드)")
        options = options or {}
        # ── PII 치환: 외부 API 로 나가기 직전 (security-review.md S-6) ──
        messages = self._pii_mask_messages(messages)
        system = "\n\n".join(_content_to_text(m.get("content")) for m in messages
                             if m.get("role") == "system")
        locale = options.get("locale")
        if locale:
            system = (f"You MUST respond in the language corresponding to locale '{locale}'. "
                      f"This overrides all other language rules.\n\n") + system
        conv = [{"role": ("assistant" if m.get("role") == "assistant" else "user"),
                 "content": _content_to_text(m.get("content"))}
                for m in messages if m.get("role") in ("user", "assistant")]
        if not conv:
            raise ValueError("user/assistant 역할 메시지가 최소 1개 필요합니다.")
        result = await asyncio.to_thread(self._invoke_complete, system, conv)
        text = "".join(b.get("text", "") for b in result.get("content", [])
                       if isinstance(b, dict) and b.get("type") == "text")
        u = result.get("usage", {}) or {}
        pt, ct = u.get("input_tokens", 0), u.get("output_tokens", 0)
        text = self._pii_unmask(text)
        return {"content": text, "model": self.model,
                "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                          "total_tokens": pt + ct}}

    def _invoke_complete(self, system: str, messages: list, model_id: str = None,
                         max_tokens: int = None, temperature: float = None) -> Dict:
        """invoke_model 비스트리밍 호출. 응답 JSON 그대로 반환."""
        body = {
            "anthropic_version": ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system
        resp = self.bedrock_runtime.invoke_model(
            modelId=model_id or self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(resp["body"].read())

    def _invoke_stream(self, system: str, messages: list, model_id: str = None,
                       max_tokens: int = None, temperature: float = None):
        """invoke_model_with_response_stream — Anthropic event stream 을 그대로 yield.

        각 event: {"chunk": {"bytes": b"<json>"}}. JSON 디코딩해서 yield.
        """
        body = {
            "anthropic_version": ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system
        resp = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=model_id or self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        for event in resp["body"]:
            chunk = event.get("chunk")
            if not chunk:
                continue
            yield json.loads(chunk["bytes"])

    # ── 필수 1: 번역 ─────────────────────────────────────────────
    def _perform_translation(self, query: str, mode: str, preserved: dict = None) -> str:
        try:
            if not self.available:
                return ""

            # 번역 시스템 프롬프트는 카트리지가 정의 — [prompts] translation_summary/normal/cve.
            # 다른 핸들러와 동일 통로. 예전엔 여기 인라인이라 도메인 탈색이 이 경로를 비껴갔다.
            from config_utils import load_translation_prompt
            system_prompt = load_translation_prompt(mode)

            result = self._invoke_complete(
                system=system_prompt,
                messages=[{"role": "user", "content": query}],
                model_id=self._translate_model,
                max_tokens=200,
                temperature=0.1,
            )
            translation = self._extract_text(result).strip()
            return translation if translation else ""
        except Exception as e:
            logger.error(f"Claude(Bedrock) 번역 오류: {e}")
            return ""

    @staticmethod
    def _extract_text(response_body: Dict) -> str:
        """Anthropic Messages 응답 → text 추출. content blocks 의 type='text' 만 concat."""
        parts = []
        for block in response_body.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    def _track_usage(self, response_body: Dict):
        """Anthropic usage → UsageTracker 누적. {input_tokens, output_tokens}."""
        usage = response_body.get("usage") or {}
        self._usage_tracker.prompt_tokens += usage.get("input_tokens", 0) or 0
        self._usage_tracker.completion_tokens += usage.get("output_tokens", 0) or 0
        self._usage_tracker.calls += 1

    # ── 필수 2: generate_complete ────────────────────────────────
    async def generate_complete(self, request: Dict, user_id=None, channel_id=None, connection=None,
                                thread_ts=None, context=None, source_files=None,
                                related_concepts=None, concepts=None, intent_info=None) -> Dict:
        start_time = time.time()
        self._usage_tracker = UsageTracker(model=self.model, user_id=user_id or "")

        if not self.available:
            return {
                "answer": "Claude(Bedrock) API가 구성되지 않았습니다.",
                "intent": "error",
                "sources": [],
                "response_time": 0,
                "question_type": None,
            }

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            history = request.get("history", [])
            response_format = request.get("response_format", False)
        else:
            query = str(request)
            response_format = False
            question_type = None
            history = []

        # ── PII 마스킹: RAG·프롬프트 조립 전에 단방향 치환 (security-review.md S-6) ──
        # 이 핸들러는 외부 API(Bedrock)로 나가므로 마스킹 없이는 원문 PII 가 외부로 전달된다.
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
                        context, source_files, related_concepts, concepts, intent_info, _ = result
                    elif len(result) >= 5:
                        context, source_files, related_concepts, concepts, intent_info = result
                    else:
                        context, source_files, related_concepts, concepts = result
                        intent_info = None
                    self.last_sources = source_files

                    if related_concepts and self.use_kg:
                        concept_context = "\n\n관련 개념 정보:\n"
                        for concept_info in related_concepts[:3]:
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
                    logger.error(f"RAG 검색 오류: {e}")
                    context, source_files, related_concepts, concepts = "", [], [], []
        elif context != "":
            self.last_sources = source_files if source_files else []

        original_prompt = self.system_prompt
        selected_intent = question_type or (intent_info.get('primary_intent', '') if intent_info else '')

        intent_condition = self.config_manager.get_option_config().get('user_intent_prompt', '').lower() == 'true'
        if intent_condition and selected_intent:
            self.system_prompt = self._select_intent_prompt(selected_intent) or self.system_prompt

        if question_type and question_type.upper() == "QNA":
            formatted_system_prompt = self._apply_response_format(self.system_prompt, response_format)
        else:
            formatted_system_prompt = self.system_prompt

        # Slack 전용 프롬프트 (handler_openai 와 동일 동작)
        request_context = request.get("context") if isinstance(request, dict) else None
        is_slack = request_context == "Slack"
        if is_slack:
            slack_path = self.config_manager.get_prompt_config().get('slack')
            if slack_path:
                slack_prompt = self.load_prompt_from_file(slack_path)
                if slack_prompt:
                    formatted_system_prompt = slack_prompt

        locale = request.get("locale", "") if isinstance(request, dict) else ""
        if locale:
            formatted_system_prompt += (
                f"\n\n[LANGUAGE RULE]\nYou MUST respond in the language corresponding to locale "
                f"'{locale}'. This overrides all other language rules."
            )

        # TITLE 전용 경로 — 간결 요약, RAG 컨텍스트 없이 Claude 호출
        if question_type and question_type.upper() == "TITLE":
            title_prompt_path = self.config_manager.get_prompt_config().get('title_claude') or \
                                self.config_manager.get_prompt_config().get('title')
            try:
                with open(title_prompt_path, 'r', encoding='utf-8') as f:
                    title_system = yaml.safe_load(f).get('prompt', '').strip()
            except Exception:
                title_system = formatted_system_prompt
            if locale:
                title_system += (
                    f"\n\n[CRITICAL LANGUAGE RULE]\nYou MUST write the title in the language of locale "
                    f"'{locale}'. This is the highest priority rule."
                )
            user_prompt = f"Question: {query}"
            if locale:
                user_prompt += f"\n\n(Respond in locale '{locale}' only)"
            try:
                resp = self._invoke_complete(
                    system=title_system,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=1024,
                )
                self._track_usage(resp)
                answer = self._extract_text(resp).strip()
            except Exception as e:
                logger.error(f"Claude(Bedrock) TITLE [{type(e).__name__}]: {e}")
                answer = classify_error(e)
            finally:
                self.system_prompt = original_prompt
            return {
                "answer": self._pii_unmask(answer),
                "sources": source_files if source_files else [],
                "intent": "TITLE",
                "question_type": "TITLE",
                "usage": self._usage_tracker.to_dict(),
            }

        # 일반 경로 — user_prompt 조립
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if locale:
            user_prompt = f"[IMPORTANT: You MUST respond in locale '{locale}']\n\nQuestion: {query}"
        else:
            user_prompt = f"Question: {query}"

        if isinstance(request, dict) and request.get("app_context") and question_type and \
           question_type.upper() in ['ACTION', 'PLAN']:
            user_prompt += f"\n\n## 시스템 컨텍스트\n현재 설치된 앱: {request['app_context']}"

        if context:
            user_prompt += f"\n\nCurrent time is {current_time}.\n\n관련 문서:{context}"

        history_context = sanitize_history(history, current_query=query)
        if history_context:
            user_prompt += f"\n\n=== 이전 대화 내역 ===\n{history_context}=== 대화 내역 완료 ===\n"

        if locale:
            user_prompt += f"\n\n[REMINDER: Respond in locale '{locale}' only]"

        images = request.get("images", []) if isinstance(request, dict) else []
        messages = self._build_messages(user_prompt, images)
        chat_session.add_message("user", query)

        # 디버그 로그 (handler_openai 패턴)
        self._dump_prompt_log(query, selected_intent, question_type, formatted_system_prompt,
                              user_prompt, history_context, source_files, history)

        try:
            start_model_time = time.time()
            print(f"🚀 Claude 응답 생성 시작 (generate_complete, model={self.model})")
            resp = await asyncio.to_thread(
                self._invoke_complete, formatted_system_prompt, messages,
            )
            elapsed = time.time() - start_model_time
            self._track_usage(resp)
            usage = resp.get("usage", {})
            print(f"✅ Claude 응답 완료 {elapsed:.2f}초 | in={usage.get('input_tokens', 0)} "
                  f"out={usage.get('output_tokens', 0)}")

            full_response = self._extract_text(resp)
            end_time = time.time()

            if connection and user_id:
                self.logger.log(user_id, query, full_response, f"{end_time - start_time:.2f}",
                                self.last_sources, concepts, related_concepts, connection)

            chat_session.add_message("assistant", full_response)
            self.last_response_time = end_time - start_time
            self.last_concepts = concepts if concepts else []
            self.last_related_concepts = related_concepts if related_concepts else []

            # Clarify 처리 (handler_openai 와 동일)
            if (request.get("type", "").upper() == "ACTION" and
                ('"action": "clarify"' in full_response or '"action":"clarify"' in full_response)):
                try:
                    clarify_data = json.loads(full_response)
                    if isinstance(clarify_data, dict) and "message" in clarify_data:
                        full_response = clarify_data["message"]
                        if "original_intent" in clarify_data:
                            request["original_intent"] = clarify_data["original_intent"]
                except json.JSONDecodeError:
                    match = re.search(r'"message":\s*"([^"]*)"', full_response)
                    if match:
                        full_response = match.group(1)

            request_type = request.get("type", "").upper()
            selected_intent_upper = selected_intent.upper() if selected_intent else ""
            is_post_action = (request_type in ["POST_ACTION", "POST-ACTION"] or
                              selected_intent_upper in ["POST_ACTION", "POST-ACTION"])

            if is_post_action:
                try:
                    json_resp = json.loads(full_response)
                    if 'msg' in json_resp:
                        full_response = json_resp['msg']
                except json.JSONDecodeError:
                    pass

            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and \
               response_format is False and not is_slack:
                full_response = self._remove_markdown_formatting(full_response)

            final_response = self._fix_unicode_response(full_response)

            show_metadata = self.config_manager.get_option_config().get('show_metadata')
            if show_metadata and not is_post_action:
                final_response = self._append_metadata(final_response, selected_intent_upper,
                                                       intent_info, source_files, end_time - start_time)

            if is_post_action:
                try:
                    json_resp = json.loads(full_response)
                    if 'updated_queue' in json_resp and json_resp['updated_queue']:
                        updated_queue = json_resp['updated_queue']
                        next_step = updated_queue[0]
                        remaining_queue = updated_queue[1:]
                        next_action = {
                            "msg": next_step,
                            "category": "action",
                            "remaining_queue": remaining_queue,
                        }
                        full_response = json.dumps([next_action], ensure_ascii=False, indent=2)
                    elif 'msg' in json_resp:
                        full_response = json_resp['msg']
                except json.JSONDecodeError:
                    pass

            self.system_prompt = original_prompt
            detected_intent = selected_intent or (intent_info.get('primary_intent', 'unknown')
                                                  if intent_info else 'unknown')
            return {
                # PII 복원 (handler_base)
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
            logger.error(f"Claude(Bedrock) generate_complete [{type(e).__name__}]: {e}")
            return {
                "answer": classify_error(e),
                "intent": "error",
                "sources": [],
                "response_time": time.time() - start_time,
                "question_type": question_type,
            }

    # ── 필수 3: generate_stream ──────────────────────────────────
    async def generate_stream(self, request: Dict, user_id=None, channel_id=None, connection=None,
                              thread_ts=None, context=None, source_files=None,
                              related_concepts=None, concepts=None, intent_info=None) -> AsyncGenerator[str, None]:
        start_handler_time = time.time()
        if not self.available:
            yield "Claude(Bedrock) API가 구성되지 않았습니다."
            return

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            history = request.get("history", [])
            response_format = request.get("response_format", False)
        else:
            query = str(request)
            question_type = None
            history = []
            response_format = False

        # ── PII 마스킹: RAG·프롬프트 조립 전에 단방향 치환 (security-review.md S-6) ──
        # 이 핸들러는 외부 API(Bedrock)로 나가므로 마스킹 없이는 원문 PII 가 외부로 전달된다.
        query = self._pii_mask_input(request, query)

        chat_session = self.chat_session

        if context is None and self.rag_system:
            result = await self.get_related_documents(request)
            if len(result) >= 6:
                context, source_files, related_concepts, concepts, intent_info, _ = result
            elif len(result) >= 5:
                context, source_files, related_concepts, concepts, intent_info = result
            else:
                context, source_files, related_concepts, concepts = result
                intent_info = None

        if related_concepts and self.use_kg:
            concept_context = "\n\n관련 개념 정보:\n"
            for concept_info in related_concepts[:3]:
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
        if intent_condition and selected_intent:
            self.system_prompt = self._select_intent_prompt(selected_intent) or self.system_prompt

        if question_type and question_type.upper() == "QNA":
            formatted_system_prompt = self._apply_response_format(self.system_prompt, response_format)
        else:
            formatted_system_prompt = self.system_prompt

        request_context = request.get("context") if isinstance(request, dict) else None
        is_slack = request_context == "Slack"
        if is_slack:
            slack_path = self.config_manager.get_prompt_config().get('slack')
            if slack_path:
                slack_prompt = self.load_prompt_from_file(slack_path)
                if slack_prompt:
                    formatted_system_prompt = slack_prompt

        locale = request.get("locale", "") if isinstance(request, dict) else ""
        if locale:
            formatted_system_prompt += (
                f"\n\n[LANGUAGE RULE]\nYou MUST respond in the language corresponding to locale "
                f"'{locale}'. This overrides all other language rules."
            )

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if locale:
            user_prompt = f"[IMPORTANT: You MUST respond in locale '{locale}']\n\nQuestion: {query}"
        else:
            user_prompt = f"Question: {query}"

        if isinstance(request, dict) and request.get("app_context") and question_type and \
           question_type.upper() in ['ACTION', 'PLAN']:
            user_prompt += f"\n\n## 시스템 컨텍스트\n현재 설치된 앱: {request['app_context']}"

        if context:
            user_prompt += f"\n\nCurrent time is {current_time}.\n\n관련 문서:{context}"

        history_context = sanitize_history(history, current_query=query)
        if history_context:
            user_prompt += f"\n\n=== 이전 대화 내역 ===\n{history_context}=== 대화 내역 완료 ===\n"

        if locale:
            user_prompt += f"\n\n[REMINDER: Respond in locale '{locale}' only]"

        images = request.get("images", []) if isinstance(request, dict) else []
        messages = self._build_messages(user_prompt, images)
        chat_session.add_message("user", query)
        self._dump_prompt_log(query, selected_intent, question_type, formatted_system_prompt,
                              user_prompt, history_context, source_files, history)

        try:
            start_model_time = time.time()
            print(f"🚀 Claude 응답 생성 시작 (generate_stream, model={self.model}) | "
                  f"전처리: {start_model_time - start_handler_time:.2f}초")

            # boto3 stream 은 sync iterator — 별도 스레드에서 enqueue
            full_response = ""
            input_tokens = output_tokens = 0
            # PII 복원(스트리밍) — 토큰이 청크 경계에서 쪼개지지 않게 꼬리 버퍼 사용 (handler_base)
            _pii_emit = self._pii_stream_restorer()
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()

            def _producer():
                try:
                    for ev in self._invoke_stream(formatted_system_prompt, messages):
                        loop.call_soon_threadsafe(queue.put_nowait, ev)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            producer = asyncio.create_task(asyncio.to_thread(_producer))

            while True:
                ev = await queue.get()
                if ev is _SENTINEL:
                    break
                if isinstance(ev, Exception):
                    raise ev
                etype = ev.get("type")
                if etype == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield _pii_emit(text)
                            full_response += text
                elif etype == "message_start":
                    msg = ev.get("message") or {}
                    usage = msg.get("usage") or {}
                    input_tokens += usage.get("input_tokens", 0) or 0
                elif etype == "message_delta":
                    usage = ev.get("usage") or {}
                    output_tokens += usage.get("output_tokens", 0) or 0

            await producer  # propagate exceptions from to_thread

            _tail = _pii_emit("", final=True)
            if _tail:
                yield _tail

            elapsed = time.time() - start_model_time
            print(f"✅ Claude 스트리밍 완료 {elapsed:.2f}초 | in={input_tokens} out={output_tokens}")

            # UsageTracker 누적 (메인 컨텍스트의 추적기와 분리 — stream 경로는 process_question 내에서
            # 별도 tracker 가 없으므로 chat_session 만 갱신)
            chat_session.add_message("assistant", full_response)

            show_metadata = self.config_manager.get_option_config().get('show_metadata')
            request_type_stream = request.get("type", "").upper() if isinstance(request, dict) else ""
            question_type_upper = question_type.upper() if question_type else ""
            is_post_action_stream = (request_type_stream in ["POST_ACTION", "POST-ACTION"] or
                                    question_type_upper in ["POST_ACTION", "POST-ACTION"])

            if show_metadata and not is_post_action_stream:
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
                            yield f"\n{i+1}. `{src.get('file', '')}` (유사도: {src.get('score', 0.0):.3f})"
                        elif isinstance(src, tuple) and len(src) >= 3:
                            yield f"\n{i+1}. `{src[0]}` (유사도: {src[2]:.3f})"
                else:
                    yield f"\n\n📚 **참조 문서:** 없음"
                yield f"\n\n⏱️ 답변 시간: {elapsed:.2f}초"

            self.last_response_time = elapsed
            self.last_concepts = concepts if concepts else []
            self.last_related_concepts = related_concepts if related_concepts else []
            self.system_prompt = original_prompt

        except Exception as e:
            self.system_prompt = original_prompt
            logger.error(f"Claude(Bedrock) generate_stream [{type(e).__name__}]: {e}")
            yield classify_error(e)

    # ── 필수 4: generate_stream_error ────────────────────────────
    async def generate_stream_error(self, request: Dict, user_id=None, channel_id=None,
                                    connection=None, thread_ts=None) -> AsyncGenerator[str, None]:
        if not self.available:
            yield "Claude(Bedrock) API가 구성되지 않았습니다."
            return
        try:
            system = request.get("system", "") if isinstance(request, dict) else ""
            user_text = request.get("user", "") if isinstance(request, dict) else str(request)
            messages = [{"role": "user", "content": user_text}]
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()
            _pii_emit = self._pii_stream_restorer()

            def _producer():
                try:
                    for ev in self._invoke_stream(system, messages):
                        loop.call_soon_threadsafe(queue.put_nowait, ev)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            asyncio.create_task(asyncio.to_thread(_producer))
            while True:
                ev = await queue.get()
                if ev is _SENTINEL:
                    break
                if isinstance(ev, Exception):
                    raise ev
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield _pii_emit(text)
            _tail = _pii_emit("", final=True)
            if _tail:
                yield _tail
        except Exception as e:
            logger.error(f"Claude(Bedrock) generate_stream_error [{type(e).__name__}]: {e}")
            yield classify_error(e)

    # ── 핸들러별 보조: intent → 프롬프트 선택 ───────────────────
    def _select_intent_prompt(self, selected_intent: str):
        """intent 별로 _claude variant 로드. 없으면 gpt variant 로 폴백 (config_utils 가 자동)."""
        pcfg = self.config_manager.get_prompt_config()
        intent_upper = selected_intent.upper()
        path = None
        if intent_upper in ('POST_ACTION', 'POST-ACTION'):
            path = pcfg.get('post_action')
        elif intent_upper == 'ACTION':
            path = pcfg.get('action_claude')
        elif intent_upper == 'PLAN':
            path = pcfg.get('plan_claude')
        elif intent_upper == 'TITLE':
            path = pcfg.get('title_claude')
        else:
            path = pcfg.get('qna_claude')
        if not path:
            return None
        return self.load_prompt_from_file(path)

    def _apply_response_format(self, system_prompt: str, response_format: bool = False) -> str:
        # handler_openai 와 동일한 동작 — 동일 시스템 안에서 일관된 응답 형식 유지.
        if response_format:
            return system_prompt + (
                "\n\n🎨 ENHANCED FORMAT MODE - MANDATORY RULES:\n"
                "- 🔥 MUST use emojis in every response! Add relevant emojis to section headers and key points\n"
                "- 📝 MUST use **bold** for important terms and concepts\n"
                "- ✨ MUST use markdown formatting: headers (#), bullet points (•), code blocks (```)\n"
                "- 🎯 Start your response with a relevant emoji\n"
                "- 💡 Use emojis to categorize information\n"
                "- Make the response visually engaging and colorful with emojis and formatting\n"
            )
        return system_prompt + (
            "\n\nPLAIN TEXT RESTRICTIONS:\n"
            "CLEAN TEXT MODE - NO VISUAL EMPHASIS BUT PRESERVE STRUCTURE:\n"
            "- NEVER use **bold**, *italic*, or visual emphasis markdown\n"
            "- NEVER use emojis or special unicode characters\n"
            "- NEVER use # headers or complex formatting\n"
            "- MUST use proper indentation for sub-items (5 spaces before dash)\n"
            "- For emphasis: use CAPITAL LETTERS or descriptive words\n"
            "- Keep language policy intact (follow the Language Rule in the system prompt)\n"
            "- Focus on clear, structured content with proper indentation\n"
        )

    def _append_metadata(self, response: str, selected_intent_upper: str, intent_info,
                         source_files, response_time: float) -> str:
        """show_metadata=True 일 때 응답 말미에 의도/참조문서/답변시간 부착 (handler_openai 와 동일)."""
        detected_intent = selected_intent_upper or (intent_info.get('primary_intent', 'unknown').upper()
                                                    if intent_info else 'UNKNOWN')
        if detected_intent != 'UNKNOWN':
            intent_text = f"🎯 **의도:** **{detected_intent}**"
            response = re.sub(r"🎯 \*\*의도:\*\* \*\*.*?\*\*", intent_text, response)
            if intent_text not in response:
                response += f"\n\n{intent_text}"
        if source_files:
            refs_text = f"📚 **참조 문서** ({len(source_files)}개):" + \
                        ''.join([f"\n{i+1}. `{src}`" for i, src in enumerate(source_files)])
        else:
            refs_text = "📚 **참조 문서:** 없음"
        response = re.sub(r"📚 \*\*참조 문서\*\*.*?(?=(\n⏱️|\n🎯|\Z))", refs_text, response, flags=re.DOTALL)
        if refs_text not in response:
            response += f"\n\n{refs_text}"
        time_text = f"⏱️ 답변 시간: {response_time:.2f}초"
        response = re.sub(r"⏱️ 답변 시간: .*?초", time_text, response)
        if time_text not in response:
            response += f"\n\n{time_text}"
        return response

    def _fix_unicode_response(self, response: str) -> str:
        try:
            parsed = json.loads(response)

            def fix_msg(obj):
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
                parsed = [fix_msg(it) for it in parsed]
            elif isinstance(parsed, dict):
                parsed = fix_msg(parsed)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return response

    def _remove_markdown_formatting(self, text: str) -> str:
        # handler_openai 와 동일 로직. response_format=False 시 마크다운 제거.
        try:
            if '```query' in text and text.count('```') % 2 != 0:
                return text
            query_blocks = []

            def save_query(m):
                query_blocks.append(m.group(0))
                return f"__QUERY_BLOCK_{len(query_blocks)-1}__"

            text = re.sub(r'```query[\s\S]*?```', save_query, text)
            text = re.sub(r'```(?:[a-zA-Z]*\n)?([\s\S]*?)```', lambda m: m.group(1).strip(), text)
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
            emoji_pattern = re.compile(
                "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
                "\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF"
                "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002328\U000023CF]+",
                flags=re.UNICODE,
            )
            text = emoji_pattern.sub('', text)
            # 빈 줄 정리
            lines = text.split('\n')
            clean = []
            prev_empty = False
            for line in lines:
                if not line.strip():
                    if not prev_empty:
                        clean.append('')
                    prev_empty = True
                else:
                    clean.append(line)
                    prev_empty = False
            return '\n'.join(clean).strip()
        except Exception as e:
            logger.warning(f"마크다운 제거 오류: {e}")
            return text

    def _dump_prompt_log(self, query, selected_intent, question_type, system_prompt, user_prompt,
                         history_context, source_files, history):
        try:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"./logs/full_messages/claude/prompt_contents_{ts}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write("=" * 80 + "\n")
                f.write("CLAUDE BEDROCK MESSAGES\n")
                f.write("=" * 80 + "\n\n")
                f.write("📨 System:\n" + "-" * 60 + "\n" + (system_prompt or "") + "\n\n")
                f.write("📨 User:\n" + "-" * 60 + "\n" + (user_prompt or "") + "\n\n")
                f.write(f"System Prompt Length: {len(system_prompt or '')} chars\n")
                f.write(f"User Prompt Length: {len(user_prompt or '')} chars\n")
                f.write(f"History Context Length: {len(history_context or '')} chars\n")
                f.write(f"RAG Sources: {len(source_files or [])}\n")
                for j, src in enumerate(source_files or []):
                    f.write(f"  {j+1}. {src}\n")
                f.write(f"History Processed: {'Yes' if history else 'No'}\n")
        except Exception as e:
            print(f"❌ Claude 메시지 로그 저장 실패: {e}")

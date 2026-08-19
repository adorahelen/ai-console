import os
import json
import time
import re
import asyncio
import textwrap
import yaml
import uuid
import datetime
from abc import ABC, abstractmethod
from typing import Dict, AsyncGenerator, List, Tuple, Any, Optional
from dataclasses import dataclass
from config_utils import ConfigManager
from aibot_logger import ChatLogger
import logging
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float
    message_id: str

class ChatSession:
    def __init__(self, session_id: str = None, max_messages: int = 10):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
        self.last_activity = time.time()

    def add_message(self, role: str, content: str) -> ChatMessage:
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )

        self.messages.append(message)
        self.last_activity = time.time()

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        return message

    def get_chat_history(self, max_tokens: int = 2000) -> List[Dict[str, str]]:
        history = []
        token_count = 0
        from aibot_embedding import get_encoding
        tokenizer = get_encoding("cl100k_base")

        for message in reversed(self.messages):
            tokens = len(tokenizer.encode(message.content))

            if token_count + tokens > max_tokens:
                break

            history.insert(0, {
                "role": message.role,
                "content": message.content
            })

            token_count += tokens

        return history

    def format_chat_history_for_llama(self, max_tokens: int = 2000) -> str:
        messages = self.get_chat_history(max_tokens)
        formatted_history = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                formatted_history += f"<human>: {content}\n\n"
            elif role == "assistant":
                formatted_history += f"<assistant>: {content}\n\n"

        return formatted_history

    def clear(self) -> None:
        self.messages = []


class UsageTracker:
    """요청 단위로 모든 LLM 호출의 토큰 사용량을 누적 추적한다."""

    def __init__(self, model: str = "", user_id: str = ""):
        self.model = model
        self.user_id = user_id
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.calls = 0
        self._has_openai_details = False

    def track(self, response: dict):
        """create_completion 응답에서 usage를 추출하여 누적한다 (gpt-oss/llama/qwen 등 dict 응답)."""
        usage = response.get("usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.calls += 1

    def track_openai_response(self, response):
        """OpenAI Pydantic 응답에서 reasoning/cached 토큰까지 누적."""
        u = getattr(response, 'usage', None)
        if not u:
            return
        self.prompt_tokens += getattr(u, 'prompt_tokens', 0) or 0
        self.completion_tokens += getattr(u, 'completion_tokens', 0) or 0
        ct = getattr(u, 'completion_tokens_details', None)
        if ct:
            self.reasoning_tokens += getattr(ct, 'reasoning_tokens', 0) or 0
        pt = getattr(u, 'prompt_tokens_details', None)
        if pt:
            self.cached_tokens += getattr(pt, 'cached_tokens', 0) or 0
        self.calls += 1
        self._has_openai_details = True

    def to_dict(self) -> dict:
        d = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "llm_calls": self.calls,
            "model": self.model,
            "user_id": self.user_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if self._has_openai_details:
            d["reasoning_tokens"] = self.reasoning_tokens
            d["cached_tokens"] = self.cached_tokens
        return d


def classify_error(e) -> str:
    """에러 종류에 따라 사용자 친화적 메시지를 반환한다. 기술 상세는 로그에만 남긴다."""
    error_type = type(e).__name__

    # OpenAI API 에러
    try:
        import openai
        if isinstance(e, openai.RateLimitError):
            return "요청이 많아 잠시 후 다시 시도해주세요."
        if isinstance(e, openai.AuthenticationError):
            return "AI 서비스 인증 오류입니다. 관리자에게 문의해주세요."
        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
            return "AI 서비스 응답 시간이 초과되었습니다. 다시 질문해주세요."
        if isinstance(e, openai.BadRequestError):
            return "요청 처리 오류입니다. 질문을 줄여서 다시 시도해주세요."
        if isinstance(e, openai.InternalServerError):
            return "AI 서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    except ImportError:
        pass

    # llama-server HTTP 에러 (requests 라이브러리)
    try:
        import requests
        if isinstance(e, requests.exceptions.HTTPError):
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 503:
                return ("로컬 AI 서버가 혼잡하거나 대화 컨텍스트가 한도를 초과했습니다. "
                        "새 대화를 시작하거나 잠시 후 다시 시도해주세요.")
            if status == 413:
                return "입력이 너무 깁니다. 새 대화를 시작하거나 질문을 줄여서 다시 시도해주세요."
            if status == 500:
                return "로컬 AI 서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
            return "로컬 AI 서버 응답 오류입니다. 잠시 후 다시 시도해주세요."
        if isinstance(e, requests.exceptions.Timeout):
            return "로컬 AI 서버 응답 시간이 초과되었습니다. 다시 질문해주세요."
        if isinstance(e, requests.exceptions.ConnectionError):
            return "로컬 AI 서버에 연결할 수 없습니다. 관리자에게 문의해주세요."
    except ImportError:
        pass

    # 메모리 / GPU 에러
    if isinstance(e, MemoryError):
        return "서버 메모리가 부족합니다. 잠시 후 다시 시도해주세요."
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return "GPU 메모리가 부족합니다. 질문을 줄여서 다시 시도해주세요."

    # 네트워크 / 연결 에러
    if isinstance(e, (ConnectionError, TimeoutError)):
        return "서비스 연결에 실패했습니다. 잠시 후 다시 시도해주세요."

    return "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."


def api_call_with_retry(func, max_retries=3, base_delay=1.0):
    """OpenAI API 호출에 지수 백오프 재시도를 적용한다.

    - 재시도 대상: RateLimitError(429), APITimeoutError, APIConnectionError, InternalServerError(500)
    - 즉시 실패: AuthenticationError, BadRequestError, PermissionDeniedError
    - 지수 백오프: base_delay * 2^attempt (1초 → 2초 → 4초)
    """
    import openai

    RETRYABLE = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except RETRYABLE as e:
            last_error = e
            if attempt == max_retries:
                logger.error(f"API 호출 최종 실패 ({attempt + 1}/{max_retries + 1}): {e}")
                raise

            delay = base_delay * (2 ** attempt)
            # RateLimitError의 경우 Retry-After 헤더 존중
            if isinstance(e, openai.RateLimitError):
                retry_after = getattr(e.response, 'headers', {}).get('retry-after')
                if retry_after:
                    delay = max(delay, float(retry_after))

            logger.warning(f"API 호출 재시도 ({attempt + 1}/{max_retries + 1}), {delay:.1f}초 대기: {type(e).__name__}: {e}")
            time.sleep(delay)
        except (openai.AuthenticationError, openai.BadRequestError, openai.PermissionDeniedError):
            raise

    raise last_error


async def api_call_with_retry_async(func, max_retries=3, base_delay=1.0):
    return await asyncio.to_thread(api_call_with_retry, func, max_retries, base_delay)


def sanitize_history(history: list, max_messages: int = 10, max_tokens: int = 2000, max_content_len: int = 2000, current_query: str = None) -> str:
    """외부 히스토리를 토큰 제한 내에서 압축하여 문자열로 변환한다.

    - base64/아이콘 데이터 제거
    - 메시지당 max_content_len 자 제한
    - 전체 토큰 수 max_tokens 제한 (최신 메시지 우선)
    - 연속 동일 role+content 메시지 dedupe (client 가 같은 user 발화를 2번 push 하는 케이스 방어)
    - current_query 가 주어지면 history 의 마지막 user 가 그와 같으면 제외
      (prompt 의 `Question: {query}` 와 `사용자: {query}` 중첩 방지)
    """
    if not history or not isinstance(history, list):
        return ""

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = None

    # 최근 메시지만 대상으로, 역순 처리 (최신 우선 토큰 예산 배분)
    recent = history[-max_messages:]

    # current_query 와 동일한 마지막 user 발화 제거
    if current_query and recent:
        last = recent[-1]
        if isinstance(last, dict) and last.get('role') == 'user':
            last_content = last.get('content', '')
            if isinstance(last_content, str) and last_content.strip() == current_query.strip():
                recent = recent[:-1]

    cleaned = []
    for msg in recent:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        # base64/아이콘 데이터 제거
        content = re.sub(r'"icon"\s*:\s*"[A-Za-z0-9+/=]{100,}"', '"icon":"[removed]"', content)
        content = re.sub(r'\b[A-Za-z0-9+/=]{500,}\b', '[base64_data_removed]', content)

        if len(content) > max_content_len:
            content = content[:max_content_len] + "...[truncated]"

        # 연속 동일 role+content dedupe
        if cleaned and cleaned[-1]["role"] == role and cleaned[-1]["content"] == content:
            continue

        cleaned.append({"role": role, "content": content})

    # 토큰 예산에 맞춰 최신 메시지부터 포함
    result_parts = []
    token_count = 0
    for msg in reversed(cleaned):
        msg_tokens = len(tokenizer.encode(msg["content"])) if tokenizer else len(msg["content"]) // 4
        if token_count + msg_tokens > max_tokens:
            break
        result_parts.insert(0, msg)
        token_count += msg_tokens

    # 문자열 변환
    history_text = ""
    for msg in result_parts:
        if msg["role"] == "user":
            history_text += f"사용자: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            history_text += f"AI: {msg['content']}\n\n"

    return history_text


class BaseModelHandler(ABC):
    """LLM 핸들러 추상 베이스.

    새 핸들러 추가 시 반드시 정의해야 할 것:
    - class attr `registry_key`: handler_registry.HANDLER_CLASSES 키와 일치하는 문자열
    - @abstractmethod 표시된 메서드들: generate_stream / generate_complete /
      _perform_translation (모델별 호출 방식이 본질적으로 달라 default 제공 불가)

    선택적으로 override 가능:
    - is_local, supports_streaming, error_response_mode, UNWANTED_PROMPT_TAGS (class attr)
    - process_question, process_question_error, _clean_prompt_content, get_search_status,
      get_model_info, clear_chat_history, get_last_sources (base default 있음)

    참고: process_question 은 generate_stream 을 answer_stream 으로 wrap 하는 공통
    boilerplate 라 base default 제공. generate_stream 만 구현하면 자동 동작.
    """

    # === 서브클래스가 override 해야 하는 class attrs ===
    registry_key: str = ""           # handler_registry.HANDLER_CLASSES 키와 sync
    is_local: bool = False           # 로컬 서버 띄우는 모델 여부 (run.sh 와 연관)
    supports_streaming: bool = True  # streaming 호출 가능 여부 (Bedrock 등 False 가능)
    error_response_mode: str = "complete"  # 에러 응답 형태: "stream"→answer_stream / "complete"→answer
    UNWANTED_PROMPT_TAGS: Tuple[str, ...] = (
        # superset — 모든 모델의 chat-template 토큰 (system_prompt 에선 strip 대상).
        # 자기 모델 토큰도 포함됨 (chat template 은 메시지 구조에서 자동 추가되므로
        # system_prompt 자체에 있으면 안 됨). 검증: /data/ai-agent/prompts/ 의
        # 어떤 파일도 이 토큰을 "설명/예시" 로 포함하지 않음 (2026-05-26).
        '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
        '<|begin_of_text|>', '<|end_of_text|>',
        '<|im_start|>', '<|im_end|>',
        '<|start|>', '<|end|>', '<|return|>', '<|channel|>', '<|message|>',
    )

    def __init_subclass__(cls, **kwargs):
        """서브클래스 정의 시점에 registry_key 누락 즉시 발견 (import 시 발동)."""
        super().__init_subclass__(**kwargs)
        if not cls.registry_key:
            raise TypeError(
                f"{cls.__name__} 는 'registry_key' class attr 를 선언해야 합니다 "
                f"(예: registry_key = \"mymodel\"). "
                f"handler_registry.HANDLER_CLASSES 키와 sync 시킬 것."
            )

    def __init__(self, rag_system=None, use_kg: bool = True, intent_analyzer=None):
        self.config_manager = ConfigManager()
        self.rag_system = rag_system
        self.use_kg = use_kg and rag_system and hasattr(rag_system, 'kg_available') and rag_system.kg_available

        self.intent_analyzer = intent_analyzer

        self.last_sources = []
        self.chat_session = ChatSession(max_messages=10)
        self.slack_max_length = 3000
        self.translation_cache = {}

        self.last_response_time = 0
        self.last_concepts = []
        self.last_related_concepts = []

    # === Abstract: 서브클래스 반드시 구현 ===

    @abstractmethod
    async def generate_stream(self, request: Dict, user_id=None, channel_id=None,
                              connection=None, thread_ts=None, *args, **kwargs):
        """모델별 streaming 응답 생성. 모델 API/호출 방식 차이 때문에 default 제공 불가."""

    @abstractmethod
    async def generate_complete(self, request: Dict, user_id=None, channel_id=None,
                                 connection=None, thread_ts=None, *args, **kwargs) -> Dict:
        """모델별 complete 응답 생성."""

    @abstractmethod
    def _perform_translation(self, query: str, mode: str, preserved: Optional[dict] = None) -> str:
        """모델별 chat template 으로 번역 호출. prompt 구성이 모델마다 다름."""

    def _filter_docs_by_app_context(self, docs: List[Tuple], app_context: str) -> List[Tuple]:
        if not app_context:
            return docs

        installed_apps = []
        if isinstance(app_context, str):
            installed_apps = [app.strip() for app in app_context.split(',')]


        filtered_docs = []
        excluded_count = 0

        for doc_tuple in docs:
            file_key = doc_tuple[0] if len(doc_tuple) > 0 else ""
            content = doc_tuple[1] if len(doc_tuple) > 1 else ""

            should_include = self._check_app_code_compatibility(file_key, content, installed_apps)

            if should_include:
                filtered_docs.append(doc_tuple)
            else:
                excluded_count += 1

        if excluded_count > 0:
            print(f"📱 [앱 필터링] 총 {excluded_count}개 문서 제외, {len(filtered_docs)}개 문서 통과")

        return filtered_docs

    def _check_app_code_compatibility(self, file_key: str, content: str, installed_apps: List[str]) -> bool:

        if hasattr(self, 'rag_system') and self.rag_system and hasattr(self.rag_system, 'document_metadata'):
            metadata = self.rag_system.document_metadata.get(file_key)
            if metadata and 'app_code' in metadata:
                app_code = metadata['app_code']

                if app_code is None:
                    return True

                print(installed_apps)
                if isinstance(app_code, str) and app_code.strip():
                    apps_str = str(installed_apps)
                    if f'"code": "{app_code}"' in apps_str:
                        print(f"📱 [앱 호환성] ✅ {file_key}: app_code={app_code} (설치됨)")
                        return True
                    else:
                        print(f"📱 [앱 호환성] ❌ {file_key}: app_code={app_code} (미설치)")
                        return False

                return True

        try:
            import yaml
            doc = yaml.safe_load(content)
            if not isinstance(doc, dict):
                return True

            app_code = doc.get('app_code')

            if app_code is None:
                return True

            if isinstance(app_code, str) and app_code.strip():
                apps_str = str(installed_apps)
                if f'"code": "{app_code}"' in apps_str:
                    print(f"📱 [앱 호환성] ✅ {file_key}: app_code={app_code} (설치됨)")
                    return True
                else:
                    print(f"📱 [앱 호환성] ❌ {file_key}: app_code={app_code} (미설치)")
                    return False

            return True

        except Exception as e:
            print(f"📱 [앱 필터링] 호환성 확인 오류 ({file_key}): {e}")
            return True

    def _clean_prompt_content(self, prompt: str) -> str:
        """프롬프트에서 [CURRENT_SYSTEM_DATE] 섹션과 chat-template 토큰 제거.

        load_prompt_from_file() 가 date_info 를 항상 추가하므로 여기서 다시 strip.
        토큰은 self.UNWANTED_PROMPT_TAGS (서브클래스가 override 가능, base 는 superset).

        주의: prompt 안에 "토큰 설명/예시" 가 평문으로 들어있으면 같이 삭제됨.
        현재 /data/ai-agent/prompts/ 의 어떤 yaml/text 도 그런 패턴을 포함하지 않음
        (grep 검증 완료 — 2026-05-26).
        """
        try:
            cleaned_prompt = prompt.strip()

            lines = cleaned_prompt.split('\n')
            filtered_lines = []
            skip_section = False
            for line in lines:
                if '[CURRENT_SYSTEM_DATE]' in line or '🔄 **현재 시스템 날짜 정보:**' in line:
                    skip_section = True
                    continue
                if skip_section and (not line.strip() or line.strip().startswith('#')):
                    skip_section = False
                    if line.strip():
                        filtered_lines.append(line)
                    continue
                if not skip_section:
                    filtered_lines.append(line)

            result = '\n'.join(filtered_lines).strip()

            for token in self.UNWANTED_PROMPT_TAGS:
                result = result.replace(token, '')

            result = re.sub(r'\n\s*\n\s*\n', '\n\n', result).strip()
            return result
        except Exception as e:
            logger.error(f"프롬프트 정리 오류: {e}")
            return prompt.strip()

    async def process_question_error(self, request: Dict, user_id=None, channel_id=None,
                                       connection=None, thread_ts=None) -> Dict:
        """에러 응답 default. error_response_mode 에 따라 분기:
        - "stream" → generate_stream_error 를 answer_stream 으로 wrap (OpenAI)
        - "complete" → generate_complete_error 를 answer 로 wrap (gpt-oss/qwen/gemma/llama)
        해당 error 메서드가 없으면 except 로 떨어져 classify_error 메시지 반환.
        """
        try:
            self.last_sources = []
            if self.error_response_mode == "stream":
                return {
                    "answer_stream": self.generate_stream_error(request, user_id, channel_id, connection, thread_ts),
                    "context_sources": self.last_sources,
                }
            result = await self.generate_complete_error(request, user_id, channel_id, connection, thread_ts)
            return {"answer": result, "context_sources": []}
        except Exception as e:
            logger.error(f"{self.__class__.__name__} process_question_error: {e}")
            return {"answer": classify_error(e), "context_sources": []}

    def _safe_analyze_qna_subtype(self, query: str) -> str:
        try:
            if (self.intent_analyzer and
                hasattr(self.intent_analyzer, 'analyze_qna_subtype_sync')):
                return self.intent_analyzer.analyze_qna_subtype_sync(query)
            else:
                if re.search(r'CVE-\d{4}-\d{4,7}', query, re.IGNORECASE):
                    return 'CVE'
                else:
                    return 'NORMAL'
        except Exception as e:
            logger.warning(f"⚠️ QNA 서브타입 분석 실패: {e}")
            return 'NORMAL'

    def _validate_and_fix_translation(self, original: str, translation: str) -> str:

        critical_keywords = {
            '공격': 'attack',
            '로그인': 'login',
            '접속': 'access',
            '실패': 'failure',
            '침입': 'intrusion',
            '차단': 'block',
            '탐지': 'detect',
            '분석': 'analyze',
            '시도': 'attempt',
        }

        original_lower = original.lower()
        translation_lower = translation.lower()
        missing_keywords = []

        for korean, english in critical_keywords.items():
            if korean in original_lower and english not in translation_lower:
                missing_keywords.append(english)
                logger.warning(f"번역에서 누락된 핵심 키워드: {korean} → {english}")

        if missing_keywords:
            fixed_translation = " ".join(missing_keywords) + " " + translation
            logger.info(f"번역 보완: '{translation}' → '{fixed_translation}'")
            return fixed_translation

        return translation

    def extract_technical_content(self, text, include_cve=True):
        vulnerability_folders = {
            'admin': 100, 'administrator': 100, 'upload': 100, 'uploads': 100,
            'filemanager': 100, 'file-manager': 100, 'config': 100, 'configuration': 100,
            'backup': 100, 'backups': 100, 'shell': 100, 'webshell': 100,
            'install': 100, 'installation': 100, 'setup': 100,

            'editor': 50, 'fckeditor': 50, 'ckeditor': 50, 'tinymce': 50,
            'plugins': 50, 'plugin': 50, 'includes': 50, 'include': 50,
            'temp': 50, 'tmp': 50, 'test': 50, 'testing': 50, 'debug': 50,

            'assets': 25, 'js': 25, 'css': 25, 'images': 25, 'img': 25,
            'templates': 25, 'themes': 25, 'api': 25, 'ajax': 25
        }

        patterns = [
            r'(?:GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+(?![가-힣])[^\s]+',
            r'/[^\s\'"]*',
            r'\.\./?[^\s]*',
            r'https?://[^\s]+',
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            r'\b[a-zA-Z0-9]+\.[a-zA-Z]{2,4}\b',
            r'\?[a-zA-Z0-9_]+=\w*',
            r'\$\{.*?j.*?n.*?d.*?i.*?:[a-zA-Z]+://[^\s\}]+\}',
            r'(?:\$\{(?:[^{}]|\$\{[^{}]*\})*\}|[a-zA-Z0-9_\-])*j(?:[^a-zA-Z0-9]{0,15}|\$\{[^{}]*\})*n(?:[^a-zA-Z0-9]{0,15}|\$\{[^{}]*\})*d(?:[^a-zA-Z0-9]{0,15}|\$\{[^{}]*\})*i(?:[^:]{0,15}|\$\{[^{}]*\})*:[a-zA-Z]+://[^\s\}]+',
            r'\$\{(?:\$\{[^{}]*\}|[^{}])*j(?:\$\{[^{}]*\}|[^{}])*n(?:\$\{[^{}]*\}|[^{}])*d(?:\$\{[^{}]*\}|[^{}])*i(?:\$\{[^{}]*\}|[^{}])*:[^}]*',
            r'(?is)\$\{\s*\$\{.*?j.*?n.*?d.*?i.*?:.*?l.*?d.*?a.*?p.*?\}.*?://.*',
            r'(?<=\/)[^\/\s\'"?#]+(?=\/|$)',
            r'\/([^\/\s]+)\/([^\/\s]+)\/',
            r'\.[a-zA-Z0-9]+$'
        ]

        if include_cve:
            patterns.insert(0, r'CVE-\d{4}-\d{4,7}')

        all_matches = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                item = match.group()
                if item:
                    all_matches.append((match.start(), item))

        seen = set()
        unique_matches = []
        for start, item in all_matches:
            if item not in seen:
                seen.add(item)

                score = 0
                item_lower = item.lower()
                for folder, points in vulnerability_folders.items():
                    if folder in item_lower:
                        score = max(score, points)

                if item.startswith('/') and len(item) > 20:
                    score += 10
                if any(ext in item.lower() for ext in ['.php', '.jsp', '.asp']):
                    score += 20
                if any(risk in item.lower() for risk in ['upload', 'admin', 'config']):
                    score += 30

                unique_matches.append((score, start, item))

        unique_matches.sort(key=lambda x: (-x[0], x[1], -len(x[2])))

        top_matches = unique_matches[:5]
        result = [item[2] for item in top_matches]

        return '\n'.join(result) if result else None

    def translate_query_for_search(self, query: str, intent: str) -> str:

        print(f"🔍 [번역함수] 호출됨: '{query}' (캐시크기: {len(self.translation_cache)})")

        option_config = self.config_manager.get_option_config()
        translation_enabled = option_config.get('translation', 'True').lower() == 'true'

        sub_intent = None
        try:
            if intent == 'QNA' or intent == 'PLAYBOOK':
                # QNA/PLAYBOOK에서도 영어 감지 체크
                if not re.search(r'[가-힣]', query):
                    print(f"🔍 [번역함수] QNA/PLAYBOOK 영어 감지, 건너뜀")
                    return query, sub_intent

                if intent == 'PLAYBOOK':
                    sub_intent = 'NORMAL'
                    translation = self._perform_translation(query,'NORMAL')
                else:
                    # BGE-M3 모드 확인을 가장 먼저 수행
                    is_bge_mode = False
                    try:
                        use_bge = self.config_manager.config.get('embedding', 'use_bge_mode', fallback='False')
                        is_bge_mode = (use_bge == 'True')
                    except:
                        is_bge_mode = False

                    if is_bge_mode:
                        # BGE-M3 모드에서는 전용 번역 함수 사용
                        print(f"🔍 [BGE-M3 모드] 전용 번역 루트 실행")
                        translated_query, sub_intent = self._translate_for_bge_mode(query)
                        # 기술패턴 정보를 어딘가에 저장하거나 전달해야 함
                        return translated_query, sub_intent
                    else:
                        qna_intent = self._safe_analyze_qna_subtype(query)
                        print(f"🔍 [QNA 의도] {qna_intent}")

                        if qna_intent.upper() == 'CVE':
                            check_pattern = self.extract_technical_content(query, True)
                            sub_intent = 'CVE'
                            if check_pattern == None:
                                translation = self._perform_translation(query,'CVE')
                                print(f"========= CVE 번역 완료 =========\n{translation}\n===============================")
                                return translation, sub_intent
                            else:
                                korean_parts = re.findall(r'[가-힣\s]+(?:찾아|분석|확인|알려|보여|해줘|주세요)[^\n]*', query)

                                if korean_parts:
                                    korean_text = ' '.join(korean_parts).strip()
                                    translated_korean = self._perform_translation(korean_text, 'CVE')
                                    combined_result = f"{translated_korean}\n{check_pattern}"
                                    print(f"========= 혼합 결과 =========\n{combined_result}\n===============================")
                                    return combined_result, sub_intent
                                else:
                                    print(f"========= PATTERN =========\n{check_pattern}\n===============================")
                                    return check_pattern, sub_intent
                        else:
                            sub_intent = 'NORMAL'

                            # GENERAL: 로그 패턴 기반으로 자연어와 로그 분리
                            def is_log_line(line):
                                log_patterns = [
                                    r'(?:CEF|LEEF):\d+\|',                      # CEF/LEEF 로그
                                    r'^<\d+>',                                  # Syslog 우선순위
                                    r'^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',   # "Dec 26 15:55:27" 형식
                                    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',   # ISO 타임스탬프
                                    r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', # "2024-01-01 12:00:00" 형식
                                    r':\s*\{.*\}',                              # JSON 로그
                                    r'device_id=|devname=|log_id=',             # 디바이스 정보
                                    r'src=.*dst=|src_ip=.*dst_ip=',             # 네트워크 로그
                                    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b.*\d+', # IP + 포트 패턴
                                ]
                                return any(re.search(pattern, line) for pattern in log_patterns)

                            lines = query.strip().split('\n')
                            natural_lines = []
                            log_lines = []

                            for line in lines:
                                line_stripped = line.strip()
                                if is_log_line(line_stripped):
                                    log_lines.append(line_stripped)
                                else:
                                    natural_lines.append(line_stripped)

                            if natural_lines and log_lines:
                                # 자연어 부분만 번역 (한글/영어 무관)
                                natural_text = ' '.join(natural_lines)
                                translated_text = self._perform_translation(natural_text, 'NORMAL')
                                log_content = '\n'.join(log_lines)
                                combined_result = f"{translated_text}\n{log_content}"
                                print(f"========= 패턴 기반 분리 (GENERAL) =========\n{combined_result}\n===============================")
                                return combined_result, sub_intent
                            else:
                                # 자연어만 있거나 로그만 있는 경우 전체 번역
                                translation = self._perform_translation(query,'NORMAL')
            else:
                if not re.search(r'[가-힣]', query):
                    print(f"🔍 [번역함수] 영어 감지, 건너뜀")
                    return query, sub_intent

                translation = self._perform_translation(query,'SUMMARY')

            if translation and self._is_valid_translation(query, translation):
                validated_translation = self._validate_and_fix_translation(query, translation)

                logger.info(f"검색용 번역: '{query}' -> '{validated_translation}'")
                self.translation_cache[query] = validated_translation
                return validated_translation, sub_intent
            else:
                dict_translation = self._enhanced_dictionary_translate(query)
                validated_dict = self._validate_and_fix_translation(query, dict_translation)

                logger.info(f"사전 번역 (검증됨): '{query}' -> '{validated_dict}'")
                self.translation_cache[query] = validated_dict
                return validated_dict, sub_intent

        except Exception as e:
            logger.error(f"번역 중 오류: {str(e)}, 사전 번역 사용")
            dict_translation = self._enhanced_dictionary_translate(query)
            validated_dict = self._validate_and_fix_translation(query, dict_translation)

            logger.info(f"오류 복구 번역 (검증됨): '{query}' -> '{validated_dict}'")
            self.translation_cache[query] = validated_dict
            return validated_dict, None

    def _is_valid_translation(self, original: str, translation: str) -> bool:
        bad_starters = ['here is', 'this is', 'the following', 'below is']
        return not any(translation.lower().startswith(bad) for bad in bad_starters)

    def _enhanced_dictionary_translate(self, text: str) -> str:

        critical_verbs = {
            "공격한": "attack", "공격하는": "attack", "공격": "attack",
            "공격들": "attack", "공격자": "attacker",
            "로그인": "login", "접속": "access", "접속한": "access",
            "실패한": "failure", "실패": "failure", "실패들": "failure",
            "침입한": "intrusion", "침입": "intrusion",
            "차단한": "block", "차단": "block",
            "탐지한": "detect", "탐지": "detect",
            "분석한": "analyze", "분석": "analyze",
            "시도한": "attempt", "시도": "attempt",
            "추천": "recommend", "추천해": "recommend", "추천해줘": "recommend",
            "추천하는": "recommend", "추천한": "recommend", "추천들": "recommend",
            "권장": "recommend", "권장해": "recommend", "권장하는": "recommend",
            "제안": "suggest", "제안해": "suggest", "제안하는": "suggest"
        }

        technical_dict = {
            "카펙": "CAPEC", "케이펙": "CAPEC", "캐펙": "CAPEC",
            "미트리": "MITRE", "미터": "MITRE", "마이터": "MITRE",
            "그룹": "group", "기술": "technique", "테크닉": "technique",
            "트래픽": "traffic", "패킷": "packet",
            "프로토콜": "protocol", "이벤트": "event", "이벤트들": "event",
            "로그": "log", "로그들": "log",

            "서버": "server", "서버들": "server",
            "데이터베이스": "database", "애플리케이션": "application",
            "서비스": "service", "서비스들": "service",

            "3월": "march", "1일": "day", "일": "day", "월": "month",
            "부터": "from", "까지": "to", "동안": "during",
            "최근": "recent", "오늘": "today", "어제": "yesterday",
            "지금까지": "until now", "현재까지": "until now",

            "찾아": "find", "찾아봐": "find", "찾아줘": "find",
            "보여": "show", "보여줘": "show",
            "알려": "tell", "알려줘": "tell",
            "확인": "check", "확인해": "check",

            "가장": "most", "많이": "most", "많은": "many", "많이": "many",
            "상위": "top", "순위": "rank",
            "개수": "count", "수량": "count"
        }

        combined_dict = {**critical_verbs, **technical_dict}

        remove_words = {
            "어떻게", "무엇을", "무엇", "언제", "어디서", "왜",
            "방법", "방식", "하는", "되는", "있는", "없는",
            "해줘", "줘", "요", "습니다", "입니다"
        }

        words = text.split()
        translated_words = []

        for word in words:
            clean_word = re.sub(r'[^\w가-힣]', '', word)

            if not clean_word or clean_word in remove_words:
                continue

            if re.match(r'^[0-9\-:]+$', clean_word) or re.match(r'^[a-zA-Z]+$', clean_word):
                translated_words.append(clean_word)
                continue

            translated = None

            if clean_word.lower() in combined_dict:
                translated = combined_dict[clean_word.lower()]

            if not translated:
                for postfix in ['을', '를', '이', '가', '에', '의', '로', '으로', '와', '과', '부터', '까지', '들']:
                    if clean_word.endswith(postfix):
                        base_word = clean_word[:-len(postfix)]
                        if base_word in combined_dict:
                            translated = combined_dict[base_word]
                            break

            if translated:
                translated_words.append(translated)
            elif len(clean_word) >= 2:
                translated_words.append(clean_word)

        result = " ".join(translated_words).strip()

        if result != text:
            logger.info(f"사전 키워드 번역: '{text}' → '{result}'")

        return result

    def load_prompt_from_file(self, prompt_path: str) -> str:

        now = datetime.datetime.now()
        current_date = now.strftime('%Y%m%d')
        current_year = str(now.year)
        current_datetime = now.strftime('%Y%m%d%H%M%S')
        current_readable = now.strftime('%Y년 %m월 %d일 (%A)')

        minus_1d = (now - datetime.timedelta(days=1)).strftime('%Y%m%d')
        minus_7d = (now - datetime.timedelta(days=7)).strftime('%Y%m%d')
        minus_30d = (now - datetime.timedelta(days=30)).strftime('%Y%m%d')
        current_time = now.strftime('%H%M%S')

        date_info = f"""
            [CURRENT_SYSTEM_DATE]
            🔄 **현재 시스템 날짜 정보:**
            - 오늘 날짜: {current_date} ({current_readable})
            - 현재 년도: {current_year}
            - 현재 시각: {current_datetime}

            **시간 변수 정의:**
            - $CUR_DATE$: {current_date}
            - $CUR_TIME$: {current_time}
            - $CUR_DATE_MINUS_1D$: {minus_1d}
            - $CUR_DATE_MINUS_7D$: {minus_7d}
            - $CUR_DATE_MINUS_30D$: {minus_30d}
            """
        try:
            if not os.path.exists(prompt_path):
                logger.warning(f"프롬프트 파일 {prompt_path}를 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
                return "You are a helpful assistant that answers questions based on provided documents. If you don't know the answer, say so honestly."

            file_ext = os.path.splitext(prompt_path)[1].lower()

            if file_ext in ['.yaml', '.yml']:
                import yaml
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_yaml = yaml.safe_load(f)
                    if 'prompt' not in prompt_yaml:
                        logger.warning("YAML 프롬프트 파일에 'prompt' 필드가 없습니다. 기본 프롬프트를 사용합니다.")
                        return "You are a helpful assistant."
                    prompt_content = prompt_yaml['prompt']
            else:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_content = f.read()

            prompt_content += date_info

            return prompt_content

        except Exception as e:
            logger.error(f"프롬프트 파일 로드 중 오류: {str(e)}")
            return "You are a helpful assistant."

    # 베이스 시스템 프롬프트 슬롯 후보 — 핸들러별 키가 없으면 gpt_oss → gpt 로 폴백.
    # handler_gemma.__init__ 의 도출 순서와 동일하게 유지할 것.
    _BASE_PROMPT_FALLBACKS = ('gpt_oss', 'gpt')

    def reload_system_prompt(self) -> str:
        """config.ini 를 다시 읽어 베이스 시스템 프롬프트를 재적용한다 (카트리지 장착 반영).

        반환값은 **재적용된 프롬프트 경로**(실패 시 빈 문자열). 호출부가 truthy 로 성공 판정도
        하고, 무엇이 실제로 물렸는지 응답에 실어 검증할 수도 있게 하기 위함이다
        (리로드가 됐는지를 모델 어투로 추측해야 했던 문제).

        intent 별 프롬프트(qna·action·plan·playbook)는 요청 시점마다
        load_prompt_from_file() 로 다시 읽으므로 이미 hot 이다. 시작 시점에 한 번만
        읽어 인스턴스에 굳는 것은 이 베이스 프롬프트뿐이라, 여기만 되살리면 재시작이 불필요하다.
        """
        try:
            # ConfigManager.load_config() 는 configparser.read() 재호출 = config.ini 재파싱
            if getattr(self, 'config_manager', None) is not None:
                self.config_manager.load_config()
            else:
                from config_utils import ConfigManager
                self.config_manager = ConfigManager()

            prompts_cfg = self.config_manager.get_prompt_config()
            path = prompts_cfg.get(getattr(self, 'handler_type', '') or '')
            for key in self._BASE_PROMPT_FALLBACKS:
                if path:
                    break
                path = prompts_cfg.get(key)

            if not path:
                logger.warning("reload_system_prompt: [prompts] 에서 베이스 프롬프트 경로를 찾지 못함")
                return ""

            self.system_prompt = self.load_prompt_from_file(path)
            logger.info(f"✅ 시스템 프롬프트 리로드: {path}")
            return path
        except Exception as e:
            logger.error(f"reload_system_prompt 실패: {e}")
            return ""

    def system_prompt_fingerprint(self) -> str:
        """현재 물려 있는 시스템 프롬프트의 지문 — 리로드가 실제로 반영됐는지 외부에서 확인용.

        load_prompt_from_file() 이 매번 현재 날짜를 덧붙이므로 날짜 블록은 지문에서 제외한다.
        """
        import hashlib
        body = (getattr(self, 'system_prompt', '') or '').split('[CURRENT_SYSTEM_DATE]')[0]
        return hashlib.sha256(body.encode('utf-8')).hexdigest()[:12]

    # ──────────────────────────────────────────────────────────────────────
    # PII 게이트 (security-review.md S-6) — 전 핸들러 공통.
    #
    # 원래 handler_gemma 에만 있었다. 그 상태에선 외부 API 핸들러(openai·claude)를 쓰는 배포에서
    # **원문 PII 가 그대로 외부로 나간다** — 온프레미스 제품에서 가장 아픈 조합이라 여기로 올렸다.
    # 게이트는 config `[pii] pii_mode` 로 켠다(기본 off).
    #
    # 초기화는 **지연(lazy)** 이다. 게이트가 self.rag_system.bge_model 을 재사용하는데
    # 그 배선 시점이 핸들러마다 달라, __init__ 순서에 의존하지 않도록 첫 사용 시점에 만든다.
    # ──────────────────────────────────────────────────────────────────────

    def _init_pii_gate(self):
        """LLM 입력 전 PII 치환 게이트 초기화. config [pii] pii_mode 로 on/off.
        의미계층은 이미 로드된 BGE-M3(self.rag_system.bge_model)를 재사용(새 로딩 X)."""
        pii_cfg = self.config_manager.get_pii_config()
        self.pii_mode = pii_cfg.get('pii_mode', False)
        self.pii_restore = pii_cfg.get('restore', True)
        self.pii = None
        if not self.pii_mode:
            print("🔒 [PII] pii_mode=False — PII 게이트 비활성화")
            return
        try:
            from aibot_PII import AibotPII, embedder_from_flagmodel
            _embedder = None
            _bge = getattr(self.rag_system, "bge_model", None) if self.rag_system else None
            if _bge is not None:
                try:
                    _embedder = embedder_from_flagmodel(_bge)
                except Exception as e:
                    print(f"⚠️ [PII] 의미계층 임베더 연결 실패(정규식+NER 만 동작): {e}")
            # 의미계층(semantic)은 현재 오탐이 많아 보류(enable_semantic=False).
            self.pii = AibotPII(
                embedder=_embedder,
                enable_semantic=False,
                verbose=pii_cfg.get('verbose', False),
            )
            print(f"🔒 [PII] pii_mode=True — PII 게이트 활성화 ({type(self).__name__})")
        except Exception as e:
            print(f"❌ [PII] 게이트 초기화 실패 — 비활성화로 폴백: {e}")
            self.pii_mode = False
            self.pii = None

    def _pii_ready(self) -> bool:
        """게이트가 아직 없으면 만든 뒤, 마스킹을 실제로 할 상태인지 답한다."""
        if not hasattr(self, "pii_mode"):
            try:
                self._init_pii_gate()
            except Exception as e:
                print(f"❌ [PII] 지연 초기화 실패 — 비활성화: {e}")
                self.pii_mode, self.pii = False, None
        return bool(getattr(self, "pii_mode", False)) and getattr(self, "pii", None) is not None

    def _pii_mask_input(self, request, query):
        """사용자 입력(query) + 대화 history 를 RAG·프롬프트 조립 전에 함께 마스킹.
        - 같은 PII 값은 query/history 전체에서 같은 토큰(공유 맵, mask_chat).
        - history 는 클라이언트가 매 턴 원본 PII 를 다시 보낼 수 있으므로 매번 in-place 마스킹.
        - 복원맵을 self._last_pii_map 에 보관(restore=True 일 때 응답 복원에 사용).
        - request['question'] 도 갱신 → RAG 검색도 마스킹된 질의 사용."""
        self._last_pii_map = {}
        if not self._pii_ready():
            return query

        history = request.get("history", []) if isinstance(request, dict) else []
        msgs, hist_idx = [], []
        if isinstance(history, list):
            for i, h in enumerate(history):
                if isinstance(h, dict) and isinstance(h.get("content"), str) and h["content"].strip():
                    msgs.append({"role": h.get("role", "user"), "content": h["content"]})
                    hist_idx.append(i)
        msgs.append({"role": "user", "content": query or ""})

        masked_msgs, mapping = self.pii.mask_chat(msgs)
        self._last_pii_map = mapping

        # history in-place 갱신 (이전 턴 PII 가 프롬프트로 재유입되는 것 차단)
        for j, i in enumerate(hist_idx):
            history[i]["content"] = masked_msgs[j]["content"]

        masked_query = masked_msgs[-1]["content"]
        if isinstance(request, dict):
            request["question"] = masked_query
        return masked_query

    def _pii_mask_messages(self, messages: List[Dict]) -> List[Dict]:
        """OpenAI 호환 messages 배열 마스킹 (agent_complete 경로). 복원맵은 _last_pii_map."""
        self._last_pii_map = {}
        if not self._pii_ready() or not isinstance(messages, list):
            return messages
        idx = [i for i, m in enumerate(messages)
               if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip()]
        if not idx:
            return messages
        masked, mapping = self.pii.mask_chat([{"role": messages[i].get("role", "user"),
                                               "content": messages[i]["content"]} for i in idx])
        self._last_pii_map = mapping
        out = list(messages)
        for j, i in enumerate(idx):
            out[i] = {**messages[i], "content": masked[j]["content"]}
        return out

    def _pii_unmask(self, text: str, mapping: dict = None) -> str:
        """비스트리밍 응답의 토큰(<ENTITY_n>)을 원본값으로 역치환.
        restore=False 면 복원을 생략한다 — 답변에 토큰이 남아 마스킹 동작을 눈으로 검증할 수 있다."""
        m = mapping if mapping is not None else getattr(self, "_last_pii_map", None)
        if not (getattr(self, "pii_mode", False) and getattr(self, "pii_restore", True)
                and getattr(self, "pii", None) is not None and m):
            return text
        try:
            return self.pii.unmask(text, m)
        except Exception as e:
            logger.warning(f"[PII] 복원 실패(원문 유지): {e}")
            return text

    def _pii_stream_restorer(self):
        """스트리밍 복원기를 만들어 돌려준다: emit(piece, final=False) -> 방출할 문자열.

        토큰(`<KR_IP_1>`)이 청크 경계에서 쪼개지면 역치환이 실패하므로, 닫히지 않은 '<' 이후는
        꼬리 버퍼에 잠시 보류했다가 다음 청크와 합쳐 복원한다. 마지막엔 final=True 로 비운다.
        """
        on = (getattr(self, "pii_mode", False) and getattr(self, "pii_restore", True)
              and getattr(self, "pii", None) is not None and getattr(self, "_last_pii_map", None))
        state = {"pending": ""}

        def emit(piece: str, final: bool = False) -> str:
            if not on:
                return piece
            buf = state["pending"] + (piece or "")
            if final:
                state["pending"] = ""
                return self.pii.unmask(buf, self._last_pii_map)
            lt = buf.rfind('<')
            if lt != -1 and '>' not in buf[lt:]:
                seg, state["pending"] = buf[:lt], buf[lt:]
            else:
                seg, state["pending"] = buf, ""
            return self.pii.unmask(seg, self._last_pii_map)

        return emit

    def clear_chat_history(self):
        self.chat_session.clear()
        return "대화 기록이 초기화되었습니다."

    def truncate_for_slack(self, text: str) -> str:
        if len(text) <= self.slack_max_length:
            return text

        truncated = text[:self.slack_max_length-100]
        truncated += "... (더 긴 응답은 생략되었습니다)"

        return truncated

    async def get_related_documents(self, request: Dict, *args, **kwargs) -> Tuple[str, List[str], List[Dict], List]:
        # 단계별 소요시간 추적
        _step_times = {}

        if isinstance(request, dict) and request.get("app_context"):
            print(f"📱 [앱 컨텍스트] 설치된 앱: {request.get('app_context')}")

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            temp_prompt = request.get("temp_prompt", None)
            history = request.get("history", [])
            app_context = request.get("app_context", None)
            sub_id = request.get("sub_id", None)
        else:
            query = str(request)
            question_type = None
            temp_prompt = None
            history = []
            app_context = None
            sub_id = None

        if sub_id is None:
            raise ValueError("request에 sub_id가 필요합니다")

        combined_query = self._get_combined_query_from_history(history, query)
        if combined_query != query:
            print(f"🔄 원본 질문 추적: '{query}' → '{combined_query}'")
            request_copy = dict(request) if isinstance(request, dict) else {"question": query}
            request_copy["question"] = combined_query
            query = combined_query

        # === 조기 라우팅: 일상 질문 / 수정 요청 ===

        # 1) 일상 질문 → RAG 검색 건너뜀
        if self._is_casual_query(query):
            print(f"💬 [일상 질문] RAG 건너뜀: '{query[:50]}'")
            return "", [], [], [], None

        # 2) 수정 요청 → 이전 컨텍스트 재사용 (POST_ACTION은 기존 로직에 위임)
        _req_type = request.get("type", "").upper() if isinstance(request, dict) else ""
        if _req_type not in ('POST_ACTION', 'POST-ACTION'):
            if self._is_modification_request(history, query):
                if hasattr(self, 'last_sources') and self.last_sources:
                    print(f"✏️ [수정 요청] 이전 컨텍스트 재사용: '{query[:50]}'")
                    reused = self._reuse_previous_context(request)
                    if reused[0] is not None:
                        return reused

        # === 조기 라우팅 끝 ===

        if not self.rag_system:
            return "", [], [], [], None

        is_incremental_mode = temp_prompt is not None

        if question_type and question_type.upper() == 'QNA':
            score_threshold = 0.4
            min_documents = 5
            top_k = 5
            max_tokens = 6000
        else:
            score_threshold = 0.35
            min_documents = 3
            top_k = 5
            max_tokens = 6000

        start_time = time.time()
        try:
            _t0 = time.time()
            if question_type:
                final_result = question_type.upper()
                intent_info = {
                    'primary_intent': final_result,
                    'distribution': None,
                    'is_multi': False,
                    'reasoning': f"사용자 지정 의도: {final_result}",
                    'analysis_method': 'user_specified'
                }
            else:
                intent_info = None
                if self.intent_analyzer:
                    try:
                        heuristic_result = self.intent_analyzer._fallback_intent_analysis(query)

                        llm_result = None
                        if self.intent_analyzer.is_llm_connected():
                            try:
                                llm_result = await self.intent_analyzer.analyze_intent(query)
                            except Exception as e:
                                logger.warning(f"LLM 의도 분석 실패: {e}")

                        if llm_result:
                            primary_agreement = (heuristic_result.get_primary_intent() == llm_result.get_primary_intent())

                            if primary_agreement:
                                final_result = llm_result
                                analysis_type = "LLM(일치)"
                                logger.info(f"✅ 의도 일치: {final_result.get_primary_intent().upper()}")
                            else:
                                llm_weight = 0.4
                                h_weight = 0.6

                                combined_qna = llm_result.qna * llm_weight + heuristic_result.qna * h_weight
                                combined_plan = llm_result.plan * llm_weight + heuristic_result.plan * h_weight
                                combined_action = llm_result.action * llm_weight + heuristic_result.action * h_weight

                                total = combined_qna + combined_plan + combined_action
                                if total > 0:
                                    combined_qna /= total
                                    combined_plan /= total
                                    combined_action /= total

                                from aibot_intent_analyzer import IntentScore
                                final_result = IntentScore(
                                    qna=combined_qna,
                                    plan=combined_plan,
                                    action=combined_action,
                                    reasoning=f"가중평균(LLM:{llm_weight}, 휴리스틱:{h_weight})"
                                )

                                analysis_type = "하이브리드(불일치)"
                                logger.warning(f"⚠️ 의도 불일치")
                        else:
                            final_result = heuristic_result
                            analysis_type = "휴리스틱"

                        intent_info = {
                            'primary_intent': final_result.get_primary_intent(),
                            'distribution': final_result.get_intent_distribution(),
                            'is_multi': final_result.is_multi_intent(),
                            'reasoning': f"[{analysis_type}] {final_result.reasoning}",
                            'analysis_method': analysis_type
                        }

                    except Exception as e:
                        logger.warning(f"의도 분석 실패: {e}")
            _step_times['의도분석'] = time.time() - _t0

            concepts = []
            if hasattr(self.rag_system, 'extract_concepts_from_query'):
                concepts = self.rag_system.extract_concepts_from_query(query)

            selected_intent = question_type or (intent_info.get('primary_intent', '') if intent_info else '')

            if not selected_intent:
                selected_intent = 'GENERAL'

            request_type = request.get("type", "").upper() if isinstance(request, dict) else ""
            history = request.get("history", []) if isinstance(request, dict) else []
            query = request.get("question", "") if isinstance(request, dict) else ""
            is_clarify_followup = self._is_clarify_followup(history, query)

            if request_type in ['POST_ACTION', 'POST-ACTION'] or is_clarify_followup:
                action_type = "POST_ACTION" if request_type in ['POST_ACTION', 'POST-ACTION'] else "Clarify 후속"

                if hasattr(self, 'last_sources') and self.last_sources:
                    reused_result = self._reuse_previous_context(request)
                    if reused_result[0] is not None:
                        return reused_result
                else:
                    selected_intent = 'ACTION'

            _t1 = time.time()
            search_query = query
            sub_intent = None
            if (ConfigManager().get_option_config().get('translation') == 'True'):
                translated_query, sub_intent = await asyncio.to_thread(
                    self.translate_query_for_search, query, selected_intent.upper()
                )
                search_query = translated_query
            _step_times['번역'] = time.time() - _t1

            if intent_info:
                primary = intent_info['primary_intent']

                if intent_info['is_multi']:
                    top_k += 2
                    max_tokens += 2000

            extracted_concepts = []
            if self.use_kg and concepts:
                try:
                    for concept in concepts[:15]:
                        kg = self.rag_system._get_kg(sub_id)
                        if kg is None:
                            continue
                        concept_nodes = kg.find_nodes_by_name(concept)
                        for concept_node in concept_nodes[:5]:
                            extracted_concepts.append({
                                'concept': concept,
                                'node': concept_node
                            })

                    if extracted_concepts:
                        logger.info(f"지식 그래프: {len(extracted_concepts)}개 개념 발견")
                except Exception as e:
                    logger.error(f"개념 추출 오류: {str(e)}")

            logger.debug(f"🔎 [RAG Search] 구독 ID: {sub_id}로 검색 시작 - Intent: {selected_intent}, Query: {search_query[:50]}...")

            # CAPEC/MITRE 쿼리 감지 시 top_k 확장
            import re
            is_capec_mitre_query = False

            capec_patterns = [r'카펙\s*\d+', r'케이펙\s*\d+', r'캐펙\s*\d+', r'CAPEC\s*[-]?\d+', r'capec\s*[-]?\d+']
            mitre_patterns = [
                r'미트리\s*(그룹|기술|테크닉)\s*[GT]?\d+',
                r'미트리\s*[GT]\d+',
                r'mitre\s*(group|technique)\s*[GT]?\d+',
                r'mitre\s*[GT]\d+',
                r'MITRE\s*(GROUP|TECHNIQUE)\s*[GT]?\d+',
                r'MITRE\s*[GT]\d+',
                r'^[GT]\d+$'
            ]

            for pattern in capec_patterns + mitre_patterns:
                if re.search(pattern, search_query, re.IGNORECASE):
                    is_capec_mitre_query = True
                    break

            # CAPEC/MITRE 쿼리면 top_k를 5배로 확장 (QNA에서만)
            search_top_k = top_k * 2  # 기본값
            if is_capec_mitre_query and question_type and question_type.upper() == 'QNA':
                search_top_k = top_k * 10  # 100개까지
                print(f"  🎯 CAPEC/MITRE 쿼리 감지, top_k 확장: {top_k*2} → {search_top_k}")

            _t2 = time.time()
            if temp_prompt:
                similar_docs = await asyncio.to_thread(
                    self.rag_system.find_similar_docs_test,
                    search_query,
                    top_k=search_top_k,
                    max_token_count=max_tokens,
                    intent=selected_intent,
                    sub_intent=sub_intent,
                    temporary_embeddings=temp_prompt,
                    sub_id=sub_id
                )
                backup_docs = similar_docs[:10]
            else:
                similar_docs = await asyncio.to_thread(
                    self.rag_system.find_similar_docs,
                    search_query,
                    top_k=search_top_k,
                    max_token_count=max_tokens,
                    intent=selected_intent,
                    sub_intent=sub_intent,
                    sub_id=sub_id
                )
            _search_total = time.time() - _t2
            # BGE-M3에서 임베딩/Qdrant/파일읽기 개별 시간 수집
            _emb_t = getattr(self.rag_system, '_last_embed_time', None)
            _qdr_t = getattr(self.rag_system, '_last_qdrant_time', None)
            if _emb_t is not None and _qdr_t is not None:
                _step_times['임베딩'] = _emb_t
                _step_times['Qdrant'] = _qdr_t
            else:
                _step_times['검색'] = _search_total

            # BGE-M3 모드 확인 (BGE에서는 의도 기반 필터링 / 가중치 적용 안 함)
            _is_bge = False
            try:
                _is_bge = self.config_manager.config.get('embedding', 'use_bge_mode', fallback='False') == 'True'
            except:
                pass
            
            if _is_bge:
                logger.debug(f"🔍 [BGE-M3 모드] 의도기반 필터링 적용 건너뜀")
            else:
                filtered_docs = []
                for file_key, content, score in similar_docs:
                    if score >= score_threshold:
                        filtered_docs.append((file_key, content, score))
                    else:
                        logger.debug(f"문서 {file_key} 제외: 점수 {score:.3f} < 임계값 {score_threshold}")

                if app_context and question_type and question_type.upper() in ['ACTION', 'PLAN']:
                    filtered_docs = self._filter_docs_by_app_context(filtered_docs, app_context)

                if question_type and question_type.upper() != 'QNA':
                    print(f"🎯 [{question_type.upper()}] 사전 필터링 완료로 인해 마지막 폴더 필터링 생략")
                    similar_docs = filtered_docs

                elif question_type and question_type.upper() == 'QNA':
                    intent = question_type

                    intent_filtered_docs = []
                    excluded_count = 0
                    action_docs_included = 0

                    for file_key, content, score in filtered_docs:

                        is_db_mode = self.config_manager.get_db_config()

                        if is_db_mode:
                            path_parts = file_key.split('/')

                            if len(path_parts) >= 2 and path_parts[0].lower() in ['gpt-oss', 'gpt', 'llama', 'general']:
                                normalized_path = '/'.join(path_parts[1:])
                            else:
                                normalized_path = file_key

                            qna_prefix = 'qna/'
                            action_prefix = 'action/'

                            if normalized_path.lower().startswith(qna_prefix):
                                intent_filtered_docs.append((file_key, content, score))
                            elif normalized_path.lower().startswith(action_prefix):
                                intent_filtered_docs.append((file_key, content, score))
                                action_docs_included += 1
                            else:
                                excluded_count += 1
                        else:
                            qna_prefix = 'qna/'
                            action_prefix = 'action/'

                            if file_key.lower().startswith(qna_prefix):
                                intent_filtered_docs.append((file_key, content, score))
                            elif file_key.lower().startswith(action_prefix):
                                intent_filtered_docs.append((file_key, content, score))
                                action_docs_included += 1
                            else:
                                excluded_count += 1

                    filtered_docs = intent_filtered_docs
                    print(f"🎯 [QNA] 필터링: {len(filtered_docs)}개 유지 (action 문서 {action_docs_included}개 동등 가중치로 포함), {excluded_count}개 제외")

                    if app_context:
                        filtered_docs = self._filter_docs_by_app_context(filtered_docs, app_context)

                    similar_docs = filtered_docs
                else:
                    similar_docs = filtered_docs

                if (question_type and question_type.upper() == 'QNA') or (question_type and question_type.upper() == 'PLAYBOOK'):
                    logger.info(f"QNA 모드: {len(filtered_docs)}개 문서 사용")
                else:
                    if len(filtered_docs) < min_documents and similar_docs:
                        logger.warning(f"임계값 적용 후 문서가 {len(filtered_docs)}개뿐. 최소 {min_documents}개 보장")
                        filtered_docs = similar_docs[:min_documents]

                similar_docs = filtered_docs

            if _is_bge:
                logger.debug(f"🔍 [BGE-M3] 의도 기반 가중치 적용 건너뜀")
            elif question_type or (intent_info and intent_info.get('primary_intent')):
                intent = question_type.upper() if question_type else intent_info['primary_intent'].upper()

                print(f"🎯 [{intent}] 의도 집중 가중치 적용 중...")

                intent_to_doc_types = {
                    'QNA': ['faq', 'reference', 'tutorial', 'troubleshooting', 'configuration', 'action'],  
                    'ACTION': ['troubleshooting', 'configuration', 'action'],
                    'PLAN': ['installation', 'tutorial'],
                    'PLAYBOOK': ['playbook', 'workflow', 'procedure']
                }

                target_doc_types = intent_to_doc_types.get(intent, [])

                if target_doc_types:
                    weighted_docs = []

                    print(f"🔍 가중치 적용 전 원본 점수들:")
                    for file_key, content, original_score in similar_docs:
                        print(f"   📄 {file_key}: {original_score:.4f}")

                    for file_key, content, original_score in similar_docs:
                        # CAPEC & MITRE 정확한 숫자 매칭 부스트 적용
                        import re
                        boost_applied = False

                        # 쿼리에서 CAPEC + 숫자 패턴 검색
                        capec_patterns = [
                            r'카펙\s*(\d+)',
                            r'케이펙\s*(\d+)',
                            r'캐펙\s*(\d+)',
                            r'CAPEC\s*[-]?(\d+)',
                            r'capec\s*[-]?(\d+)'
                        ]

                        # 쿼리에서 MITRE 그룹/테크닉 패턴 검색
                        mitre_group_patterns = [
                            r'미트리\s*그룹\s*G?(\d+)',
                            r'미트리\s*G(\d+)',
                            r'mitre\s*group\s*G?(\d+)',
                            r'mitre\s*G(\d+)',
                            r'MITRE\s*GROUP\s*G?(\d+)',
                            r'MITRE\s*G(\d+)',
                            r'G(\d+)'
                        ]

                        mitre_technique_patterns = [
                            r'미트리\s*기술\s*T?(\d+(?:\.\d+)?)',
                            r'미트리\s*테크닉\s*T?(\d+(?:\.\d+)?)',
                            r'미트리\s*T(\d+(?:\.\d+)?)',
                            r'mitre\s*technique\s*T?(\d+(?:\.\d+)?)',
                            r'mitre\s*T(\d+(?:\.\d+)?)',
                            r'MITRE\s*TECHNIQUE\s*T?(\d+(?:\.\d+)?)',
                            r'MITRE\s*T(\d+(?:\.\d+)?)',
                            r'T(\d+(?:\.\d+)?)'
                        ]

                        # CAPEC 매칭 체크
                        capec_numbers = []
                        for pattern in capec_patterns:
                            matches = re.findall(pattern, search_query, re.IGNORECASE)
                            capec_numbers.extend(matches)

                        if capec_numbers:
                            query_number = capec_numbers[0]
                            file_capec_match = re.search(r'capec[-]?(\d+)', file_key.lower())
                            if file_capec_match:
                                file_number = file_capec_match.group(1)
                                if query_number == file_number:
                                    original_score = min(0.95, original_score * 2.5)
                                    boost_applied = True
                                    print(f"  🎯 [CAPEC-{query_number}] 정확 매칭 부스트: {file_key} → {original_score:.3f}")

                        # MITRE 그룹 매칭 체크
                        mitre_group_numbers = []
                        for pattern in mitre_group_patterns:
                            matches = re.findall(pattern, search_query, re.IGNORECASE)
                            mitre_group_numbers.extend(matches)

                        if mitre_group_numbers:
                            query_number = mitre_group_numbers[0]
                            file_group_match = re.search(r'mitre-group-g(\d+)', file_key.lower())
                            if file_group_match:
                                file_number = file_group_match.group(1)
                                if query_number.zfill(4) == file_number.zfill(4):  # 앞자리 0 처리
                                    original_score = min(0.95, original_score * 2.5)
                                    boost_applied = True
                                    print(f"  🎯 [MITRE-G{query_number}] 정확 매칭 부스트: {file_key} → {original_score:.3f}")

                        # MITRE 테크닉 매칭 체크
                        mitre_technique_numbers = []
                        for pattern in mitre_technique_patterns:
                            matches = re.findall(pattern, search_query, re.IGNORECASE)
                            mitre_technique_numbers.extend(matches)

                        if mitre_technique_numbers:
                            query_number = mitre_technique_numbers[0]
                            file_technique_match = re.search(r'mitre-technique-t(\d+(?:\.\d+)?)', file_key.lower())
                            if file_technique_match:
                                file_number = file_technique_match.group(1)
                                if query_number == file_number:
                                    original_score = min(0.95, original_score * 2.5)
                                    boost_applied = True
                                    print(f"  🎯 [MITRE-T{query_number}] 정확 매칭 부스트: {file_key} → {original_score:.3f}")

                        file_lower = file_key.lower()
                        doc_type = 'other'
                        if file_key.startswith('qna/'):
                            doc_type = 'faq'
                        elif file_key.startswith('action/'):
                            doc_type = 'action'

                        if any(keyword in file_lower for keyword in ['install', '설치']):
                            doc_type = 'installation'
                        elif any(keyword in file_lower for keyword in ['config', '설정', 'setup']):
                            doc_type = 'configuration'
                        elif any(keyword in file_lower for keyword in ['trouble', '문제', 'error', 'fix', '해결']):
                            doc_type = 'troubleshooting'
                        elif any(keyword in file_lower for keyword in ['reference', 'ref', '참조', 'api']):
                            doc_type = 'reference'
                        elif any(keyword in file_lower for keyword in ['tutorial', '튜토리얼', 'lesson']):
                            doc_type = 'tutorial'
                        elif any(keyword in file_lower for keyword in ['faq', '자주', 'q&a']):
                            doc_type = 'faq'
                        elif any(keyword in file_lower for keyword in ['playbook', 'workflow', 'procedure', 'runbook']):
                            doc_type = 'playbook'
                        elif file_key.lower().startswith('playbook/'):
                            doc_type = 'playbook'

                        if content.strip().startswith('---') and doc_type == 'other':
                            try:
                                import yaml
                                yaml_data = yaml.safe_load(content)
                                if isinstance(yaml_data, dict):
                                    question = yaml_data.get('question', '').lower()
                                    answer = yaml_data.get('answer', '').lower()

                                    if '?' in question or '어떻게' in question or 'how' in question:
                                        doc_type = 'faq'
                                    elif any(kw in (question + answer) for kw in ['설치', 'install']):
                                        doc_type = 'installation'
                                    elif any(kw in (question + answer) for kw in ['설정', 'config']):
                                        doc_type = 'configuration'
                                    elif any(kw in (question + answer) for kw in ['문제', 'error', '오류']):
                                        doc_type = 'troubleshooting'
                                    elif any(kw in (question + answer) for kw in ['api', 'reference', '참조']):
                                        doc_type = 'reference'
                                    elif any(kw in (question + answer) for kw in ['tutorial', '단계', 'step']):
                                        doc_type = 'tutorial'
                            except:
                                pass

                        if doc_type in target_doc_types:
                            if intent == 'QNA' and doc_type == 'action':
                                weight_multiplier = 1.2 
                                print(f"  🎯 [QNA-ACTION] {file_key}: {doc_type} x{weight_multiplier} = {original_score:.3f} → {min(0.99, original_score * weight_multiplier):.3f}")
                            else:
                                weight_multiplier = 1.2 
                                print(f"  🔥 {file_key}: {doc_type} x{weight_multiplier} = {original_score:.3f} → {min(0.99, original_score * weight_multiplier):.3f}")
                            adjusted_score = min(0.99, original_score * weight_multiplier)
                        else:
                            weight_multiplier = 0.5 
                            adjusted_score = original_score * weight_multiplier
                            print(f"  ❄️ {file_key}: {doc_type} x{weight_multiplier} = {original_score:.3f} → {adjusted_score:.3f}")

                        weighted_docs.append((file_key, content, adjusted_score))

                    weighted_docs.sort(key=lambda x: x[2], reverse=True)
                    similar_docs = weighted_docs

                    print(f"✅ [{intent}] 집중 가중치 적용 완료")

            similar_docs = similar_docs[:top_k]

            if not similar_docs:
                logger.info(f"검색된 문서 없음")
                return "", [], extracted_concepts, concepts

            search_time = time.time() - start_time
            logger.info(f"검색 완료: {len(similar_docs)}개 문서 (시간: {search_time:.2f}초) (임계값: {score_threshold})")

            _t3 = time.time()
            context_max_tokens = 20000
            context = self.extract_answers_from_documents(
                documents=similar_docs,
                max_tokens=context_max_tokens,
                system_prompt_tokens=2000,
                query_tokens=500
            )
            _step_times['컨텍스트구성'] = time.time() - _t3

            if intent_info:
                intent_desc = f"\n\n[의도 분석: {intent_info['primary_intent'].upper()} - {intent_info['distribution']}]"
                context = intent_desc + context

            if is_incremental_mode:
                file_paths = [file_key for file_key, _, _ in backup_docs]
                print(f"증분 모드: {len(file_paths)}개 참조 문서 표시")
            else:
                file_paths = [file_key for file_key, _, _ in similar_docs]
            logger.info(f"참고 문서: {file_paths}")

            total_time = time.time() - start_time
            logger.info(f"전체 문서 검색 완료 (총 시간: {total_time:.2f}초)")

            if is_incremental_mode:
                detailed_references = []
                temporary_count = 0
                existing_count = 0

                for i, file_key in enumerate(file_paths):
                    is_temporary = file_key in temp_prompt if temp_prompt else False

                    score = 0.0
                    for backup_file, _, backup_score in backup_docs:
                        if backup_file == file_key:
                            score = backup_score
                            break

                    detailed_references.append({
                        'rank': i + 1,
                        'file': file_key,
                        'score': round(score, 4),
                        'source': 'temporary' if is_temporary else 'existing',
                        'is_new': is_temporary
                    })

                    if is_temporary:
                        temporary_count += 1
                    else:
                        existing_count += 1

                reference_summary = {
                    'detailed_references': detailed_references,
                    'total_documents': len(detailed_references),
                    'temporary_count': temporary_count,
                    'existing_count': existing_count,
                    'is_incremental_mode': True
                }
            else:
                reference_summary = None

            # 단계별 소요시간 합산 출력
            total_time = time.time() - start_time
            _step_times['후처리'] = total_time - sum(_step_times.values())
            parts = ' + '.join([f"{k} {v:.2f}초" for k, v in _step_times.items() if v >= 0.01])
            print(f"⏱️ [문서검색 소요시간] {parts} = 총 {total_time:.2f}초")

            return context, file_paths, extracted_concepts, concepts, intent_info, reference_summary
        except Exception as e:
            logger.error(f"관련 문서 검색 중 오류: {str(e)}")
            return "", [], [], [], None

    def _expand_with_stems(self, query: str) -> str:
        if not query:
            return query

        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)

        if not words:
            return query

        stem_important_words = {
            'normalize', 'attack', 'analyze', 'process', 'configure',
            'monitor', 'access', 'authenticate', 'authorize', 'validate',
            'parse', 'format', 'convert', 'transform', 'extract',
            'verify', 'encrypt', 'decrypt', 'compress', 'decompress'
        }

        expanded_terms = []

        for word in words:
            word_lower = word.lower()

            if word_lower in stem_important_words:
                variations = self._get_word_variations(word_lower)
                expanded_terms.extend(variations)
                print(f"🔍 [쿼리 확장] '{word}' → {variations}")
            else:
                expanded_terms.append(word)

        unique_terms = list(dict.fromkeys(expanded_terms))

        if len(unique_terms) > 10:
            unique_terms = unique_terms[:10]
            print(f"🔍 [쿼리 확장] 용어 수 제한: {len(unique_terms)}개")

        expanded_query = ' '.join(unique_terms)

        if expanded_query != query:
            print(f"🔍 [쿼리 확장] '{query}' → '{expanded_query}'")

        return expanded_query

    def _get_word_variations(self, word: str) -> List[str]:
        variations = [word]

        patterns = [
            ('', 'ing'),
            ('', 'ed'),
            ('', 's'),
            ('e', 'ing'),
            ('e', 'ed'),
            ('', 'er'),
            ('', 'ion'),
            ('e', 'ation'),
            ('', 'tion'),

            ('', 'ly'),
            ('', 'ness'),
            ('', 'able'),
            ('y', 'ies'),
        ]

        for remove_suffix, add_suffix in patterns:
            if not remove_suffix or word.endswith(remove_suffix):
                stem = word[:-len(remove_suffix)] if remove_suffix else word
                if len(stem) >= 3:
                    new_word = stem + add_suffix
                    if new_word != word and len(new_word) <= 15:
                        variations.append(new_word)

        for remove_suffix, add_suffix in patterns:
            if word.endswith(add_suffix):
                stem = word[:-len(add_suffix)]
                if len(stem) >= 3:
                    original_word = stem + (remove_suffix if remove_suffix else '')
                    if original_word != word and len(original_word) >= 3:
                        variations.append(original_word)

        unique_variations = list(dict.fromkeys(variations))
        unique_variations.sort(key=len)

        return unique_variations[:5]

    def extract_answers_from_documents(self, documents: List[Tuple[str, str, float]],
                               max_tokens: int = 20000,
                               system_prompt_tokens: int = 1500,
                               query_tokens: int = 500) -> str:
        logger.info(f"🔍 DEBUG: extract_answers_from_documents called with {len(documents)} documents")
        import yaml
        context_parts = []
        total_tokens = 0

        reserved_tokens = system_prompt_tokens + query_tokens
        available_tokens = max_tokens - reserved_tokens

        logger.info(f"사용 가능한 토큰: {available_tokens} (최대: {max_tokens}, 예약: {reserved_tokens})")

        from aibot_embedding import get_encoding
        tokenizer = get_encoding("cl100k_base")

        for doc_index, (file_key, content, score) in enumerate(documents):
            try:
                if hasattr(self, 'use_db') and self.use_db:
                    subscription_file_contents = self.rag_system._get_file_contents(sub_id) if sub_id else {}
                    if file_key in subscription_file_contents:
                        actual_content = subscription_file_contents[file_key]
                        logger.debug(f"DB 모드: {file_key}의 정제된 내용 사용")
                    else:
                        logger.warning(f"DB 모드: {file_key}의 내용을 찾을 수 없습니다.")
                        continue
                else:
                    actual_content = content
                    logger.debug(f"파일 모드: {file_key}의 전달받은 내용 사용")

                data = yaml.safe_load(actual_content)

                if not isinstance(data, dict):
                    if hasattr(self, 'use_db') and self.use_db:
                        logger.warning(f"DB 모드에서 {file_key}가 dict가 아님: {type(data)}")

                    if data is None:
                        logger.info(f"문서 {file_key}는 빈 YAML 문서입니다.")
                        continue
                    elif isinstance(data, list):
                        logger.warning(f"문서 {file_key}는 리스트 형식입니다. 첫 번째 항목 확인 중...")
                        if data and isinstance(data[0], dict):
                            data = data[0]
                            logger.info(f"문서 {file_key}의 첫 번째 dict 항목 사용")
                        else:
                            logger.warning(f"문서 {file_key}의 리스트에 dict가 없습니다.")
                            continue
                    elif isinstance(data, (str, int, float)):
                        logger.warning(f"문서 {file_key}는 단순 값({type(data).__name__})입니다.")
                        doc_intro = f"\n\n📄 문서 #{doc_index + 1}: {file_key} (유사도: {score:.3f})"
                        doc_body = f"CONTENT: {str(data)}"
                        full_doc = doc_intro + "\n" + doc_body

                        doc_tokens = len(tokenizer.encode(full_doc))
                        if total_tokens + doc_tokens <= available_tokens:
                            context_parts.append(full_doc)
                            total_tokens += doc_tokens
                            logger.info(f"✅ 단순 값 문서 {file_key} 추가 완료")
                        continue
                    else:
                        logger.warning(f"문서 {file_key}는 처리할 수 없는 형식입니다: {type(data)}")
                        continue

                question = str(data.get("question", "")).strip()
                cot = str(data.get("cot", "")).strip()
                answer_raw = data.get("answer", "")
                answer = json.dumps(answer_raw, indent=2) if isinstance(answer_raw,
                (dict, list)) else str(answer_raw).strip()
                spec_raw = data.get("spec", "")
                spec = json.dumps(spec_raw, indent=2) if isinstance(spec_raw,
                (dict, list)) else str(spec_raw).strip()
                require_raw = data.get("require", "")
                require = json.dumps(require_raw, indent=2) if isinstance(require_raw,
                (dict, list)) else str(require_raw).strip()

                doc_intro = f"\n\n📄 문서 #{doc_index + 1}: {file_key} (유사도: {score:.3f})"

                content_parts = []
                if question:
                    content_parts.append(f"QUESTION: {question}")
                if cot:
                    content_parts.append(f"COT: {cot}")
                if spec:
                    content_parts.append(f"SPEC: {spec}")
                if require:
                    content_parts.append(f"REQUIRE: {require}")
                if answer:
                    content_parts.append(f"ANSWER: {answer}")

                doc_body = "\n".join(content_parts)
                full_doc = doc_intro + "\n" + doc_body

                doc_tokens = len(tokenizer.encode(full_doc))
                if total_tokens + doc_tokens <= available_tokens:
                    context_parts.append(full_doc)
                    total_tokens += doc_tokens
                    logger.info(f"✅ 문서 {file_key} 추가 완료 (토큰 수: {doc_tokens})")
                else:
                    logger.warning(f"⛔ 문서 {file_key}는 토큰 초과로 생략됨 (필요: {doc_tokens}, 남은: {available_tokens - total_tokens})")
                    break

            except yaml.YAMLError as e:
                logger.error(f"문서 {file_key} YAML 파싱 오류: {e}")
                continue
            except Exception as e:
                logger.error(f"문서 {file_key} 처리 중 오류 발생: {e}")
                continue

        logger.info(f"🔚 총 {len(context_parts)}개 문서 포함 완료 (총 토큰: {total_tokens}/{available_tokens})")
        return "".join(context_parts)

    def get_last_sources(self) -> List[str]:
        return self.last_sources


    def get_search_status(self):
        try:
            if not hasattr(self, 'config_manager'):
                self.config_manager = ConfigManager()

            search_config = self.config_manager.get_search_config()

        except Exception as e:
            print(f"⚠️ 검색 설정 로드 실패: {e}")
            search_config = {
                'search_strategy': 'enhanced',
                'max_parallel_searches': 4,
                'cache_enabled': True,
                'cache_expiry_seconds': 3600,
                'search_timeout_seconds': 10
            }

        return {
            'rag_system_available': hasattr(self, 'rag_system') and self.rag_system is not None,
            'rag_enabled': hasattr(self, 'rag_system') and self.rag_system is not None,
            'knowledge_graph_enabled': hasattr(self, 'use_kg') and self.use_kg,
            'knowledge_graph': hasattr(self, 'use_kg') and self.use_kg,
            'search_config': search_config,
            'total_documents': len(self.rag_system.embeddings) if (hasattr(self, 'rag_system') and
                                                                self.rag_system and
                                                                hasattr(self.rag_system, 'embeddings')) else 0,
            'cache_size': 0
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": self.__class__.__name__,
            "rag_enabled": bool(self.rag_system),
            "kg_enabled": self.use_kg,
            "intent_analyzer_enabled": bool(self.intent_analyzer),
            "available": getattr(self, 'available', False)
        }

    def get_response_mode_by_intent(self, question_type: str = None) -> str:
        if not question_type:
            return self.config_manager.get_option_config().get('response', 'streaming')

        intent_upper = question_type.upper()

        if intent_upper == 'QNA':
            return 'streaming'
        elif intent_upper in ['ACTION', 'PLAN']:
            return 'complete'
        else:
            return self.config_manager.get_option_config().get('response', 'streaming')


    async def process_question(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> Dict:
        """streaming 응답 wrap — generate_stream 을 answer_stream 으로 노출.
        모든 핸들러 공통 boilerplate. 커스텀이 필요한 핸들러만 override.
        """
        try:
            self.last_sources = []
            return {
                "answer_stream": self.generate_stream(request, user_id, channel_id, connection, thread_ts),
                "context_sources": []
            }
        except Exception as e:
            logger.error(f"{self.__class__.__name__} process_question [{type(e).__name__}]: {e}")
            user_msg = classify_error(e)

            async def error_stream():
                yield user_msg

            return {"answer_stream": error_stream(), "context_sources": []}

    # ──────────────────────────────────────────────────────────────────────
    # Agent (/agent/chat/completions2) 균일 진입점
    #
    # 목적: OpenAI 호환 messages → 응답. 모델별 chat-template 조립을 핸들러가 캡슐화해
    # 새 모델 추가 시 completions2 엔드포인트 코드를 건드리지 않게 한다.
    #   - 로컬 모델(GGUF/llama-server): build_agent_prompt + _agent_generate 만 구현하면
    #     아래 기본 agent_complete 가 그대로 동작.
    #   - OpenAI류(messages passthrough): agent_complete 자체를 override.
    # ──────────────────────────────────────────────────────────────────────
    agent_model_name: str = ""          # 응답 model 필드용 (비면 registry_key 사용)

    def build_agent_prompt(self, messages: List[Dict], options: Dict = None) -> str:
        """OpenAI messages → 모델별 chat-template 프롬프트 문자열. 로컬 핸들러가 override.
        options 로 locale 등 부가 지시 전달 (예: options['locale'] → 응답 언어 강제)."""
        raise NotImplementedError(
            f"{type(self).__name__} 는 build_agent_prompt 를 구현하거나 "
            f"agent_complete 를 override 해야 합니다."
        )

    def _agent_generate(self, prompt: str) -> str:
        """build_agent_prompt 결과를 로컬 모델에 투입해 raw 응답 반환.
        local_model.generate_agent 시그니처가 핸들러마다 달라 여기서 흡수한다."""
        raise NotImplementedError(
            f"{type(self).__name__} 는 _agent_generate 를 구현하거나 "
            f"agent_complete 를 override 해야 합니다."
        )

    async def agent_complete(self, messages: List[Dict], options: Dict = None) -> Dict[str, Any]:
        """completions2 균일 진입점. 반환: {"content": str, "model": str, "usage": dict}.
        로컬 모델 기본 구현 (prompt 빌드 → 스레드풀에서 생성 → 단어수 usage).
        length/reasoning_effort 등 options 는 로컬 모델에선 무시 (OpenAI류만 override 에서 사용)."""
        if not getattr(self, "available", False) or getattr(self, "local_model", None) is None:
            raise RuntimeError(f"{self.registry_key} 모델을 사용할 수 없습니다. (핸들러 미로드)")
        prompt = self.build_agent_prompt(messages, options or {})
        # ── PII 치환: 로컬 모델 투입 직전, 조립이 끝난 프롬프트 전체에 (S-6) ──
        # 여기서 감싸므로 로컬 핸들러(gemma·gpt-oss·llama·qwen)는 _agent_generate 만 구현하면 된다.
        _pii_map = {}
        if self._pii_ready():
            prompt, _pii_map = self.pii.mask_reversible(prompt)
        print(f"🔍 [completions2/{self.registry_key}] prompt_len={len(prompt)}")
        text = await asyncio.to_thread(self._agent_generate, prompt)
        text = text or ""
        # 복원은 비스트리밍이라 끝에서 1회. restore=False 면 토큰이 남아 마스킹을 눈으로 검증할 수 있다.
        text = self._pii_unmask(text, _pii_map)
        print(f"🔍 [completions2/{self.registry_key}] response_text len={len(text)}")
        pt, ct = len(prompt.split()), len(text.split())
        return {
            "content": text,
            "model": self.agent_model_name or self.registry_key,
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }

    def _get_combined_query_from_history(self, history: List, current_query: str) -> str:
        if not history or not current_query:
            return current_query

        if self._is_complete_query(current_query):
            return current_query

        original_question = self._find_original_question_from_history(history)

        if original_question and original_question != current_query:
            return f"{original_question} {current_query}"

        return current_query

    def _is_complete_query(self, query: str) -> bool:
        query_lower = query.lower().strip()

        if len(query_lower) < 8:
            return False

        action_keywords = [
            "조회", "검색", "찾", "분석", "확인", "보여", "알려", "생성", "만들",
            "query", "search", "show", "find", "analyze", "create", "generate",
            "어떤", "무엇", "언제", "어디", "왜", "어떻게"
        ]

        has_action = any(keyword in query_lower for keyword in action_keywords)

        if has_action:
            # 행위 동사만 있는 짧은 질문은 불완전 ("찾아줘", "보여줘", "다시 만들어줘")
            bare_verb_patterns = [
                r'^(그럼\s*|좀\s*|다시\s*)?(찾아|보여|알려|만들어|생성해|분석해|확인해|조회해|검색해)\s*(줘|주세요|줄래|봐|봐줘)?\s*[.!?]?\s*$',
            ]
            for pattern in bare_verb_patterns:
                if re.search(pattern, query_lower):
                    return False
            return True

        if "?" in query or query_lower.endswith(("까", "나", "요", "니")):
            return True

        return False


    def _find_original_question_from_history(self, history: List) -> str:
        for msg in reversed(history):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "").strip()
                if self._is_complete_query(user_msg):
                    return user_msg

        return None

    def _is_clarify_followup(self, history: List, current_query: str) -> bool:
        if not history or len(history) < 2:
            return False

        last_assistant_msg = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break

        if not last_assistant_msg:
            return False

        is_clarify_response = (
            '"action": "clarify"' in last_assistant_msg or
            '"action":"clarify"' in last_assistant_msg or
            any(keyword in last_assistant_msg.lower() for keyword in [
                "어떤 시간", "어떤 ip", "어떤 테이블", "무엇을", "어떤 값을",
                "필요한 매개변수", "추가 정보", "구체적으로", "알려주세요",
                "조회 기간", "시간 범위", "기간을 지정", "날짜를 입력",
                "specify", "provide", "please enter", "time range",
                "어떤 기간", "언제부터", "언제까지", "몇 시부터", "몇 일",
                "시작 시간", "종료 시간", "from.*to", "기간.*알려"
            ]) or
            ("?" in last_assistant_msg and any(keyword in last_assistant_msg.lower() for keyword in [
                "언제", "어떤", "무엇", "어디", "몇", "얼마", "어느"
            ]))
        )

        if not is_clarify_response:
            return False

        return self._is_parameter_response(current_query)

    def _is_parameter_response(self, query: str) -> bool:
        if not query or len(query) > 200:
            return False

        query_lower = query.lower().strip()

        import re
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        if re.search(ip_pattern, query):
            return True

        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{8}',
            r'최근.*\d+.*일',
            r'지난.*\d+.*일',
            r'\d+월\s*\d+일',
            r'어제.*오늘',
            r'오늘.*어제',
            r'이번.*주',
            r'지난.*주',
            r'작년.*올해',
        ]
        if any(re.search(pattern, query) for pattern in date_patterns):
            return True

        time_keywords = [
            "어제", "오늘", "내일", "모레", "그제",
            "yesterday", "today", "tomorrow",
            "부터", "까지", "동안", "from", "to", "until",
            "이번", "지난", "다음", "this", "last", "next"
        ]
        if any(keyword in query_lower for keyword in time_keywords):
            return True

        if query_lower in ['yes', 'no', '네', '아니오', '예']:
            return True

        if re.match(r'^\d+$', query_lower):
            return True

        words = query.split()
        if len(words) <= 5 and not any(word in query_lower for word in ['조회', 'search', 'query', 'show', 'list']):
            return True

        return False

    def _reuse_previous_context(self, request: Dict) -> tuple:
        if not hasattr(self, 'last_sources') or not self.last_sources:
            print("🔄 이전 참조 문서가 없어서 새로 검색합니다.")
            return None, None, None, None, None

        if isinstance(request, dict):
            sub_id = request.get("sub_id", None)
        else:
            sub_id = None

        print(f"🔄 이전 참조 문서 재사용: {len(self.last_sources)}개")

        if not self.rag_system:
            return "", self.last_sources, [], [], None

        try:
            context_parts = []
            for file_path in self.last_sources:
                try:
                    content = self.rag_system.read_file_content(file_path, sub_id)
                    if content:
                        context_parts.append(f"=== {file_path} ===\n{content}\n")
                except Exception as e:
                    print(f"⚠️ 문서 {file_path} 읽기 실패: {e}")
                    continue

            context = "\n".join(context_parts)

            extracted_concepts = []
            concepts = []
            intent_info = None

            return context, self.last_sources, extracted_concepts, concepts, intent_info

        except Exception as e:
            print(f"❌ 이전 컨텍스트 재사용 실패: {e}")
            return None, None, None, None, None

    def _load_domain_keywords(self) -> list:
        """카트리지 정의 도메인 키워드 로드 ([prompts] domain_keywords 경로, 줄 단위·# 주석).
        미설정/미존재 시 빈 리스트 — Stage 1 없이 casual 패턴만으로 판별한다."""
        if not hasattr(self, '_domain_keywords_cache'):
            keywords = []
            try:
                path = self.config_manager.get_prompt_config().get('domain_keywords')
                if path and os.path.exists(path):
                    with open(path, encoding='utf-8') as f:
                        keywords = [ln.strip().lower() for ln in f
                                    if ln.strip() and not ln.startswith('#')]
            except Exception:
                pass
            self._domain_keywords_cache = keywords
        return self._domain_keywords_cache

    def _is_casual_query(self, query: str) -> bool:
        """일상 질문 판별: RAG 검색을 건너뛸 비도메인 질문인지 확인"""
        query_lower = query.lower().strip()
        if not query_lower or len(query_lower) > 300:
            return False

        # Stage 1: 도메인 키워드가 있으면 기술 질문 (키워드는 카트리지가 정의 — [prompts] domain_keywords)
        if any(kw in query_lower for kw in self._load_domain_keywords()):
            return False
        if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', query):
            return False

        # Stage 2: 일상 패턴
        casual_patterns = [
            r'^안녕', r'^반갑', r'^하이$', r'^헬로', r'^hi$', r'^hello',
            r'날씨', r'weather', r'맛집', r'음식', r'식당',
            r'영화', r'드라마', r'음악', r'노래',
            r'축구', r'야구', r'스포츠', r'주가', r'주식', r'비트코인',
            r'여행', r'관광', r'몇\s*살', r'재미있', r'심심',
            r'고마워', r'감사합니다', r'자기\s*소개',
            r'너\s*(는|가)\s*(누구|뭐)', r'이름이\s*뭐',
        ]
        if any(re.search(p, query_lower) for p in casual_patterns):
            return True

        # Stage 3: 극단적으로 짧은 질문 (도메인 키워드 없이)
        if len(query_lower) <= 4:
            return True

        return False

    def _is_modification_request(self, history: list, query: str) -> bool:
        """수정 요청 판별: 이전 답변을 수정/변경하는 요청인지 확인"""
        if not history or len(history) < 2:
            return False

        # assistant 응답이 있어야 수정 대상이 존재
        if not any(msg.get("role") == "assistant" for msg in history):
            return False

        query_lower = query.lower().strip()

        modification_keywords = [
            '바꿔', '바꾸', '변경', '수정', '고쳐', '고치',
            '다시 만들', '다시만들', '다시 해', '다시해',
            '포함조건', '포함 조건', '제외조건', '제외 조건',
            '조건을', '조건으로', '필터를', '필터로',
            '시간을', '기간을', '날짜를', '테이블을',
            '그걸로', '그거를', '그걸', '그것을', '위에서', '아까',
            '추가해', '추가로', '빼줘', '빼고', '제거해', '제외해',
            '대신', '말고',
            '으로 해', '으로해', '로 해줘', '로해줘',
        ]

        if any(kw in query_lower for kw in modification_keywords):
            # 가드: 긴 질문 + 도메인 키워드 → 새 질문일 가능성 높음
            if len(query_lower) > 15:
                new_topic_terms = ['로그', 'log', '쿼리', 'query', '공격', 'attack', '보안']
                if any(t in query_lower for t in new_topic_terms):
                    return False
            return True

        return False

    def find_docs(self, query: str, top_k: int = 10, max_token_count: int = 8000,
                  faiss_weight: float = 0.7, bm25_weight: float = 0.3,
                  score_threshold: float = 0.35) -> Tuple[str, List[str], List[str], List[str], Dict]:

        print(f"🔍 하이브리드 문서 검색: '{query}'")
        if not self.rag_system:
            print("❌ RAG 시스템이 없어서 빈 결과 반환")
            return "", [], [], [], None

        try:
            similar_docs = self.rag_system.find_docs_hybrid(
                query=query,
                top_k=top_k,
                max_token_count=max_token_count,
                faiss_weight=faiss_weight,
                bm25_weight=bm25_weight
            )

            if not similar_docs:
                print("❌ 검색 결과가 없습니다")
                return "", [], [], [], None

            filtered_docs = []
            for file_key, content, score in similar_docs:
                if score >= score_threshold:
                    filtered_docs.append((file_key, content, score))
                    print(f"✅ 문서 선택: {file_key} (점수: {score:.3f})")
                else:
                    print(f"❌ 문서 제외: {file_key} (점수: {score:.3f} < 임계값: {score_threshold})")

            if not filtered_docs:
                print("❌ 임계값을 만족하는 문서가 없습니다")
                return "", [], [], [], None

            context_parts = []
            source_files = []

            for i, (file_key, content, score) in enumerate(filtered_docs):
                if content.strip():
                    display_name = file_key.replace('.yaml', '').replace('.json', '').replace('.md', '')

                    context_parts.append(f"## 참조문서 {i+1}: {display_name} (점수: {score:.3f})")
                    context_parts.append(content.strip())
                    context_parts.append("")

                    source_files.append(file_key)

            final_context = "\n".join(context_parts).strip()

            self.last_sources = source_files
            print(f"✅ 하이브리드 검색 완료: {len(source_files)}개 문서, {len(final_context)} 문자")

            intent_info = {
                'primary_intent': 'hybrid_search',
                'distribution': {'hybrid': 1.0},
                'is_multi': False,
                'reasoning': 'FAISS + BM25 하이브리드 검색',
                'analysis_method': 'hybrid'
            }

            return final_context, source_files, [], [], intent_info

        except Exception as e:
            print(f"❌ 하이브리드 검색 오류: {e}")
            return "", [], [], [], None

    def _apply_response_format(self, system_prompt: str, response_format: bool = False) -> str:

        if response_format is True:
            format_instruction = """

                FORMAT ENHANCEMENT RULES:
                - Use emoji whenever possible to make responses more engaging
                - Use markdown formatting for better readability (bold, italic, tables, code blocks)
                - Use rich text formatting including **bold**, *italic*, `code`, and tables
                - When explaining query steps or commands sequentially, use numbered lists with proper indentation
                - Make responses visually appealing and easy to read
            """

        else:
            format_instruction = """

                PLAIN TEXT RESTRICTIONS:
                STRICT PLAIN TEXT MODE - ABSOLUTELY NO FORMATTING:
                - NEVER use **bold**, *italic*, `code`, ``` blocks, or any markdown
                - NEVER use tables with | symbols or dashes for headers
                - NEVER use # headers or bullet points with special characters
                - ONLY use simple text paragraphs separated by blank lines
                - For lists: use simple numbered lines (1. 2. 3.) or hyphens at start
                - For emphasis: use CAPITAL LETTERS or repeat words instead of formatting
                - For code examples: use plain text without any special formatting
                - Treat this as writing for a plain text email or simple notepad
            """


        return system_prompt + format_instruction

    def _remove_markdown_formatting(self, text: str) -> str:
        import re
        
        try:
            query_blocks = []
            def save_query_block(match):
                query_blocks.append(match.group(0))
                return f"__QUERY_BLOCK_{len(query_blocks)-1}__"

            text = re.sub(r'```query[\s\S]*?```', save_query_block, text)

            text = re.sub(r'```[\s\S]*?```', '', text)

            for i, block in enumerate(query_blocks):
                text = text.replace(f"__QUERY_BLOCK_{i}__", block)

            if not query_blocks:
                text = re.sub(r'`([^`]+)`', r'\1', text)

            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'__([^_]+)__', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            text = re.sub(r'_([^_]+)_', r'\1', text)

            text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

            emoji_pattern = re.compile("["
                                    "\U0001F600-\U0001F64F"
                                    "\U0001F300-\U0001F5FF"
                                    "\U0001F680-\U0001F6FF"
                                    "\U0001F1E0-\U0001F1FF"
                                    "\U00002702-\U000027B0"
                                    "\U000024C2-\U0001F251"
                                    "]+", flags=re.UNICODE)
            text = emoji_pattern.sub('', text)

            text = re.sub(r'^(\s*)[-*+]\s+', r'\1- ', text, flags=re.MULTILINE)

            lines = text.split('\n')
            clean_lines = []
            for line in lines:
                if '|' in line and '---' not in line:
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if cells:
                        clean_lines.append(' '.join(cells))
                elif '---' not in line:
                    clean_lines.append(line)

            text = '\n'.join(clean_lines)

            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = text.strip()

            return text

        except Exception:
            return text

    def _translate_for_bge_mode(self, query: str) -> Tuple[str, str]:
        """BGE-M3 모드 전용 번역 함수 - 기술패턴이 있으면 번역 건너뛰기
        Returns: (translated_query, sub_intent, has_tech_patterns)
        """
        try:
            print(f"🔍 [BGE-M3 번역] 시작: '{query[:50]}...'")

            # 순수 영어 감지 - 한글이 없고 영어만 있으면 번역 건너뛰기
            has_korean = re.search(r'[가-힣]', query)
            has_english = re.search(r'[a-zA-Z]', query)

            if not has_korean and has_english:
                print(f"🔍 [BGE-M3 번역] 순수 영어 감지, 번역 건너뛰기")
                return query, 'ENGLISH_ONLY'
            elif not has_korean:
                print(f"🔍 [BGE-M3 번역] 한글 없음, 번역 건너뛰기")
                return query, 'NO_KOREAN'

            # 기술 패턴 감지
            tech_patterns = [
                r'DETECT\|WAF\|',                           # WAF 로그
                r'HTTP/\d\.\d',                             # HTTP 프로토콜
                r'User-Agent:',                             # HTTP 헤더
                r'Accept:',                                 # HTTP 헤더
                r'Accept-Encoding:',                        # HTTP 헤더
                r'Accept-Language:',                        # HTTP 헤더
                r'Host:',                                   # HTTP 헤더
                r'GET\s+/|POST\s+/|PUT\s+/|DELETE\s+/',    # HTTP 메소드
                r'(?:src|dst|sip|dip|src_ip|dst_ip|dvc|outside|inside)[=:]\s*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', # 로그 원문 내 IP (key=value 형식)
                r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # 타임스탬프
                r'CVE-\d{4}-\d{4,7}',                       # CVE ID
                r'\|.*\|.*\|',                              # 파이프 구분자 3개 이상
                r'/[a-zA-Z0-9_\-\.]+\.(php|jsp|asp|do)',    # 파일 경로
                r'Mozilla/\d+\.\d+',                        # User-Agent의 Mozilla
                r'device_id=|devname=|log_id=',             # 디바이스 정보
            ]

            # 기술 패턴이 하나라도 있으면 번역하지 않음
            for pattern in tech_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    print(f"🔍 [BGE-M3 번역] 기술패턴 감지 ({pattern}), 번역 건너뛰기")
                    return query, 'TECH_PATTERN'

            # 한글+영어 혼합: 영어 토큰을 플레이스홀더로 보호 후 번역
            preserved = {}
            protected_query = query
            for match in re.finditer(r'[a-zA-Z][\w-]*', query):
                token = match.group()
                if len(token) <= 2:  # a, an, is 등 짧은 토큰 무시
                    continue
                key = f'__TK{len(preserved)}__'
                preserved[key] = token
                protected_query = protected_query.replace(token, key, 1)

            if preserved:
                print(f"🔍 [BGE-M3 번역] 영어 토큰 보호: {preserved}")

            translated_text = self._perform_simple_translation(protected_query, preserved)

            # 플레이스홀더를 원래 토큰으로 복원
            missing_tokens = []
            for key, token in preserved.items():
                if key in translated_text:
                    translated_text = translated_text.replace(key, token)
                else:
                    # LLM이 플레이스홀더를 누락한 경우
                    missing_tokens.append(token)

            # 누락된 토큰이 있으면 번역 결과 앞에 붙여줌
            if missing_tokens:
                prefix = ' '.join(missing_tokens)
                translated_text = f"{prefix} {translated_text}".strip()
                print(f"🔍 [BGE-M3 번역] 누락 토큰 복원: {missing_tokens}")

            print(f"🔍 [BGE-M3 번역] 완료: '{translated_text}'")

            # 번역 후 명령어 사용법 질문인지 분류
            sub_intent = 'NORMAL'
            if self.intent_analyzer and hasattr(self.intent_analyzer, 'analyze_command_ref_sync'):
                sub_intent = self.intent_analyzer.analyze_command_ref_sync(translated_text)

            return translated_text, sub_intent

        except Exception as e:
            logger.error(f"BGE-M3 번역 오류: {e}")
            return query, 'NORMAL'

    def _perform_simple_translation(self, text: str, preserved: dict = None) -> str:
        """순수한 한-영 번역만 수행하는 함수"""
        try:
            # BGE-M3 모드에서도 실제 LLM 번역 사용
            return self._perform_translation(text, 'NORMAL', preserved)
        except Exception as e:
            logger.error(f"BGE-M3 번역 오류: {e}")
            # 오류 시에만 사전 번역으로 폴백
            return self._enhanced_dictionary_translate(text)
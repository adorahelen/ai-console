"""LLM 핸들러 클래스 레지스트리.

새 모델 핸들러를 추가할 때 손대는 곳은 이 파일 하나로 충분하다.
- HANDLER_CLASSES 에 한 줄 추가 → aibot_llm_module 의 모든 라우팅이 자동 인식.
- LEGACY_ATTR_TO_KEY 는 기존 코드(qa_llm.py 등)에서 self.openai_handler 같은
  속성으로 직접 접근하는 부분을 그대로 살리기 위한 backward-compat 매핑.
"""
from typing import Dict, Tuple, Type

from handler_base import BaseModelHandler
from handler_openai import OpenAIHandler
from handler_llama import LlamaHandler
from handler_gpt_oss import GPTOSSHandler
from handler_qwen import QwenHandler
from handler_gemma import GemmaHandler
from handler_claude import ClaudeHandler


# 모델 이름(DEFAULT_MODEL / set_model() 인자) -> 핸들러 클래스
HANDLER_CLASSES: Dict[str, Type[BaseModelHandler]] = {
    "gpt":     OpenAIHandler,
    "gpt-oss": GPTOSSHandler,
    "qwen":    QwenHandler,
    "gemma":   GemmaHandler,
    "llama":   LlamaHandler,
    "claude":  ClaudeHandler,
}


def _validate_registry_keys() -> None:
    """HANDLER_CLASSES 의 dict 키와 각 핸들러 클래스의 registry_key class attr 일치 검증.
    새 핸들러 추가 시 두 곳을 sync 안 시키면 import 시점에 즉시 에러.
    """
    for key, cls in HANDLER_CLASSES.items():
        cls_key = getattr(cls, "registry_key", None)
        if not cls_key:
            raise ImportError(
                f"{cls.__name__} 는 'registry_key' class attr 를 선언해야 합니다 "
                f"(handler_registry.HANDLER_CLASSES 키 '{key}' 와 일치하도록)"
            )
        if cls_key != key:
            raise ImportError(
                f"registry mismatch: HANDLER_CLASSES['{key}'] = {cls.__name__}, "
                f"but {cls.__name__}.registry_key = '{cls_key}'. 한 곳을 고쳐서 sync 시키세요."
            )


_validate_registry_keys()

# DEFAULT_MODEL 이 HANDLER_CLASSES 에 없을 때 폴백
FALLBACK_MODEL = "llama"

# intent_analyzer active handler 우선순위 (원본 코드 L255-264 순서 보존 + claude 는 OpenAI 류이므로 gpt 뒤)
HANDLER_PRIORITY: Tuple[str, ...] = ("gpt", "claude", "gpt-oss", "qwen", "gemma", "llama")

# 기존 self.openai_handler / self.gptoss_handler 등 직접 접근 호환용
LEGACY_ATTR_TO_KEY: Dict[str, str] = {
    "openai_handler": "gpt",
    "gptoss_handler": "gpt-oss",
    "qwen_handler":   "qwen",
    "gemma_handler":  "gemma",
    "llama_handler":  "llama",
    "claude_handler": "claude",
}

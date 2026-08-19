import os
import shlex
import json
import time
import re
import asyncio
import textwrap
import yaml
import signal
import subprocess
import threading
import requests as http_requests
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

try:
    from handler_llama import LocalLlamaModel
except ImportError:
    LocalLlamaModel = None


# ---------------------------------------------------------------------------
# llama-server 프로세스 관리 + HTTP shim
# ---------------------------------------------------------------------------

_global_gemma_server_manager = None
_global_translation_server_manager = None


def get_server_manager():
    """qa_llm.py shutdown 시 접근용"""
    return _global_gemma_server_manager


def get_translation_server_manager():
    """qa_llm.py shutdown 시 접근용"""
    return _global_translation_server_manager


# harmony/channel/thought 제어 토큰 (닫힘 누락·중첩·깨진 변종 포함) 일괄 제거용.
# 키워드 뒤에 | 또는 > 가 따라오는 토큰형만 매칭 → 일반 단어 오제거 최소화.
_GEMMA_ARTIFACT_RE = re.compile(
    r"<\s*/?\s*\|?\s*(?:channel|message|thought|analysis|final|start|end|turn)\s*\|?\s*>?",
    re.IGNORECASE,
)


def _strip_gemma_tokens(text: str) -> str:
    """Gemma 4 think 모드 답변에서 <channel|> 이전 (= thinking 부분) 제거.

    2026-05-21 실측 (50건 raw 응답, stop list & strip 우회):
    - 모델은 prompt 가 `<|channel>thought\\n` 으로 열어둘 때 답변을 항상 `<channel|>` 으로
      시작 (thinking 부분 실제 출력 0자).
    - 그 외 모델 토큰 (<turn|>, <eos>, <bos>, <start_of_turn>, <end_of_turn>, <think> 등)
      은 출현 0건. 자연 EOG 로 종료.
    - 따라서 strip 의 본질은 답변 시작부의 `<channel|>` 제거 하나로 충분.

    stop list 4종 (<turn|>, <end_of_turn>, <eos>, <|endoftext|>) 은 안전망으로 유지하되
    실측 출현 0건. 본 strip 함수는 그 외 모든 legacy 룰을 제거한 minimal 버전이다.
    """
    if not text:
        return text

    # 1) 첫 <channel|> 이전을 thinking 으로 간주하고 cut. 토큰 자체도 제거.
    ch_close = text.find("<channel|>")
    if ch_close >= 0:
        text = text[ch_close + len("<channel|>"):]

    # 2) 잔여 harmony/channel/thought 토큰 제거.
    #    닫는 <channel|> 가 없는 변종(`<|channel>thought` 만 출력)·중첩·깨진 출력 시
    #    위 cut 이 동작하지 않아 토큰이 응답에 남는다. 이 응답이 멀티턴/eval judge 로
    #    재투입되면 llama-server 가 입력을 토크나이즈하지 못해
    #    500 "Failed to parse input at pos 0" 을 던진다 → 아래에서 일괄 제거.
    text = _GEMMA_ARTIFACT_RE.sub("", text)
    # 3) 깨진 유니코드 치환문자(U+FFFD) 제거.
    text = text.replace("�", "")

    return text.strip()



class LlamaServerManager:
    """llama-server 자식 프로세스를 관리한다 (시작/중지/헬스체크/자동 재시작)."""

    def __init__(self, config: dict, model_path: str):
        self._config = config
        self._model_path = model_path
        self._process = None
        self._lock = threading.RLock()
        self._base_url = f"http://{config['host']}:{config['port']}"
        # 도커처럼 별도 컨테이너에서 llama-server 가 이미 돌고 있을 때 사용.
        # True 면 binary 띄우지 않고 /health 만 폴링하여 ready 확인.
        self._external = bool(config.get('external_server', False))

        # 프로세스 종료 시 자식도 정리되도록 atexit 등록 (external 모드는 우리 관리 아님)
        if not self._external:
            import atexit
            atexit.register(self.stop)

    # -- lifecycle ------------------------------------------------------------

    def start(self, timeout: int = 120):
        with self._lock:
            # external 모드: subprocess 안 띄우고 /health 만 폴링 (도커 컴포즈 환경)
            if self._external:
                deadline = time.time() + timeout
                while time.time() < deadline:
                    try:
                        r = http_requests.get(f"{self._base_url}/health", timeout=2)
                        if r.status_code == 200 and r.json().get("status") == "ok":
                            logger.info(f"✅ external llama-server ready: {self._base_url}")
                            return
                    except Exception:
                        pass
                    time.sleep(2)
                raise TimeoutError(f"external llama-server {timeout}초 내 ready 안 됨: {self._base_url}")

            if self._process and self._process.poll() is None:
                logger.info("llama-server 이미 실행 중")
                return

            cfg = self._config
            cmd = [
                cfg['binary_path'],
                '-m', self._model_path,
                '--host', cfg['host'],
                '--port', str(cfg['port']),
                '-ngl', str(cfg['n_gpu_layers']),
                '-c', str(cfg['n_ctx']),
                '-np', str(cfg['n_parallel']),
                '-b', str(cfg['n_batch']),
            ]
            # KV 캐시 재사용 (원본 운영 서버 운용값이 레거시 기본) — config cache_reuse/slot_prompt_similarity 로 조정
            # docs/kv-cache-reuse.md 참조. prefix 재사용→TTFT 단축, 유사 프롬프트→캐시 보유 슬롯 라우팅
            _cr = cfg.get('cache_reuse')
            _sps = cfg.get('slot_prompt_similarity')
            cmd += ['--cache-reuse', str(_cr if _cr is not None else 256),
                    '--slot-prompt-similarity', str(_sps if _sps is not None else 0.5)]
            # 추가 서버 인자 — 카트리지/설치기가 정의 (예: --cpu-moe, --flash-attn)
            if cfg.get('extra_args'):
                cmd += shlex.split(cfg['extra_args'])

            logger.info(f"llama-server 시작: {' '.join(cmd)}")
            # DEVNULL이면 OOM/alloc 실패 원인이 어디에도 안 남는다 — 파일로 보존 (verification-log #11)
            os.makedirs('logs', exist_ok=True)
            _srv_log = open(os.path.join('logs', 'llama-server-gemma.log'), 'ab')
            self._process = subprocess.Popen(
                cmd,
                stdout=_srv_log,
                stderr=_srv_log,
                preexec_fn=os.setsid,
            )
            _srv_log.close()  # 자식이 fd 상속 — 부모 핸들은 닫는다

            # /health 폴링하여 ready 대기
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    r = http_requests.get(f"{self._base_url}/health", timeout=2)
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("status") == "ok":
                            logger.info("✅ llama-server is ready")
                            return
                except Exception:
                    pass

                if self._process.poll() is not None:
                    raise RuntimeError(f"llama-server 시작 실패 (exit={self._process.returncode})")

                time.sleep(2)

            self.stop()
            raise TimeoutError(f"llama-server {timeout}초 내 ready 안 됨")

    def stop(self):
        with self._lock:
            if self._external:
                return  # 외부 서버는 앱이 관리하지 않음
            if self._process is None:
                return
            pid = self._process.pid
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
                self._process.wait(timeout=10)
            except (ProcessLookupError, OSError):
                pass
            except Exception:
                # SIGTERM 실패 시 SIGKILL
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    self._process.wait(timeout=5)
                except Exception:
                    # 프로세스 그룹 접근 실패 시 직접 kill
                    try:
                        self._process.kill()
                        self._process.wait(timeout=5)
                    except Exception:
                        pass
            logger.info(f"llama-server 종료됨 (pid={pid})")
            self._process = None

    def is_alive(self) -> bool:
        if self._external:
            try:
                r = http_requests.get(f"{self._base_url}/health", timeout=2)
                return r.status_code == 200 and r.json().get("status") == "ok"
            except Exception:
                return False
        return self._process is not None and self._process.poll() is None

    def ensure_running(self):
        if self.is_alive():
            return
        logger.warning("llama-server 죽어있음 — 재시작 시도")
        self.start()

    @property
    def base_url(self) -> str:
        return self._base_url


class _LlamaServerShim:
    """llama-server HTTP API를 기존 llama-cpp-python 인터페이스처럼 제공한다.

    LocalGemmaModel.llm 을 이 객체로 교체하면 기존 코드가 그대로 동작한다.
    """

    def __init__(self, manager: LlamaServerManager):
        self._mgr = manager

    def tokenize(self, text_bytes: bytes) -> list:
        self._mgr.ensure_running()
        content = text_bytes.decode('utf-8') if isinstance(text_bytes, bytes) else text_bytes
        r = http_requests.post(
            f"{self._mgr.base_url}/tokenize",
            json={"content": content},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("tokens", [])

    def create_completion(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01,
                          top_p: float = 0.9, repeat_penalty: float = 1.1, top_k: int = 40,
                          stop: list = None, echo: bool = False, stream: bool = False,
                          frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                          min_p: float = 0.0, repeat_last_n: int = 64,
                          **kwargs) -> Any:
        self._mgr.ensure_running()

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop or [],
            "stream": stream,
        }

        if stream:
            return self._stream_completion(payload)

        r = http_requests.post(
            f"{self._mgr.base_url}/completion",
            json=payload,
            timeout=300,
        )
        if not r.ok:
            # 400/5xx 시 응답 본문 같이 노출 (디버그용 — llama-server 가 거부한 이유)
            logger.error(f"llama-server 응답 {r.status_code}: body={r.text[:500]}, payload_keys={list(payload.keys())}, n_predict={payload.get('n_predict')}, prompt_len={len(payload.get('prompt',''))}")
        r.raise_for_status()
        data = r.json()

        # llama-cpp-python 호환 형식으로 정규화
        return {
            "choices": [{"text": data.get("content", "")}],
            "usage": {
                "prompt_tokens": data.get("tokens_evaluated", 0),
                "completion_tokens": data.get("tokens_predicted", 0),
                "total_tokens": data.get("tokens_evaluated", 0) + data.get("tokens_predicted", 0),
            },
        }

    def _stream_completion(self, payload: dict):
        """SSE 스트림을 llama-cpp-python 호환 제너레이터로 변환한다."""
        r = http_requests.post(
            f"{self._mgr.base_url}/completion",
            json=payload,
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        r.encoding = "utf-8"  # llama-server SSE 는 UTF-8 — 미설정 시 requests 가 latin-1 로 오디코드(한글 깨짐)

        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                yield {"choices": [{"text": data.get("content", "")}]}
                if data.get("stop", False):
                    break
            except json.JSONDecodeError:
                continue

class LocalGemmaModel:
    def __init__(self, llm, server_manager: LlamaServerManager = None):
        self._server_manager = server_manager
        self.llm = llm
        # llama-cpp-python 모드에서는 동시 접근 방지용 Lock
        self._lock = threading.Lock() if server_manager is None else None
        logger.info(f"✅ LocalGemmaModel 초기화 완료 ({'llama-server' if server_manager else 'llama-cpp-python'} 모드)")

    @classmethod
    def create_with_config(cls) -> 'LocalGemmaModel':
        global _global_gemma_server_manager

        config_manager = ConfigManager()
        paths_config = config_manager.get_paths_config()
        server_config = config_manager.get_llama_server_gemma_config()

        is_external = bool(server_config.get('external_server', False))

        model_path = str(Path(paths_config.get('gemma_model_path', '')).resolve())
        # external_server 면 모델 파일은 외부 컨테이너가 보유 — 앱 측 존재 체크 skip
        if not is_external and not os.path.exists(model_path):
            raise FileNotFoundError(f"Gemma 모델 파일을 찾을 수 없습니다: {model_path}")

        if server_config.get('use_server_mode', False):
            # llama-server HTTP 모드
            if not is_external and not os.path.exists(server_config['binary_path']):
                raise FileNotFoundError(f"llama-server 바이너리를 찾을 수 없습니다: {server_config['binary_path']}")

            mgr = LlamaServerManager(server_config, model_path)
            mgr.start()
            _global_gemma_server_manager = mgr
            return cls(_LlamaServerShim(mgr), server_manager=mgr)
        else:
            # 기존 llama-cpp-python 직접 로드 모드
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python 라이브러리가 필요합니다.")

            context_size = int(os.getenv("LLAMA_CONTEXT_SIZE", "32768"))
            logger.info(f"🚀 로컬 Gemma 모델 로드 중: {model_path} (컨텍스트={context_size})")

            llm = Llama(
                model_path=model_path,
                n_ctx=context_size,
                n_threads=8,
                n_gpu_layers=24,
                n_batch=1024,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                logits_all=False,
                embedding=False,
                mul_mat_q=True,
            )
            return cls(llm)

    def count_tokens(self, text: str) -> int:
        if self._lock:
            with self._lock:
                return len(self.llm.tokenize(text.encode('utf-8')))
        return len(self.llm.tokenize(text.encode('utf-8')))

    def generate_raw(self, prompt: str, max_tokens: int = 1024, **kwargs) -> dict:
        if self._lock:
            with self._lock:
                return self.llm.create_completion(prompt=prompt, max_tokens=max_tokens, **kwargs)
        return self.llm.create_completion(prompt=prompt, max_tokens=max_tokens, **kwargs)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3, top_p: float = 0.95) -> Dict:
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.15,
            top_k=64,
            stop=[
                "<end_of_turn>",
                "<eos>",
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

    def generate_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3, top_p: float = 0.95, repeat_penalty: float = 1.15, top_k: int = 64, frequency_penalty: float = 0.5):
        # llama-cpp-python 모드에서는 Lock으로 직렬화
        if self._lock:
            self._lock.acquire()
        try:
            yield from self._generate_stream_inner(prompt, max_tokens, temperature, top_p, repeat_penalty, top_k, frequency_penalty)
        finally:
            if self._lock:
                self._lock.release()

    def _generate_stream_inner(self, prompt, max_tokens, temperature, top_p, repeat_penalty, top_k, frequency_penalty=0.5):
        buffer = ""
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            stop=[
                "<end_of_turn>",
                "<eos>",
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

    def generate_complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3, top_p: float = 0.95) -> str:
        try:
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.15,
                top_k=64,
                stop=[
                    "<end_of_turn>",
                    "<eos>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )
            result = response["choices"][0]["text"].strip()

            # Gemma 토큰 정리
            unwanted_tokens = ["<end_of_turn>", "<eos>", "<bos>", "<start_of_turn>"]
            for token in unwanted_tokens:
                result = result.replace(token, "")

            return result.strip()

        except Exception as e:
            logger.error(f"Gemma 생성 오류: {e}")
            return ""


    def generate_agent(self, prompt: str, max_tokens: int = 131072, temperature: float = 0.3,
                           top_p: float = 0.95) -> str:
        """Agent용 응답 생성. prompt는 호출측(build_agent_prompt)이 이미 chat-template 포맷 완료."""
        try:
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.15,
                top_k=64,
                stop=[
                    "<end_of_turn>",
                    "<eos>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )

            result = response["choices"][0]["text"]
            print(f"🔬 [generate_agent] raw_text len={len(result)}, finish={response.get('choices', [{}])[0].get('finish_reason')}")
            cleaned = _strip_gemma_tokens(result)
            print(f"🔬 [generate_agent] cleaned len={len(cleaned)}: {cleaned[:300]}")
            return cleaned

        except Exception as e:
            logger.error(f"Gemma Agent 생성 오류: {e}")
            return ""

class GemmaHandler(BaseModelHandler):
    registry_key = "gemma"  # handler_registry.HANDLER_CLASSES 키와 일치
    is_local = True          # 로컬 GGUF / llama-server (RoPE 2x 보간)

    def __init__(self, rag_system=None, use_kg: bool = True, intent_analyzer=None):
        super().__init__(rag_system, use_kg, intent_analyzer)

        self.config_manager = ConfigManager()

        self.handler_type = "gemma"
        # 실제 서빙 GGUF 파일명에서 라벨 파생 — 하드코딩 'gemma-4-26b-a4b'는 12b 프리셋도
        # 26b로 오표기해 A/B 정체성 검증을 오염시킴 [2026-07-23 수정]
        try:
            _gguf_stem = Path(self.config_manager.get_paths_config().get('gemma_model_path', '')).stem
        except Exception:
            _gguf_stem = ''
        self.model_name = _gguf_stem or 'gemma-4-26b-a4b'
        # OpenAI 호환 경로(usage.model)도 같은 라벨 사용 — 클래스 기본값 하드코딩을 실서빙 파일명으로 덮는다
        self.agent_model_name = self.model_name
        self.local_model = None
        self.available = False

        prompt_path = self.config_manager.get_prompt_config().get('gemma')
        if not prompt_path:
            prompt_path = self.config_manager.get_prompt_config().get('gpt_oss')
        if not prompt_path:
            prompt_path = self.config_manager.get_prompt_config().get('gpt')

        self.system_prompt = self.load_prompt_from_file(prompt_path)

        self.logger = ChatLogger(self.model_name)

        self._initialize_model()

        self.translation_llama = None
        self._init_translation_llama()

        self._init_pii_gate()

    # PII 게이트(_init_pii_gate·_pii_mask_input)는 handler_base 공통으로 올라갔다 (S-6).
    # 여기서는 기존 순서대로 명시 초기화만 하고, 구현은 베이스를 그대로 쓴다.

    def _init_translation_llama(self):
        """번역 모델 선택 — config 의 [llama_server_translation] use_server_mode 로 토글.

        - True: 별도 llama-server (llama-3.1 8B 등) 띄워서 번역. 기존 동작.
        - False: self.local_model (Gemma 4) 을 번역에도 재사용 — 별도 모델 미사용.

        _perform_translation 가 self.translation_use_server flag 보고 prompt chat template 선택.
        """
        global _global_translation_server_manager
        trans_config = self.config_manager.get_llama_server_translation_config()
        self.translation_use_server = bool(trans_config.get('use_server_mode', False))

        if self.translation_use_server:
            # 모드 A: 별도 llama-server 띄움 (llama-3.1 chat template 사용)
            try:
                paths_config = self.config_manager.get_paths_config()
                model_path = str(Path(paths_config.get('local_llama_model_path', '')).resolve())
                is_external = bool(trans_config.get('external_server', False))

                if not is_external and not os.path.exists(model_path):
                    raise FileNotFoundError(f"번역 모델 파일 없음: {model_path}")

                logger.info("🦙 번역 전용 llama-server 시작 중...")
                mgr = LlamaServerManager(trans_config, model_path)
                mgr.start()
                _global_translation_server_manager = mgr

                class _LlamaServerTransWrapper:
                    def __init__(self, shim):
                        self.llm = shim
                    def generate_complete(self, prompt, max_tokens=200, temperature=0.1, **kwargs):
                        r = self.llm.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                        return r["choices"][0]["text"].strip()

                self.translation_llama = _LlamaServerTransWrapper(_LlamaServerShim(mgr))
                logger.info("✅ 번역 전용 llama-server 로드 완료 (use_server_mode=True)")
                return
            except Exception as e:
                logger.warning(f"⚠️ 번역 전용 llama-server 로드 실패: {e}. 응답 모델로 fallback")
                self.translation_use_server = False
                # fall through to local_model 재사용

        # 모드 B: self.local_model (Gemma 4) 재사용 (Gemma chat template)
        if not self.available or self.local_model is None:
            logger.warning("⚠️ Gemma local_model 미로드 — 번역도 사전 fallback 사용")
            self.translation_llama = None
            return

        class _GemmaTranslationWrapper:
            def __init__(self, gemma_local):
                self._gemma = gemma_local

            def generate_complete(self, prompt, max_tokens=200, temperature=0.1, **kwargs):
                kwargs.setdefault('stop', ['<turn|>', '<end_of_turn>', '<eos>', '<|endoftext|>'])
                r = self._gemma.generate_raw(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                raw = r["choices"][0]["text"]
                return _strip_gemma_tokens(raw)

        self.translation_llama = _GemmaTranslationWrapper(self.local_model)
        logger.info("✅ 번역도 응답 모델(Gemma 4)로 처리 — 별도 모델 미사용 (use_server_mode=False)")

    def _initialize_model(self):
        try:
            self.local_model = LocalGemmaModel.create_with_config()
            self.available = True
            mode = "llama-server" if self.local_model._server_manager else "llama-cpp-python"
            logger.info(f"✅ 로컬 Gemma 모델 초기화 완료 ({mode})")
        except Exception as e:
            logger.error(f"❌ Gemma 모델 초기화 실패: {str(e)}")
            self.local_model = None
            self.available = False
    
    def _perform_translation(self, query: str, mode: str, preserved: dict = None) -> str:
        try:
            if not self.available:
                logger.warning("Llama 모델이 사용 불가능하여 사전 번역 사용")
                return self._enhanced_dictionary_translate(query)

            print(f"🔍 번역 모드: {mode}, use_server_mode={getattr(self, 'translation_use_server', False)}")
            # mode 별 system instructions (chat template 과 무관, 공통)
            # 번역 시스템 프롬프트는 카트리지가 정의 — [prompts] translation_summary/normal/cve
            from config_utils import load_translation_prompt
            system_instructions = load_translation_prompt(mode)

            # chat template 분기 — use_server_mode 따라 모델별 형식 적용
            if getattr(self, 'translation_use_server', False):
                # 별도 llama-server (llama-3.1 chat template)
                prompt = (
                    f"<|start_header_id|>system<|end_header_id|>\n{system_instructions}\n<|eot_id|>\n"
                    f"<|start_header_id|>user<|end_header_id|>\n{query}\n<|eot_id|>\n"
                    f"<|start_header_id|>assistant<|end_header_id|>\n"
                )
            else:
                # self.local_model 재사용 (Gemma 4 chat template, thinking 모드)
                prompt = f"<|turn>user\n{system_instructions}\n\n{query}<turn|>\n<|turn>model\n<|channel>thought\n"
            
            if self.translation_llama:
                translation = self.translation_llama.generate_complete(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.1
                )
            else:
                translation = ""
    
            # 로그 출력용: 플레이스홀더를 원래 토큰으로 복원
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
            logger.error(f"LLaMA 번역 오류: {e}")
            return self._enhanced_dictionary_translate(query)

    async def generate_complete(self, request, user_id=None, channel_id=None, connection=None, thread_ts=None,
                          context=None, source_files=None, related_concepts=None, concepts=None, intent_info=None) -> Dict:
        start_time = time.time()
        self._usage_tracker = UsageTracker(model=self.model_name, user_id=user_id or "")

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
                "answer": "Gemma 모델을 사용할 수 없습니다.",
                "intent": None,
                "sources": [],
                "response_time": 0,
                "question_type": question_type
            }

        # ── PII 마스킹: 사용자 입력을 RAG·프롬프트 조립 전에 단방향 치환 ──
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
            intent_upper = selected_intent.upper()

            if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                post_action_path = self.config_manager.get_prompt_config().get('post_action')
                print(f"🔍 일반 POST-ACTION 프롬프트 경로: {post_action_path}")
                if post_action_path:
                    gpt_prompt = self.load_prompt_from_file(post_action_path)
                    self.system_prompt = self._clean_prompt_content(gpt_prompt)
                    print(f"🔧 일반 POST-ACTION 프롬프트 로드 완료: {post_action_path}")
                else:
                    print(f"❌ POST-ACTION 프롬프트 경로가 None입니다!")
                    fallback_path = "./prompts/system/post_action.yaml"
                    print(f"🔧 임시 경로 사용: {fallback_path}")
                    gpt_prompt = self.load_prompt_from_file(fallback_path)
                    self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'ACTION':
                # Gemma 전용 action prompt 우선, 없으면 action_gpt fallback
                prompts_cfg = self.config_manager.get_prompt_config()
                gemma_prompt_path = prompts_cfg.get('action_gemma') or prompts_cfg.get('action_gpt')
                gpt_prompt = self.load_prompt_from_file(gemma_prompt_path)
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'PLAN':
                print(f"🔧 [PLAN Gemma] 일반 계획 수립 프롬프트 사용")
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('plan_gpt'))
                print(f"✅ [PLAN Gemma] 일반 계획 프롬프트 로드 완료")

                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'PLAYBOOK':
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('playbook'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'TITLE':
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('title'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            else:
                # Gemma 전용 qna prompt 우선, 없으면 qna_gpt fallback
                prompts_cfg = self.config_manager.get_prompt_config()
                gemma_prompt_path = prompts_cfg.get('qna_gemma') or prompts_cfg.get('qna_gpt')
                gpt_prompt = self.load_prompt_from_file(gemma_prompt_path)
                self.system_prompt = self._clean_prompt_content(gpt_prompt)

        # locale 지시문 주입
        locale = request.get("locale", "") if isinstance(request, dict) else ""
        if locale:
            self.system_prompt += f"\n\n[LANGUAGE RULE]\nYou MUST respond in the language corresponding to locale '{locale}'. This overrides all other language rules."

        history_context = sanitize_history(history, current_query=query)

        chat_session.add_message("user", query)

        formatted_prompt = self.create_clean_prompt(
            query=query,
            context=context,
            chat_history=history_context,
            intent=selected_intent.upper() if selected_intent else "",
            response_format = response_format,
            question_type=question_type,
            request=request
        )

        """전체 Gemma 메시지 구조 로깅 (generate_complete)"""
        import time as time_module
        import json
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/gemma/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write(f"Method: generate_complete\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE Gemma PROMPT STRUCTURE\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"📝 Formatted Prompt:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{formatted_prompt}\n")
                f.write("\n" + "=" * 60 + "\n\n")

                f.write(f"System Prompt Length: {len(self.system_prompt)} chars\n")
                f.write(f"Context Length: {len(context)} chars\n")
                f.write(f"History Context Length: {len(history_context)} chars\n")
                f.write(f"Full Prompt Length: {len(formatted_prompt)} chars\n")

                f.write(f"\nPrompt File Used: ")
                if selected_intent:
                    intent_upper = selected_intent.upper()
                    if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                        f.write("post_action")
                    elif intent_upper == 'ACTION':
                        f.write("action_gemma")
                    elif intent_upper == 'PLAN':
                        f.write("plan_gemma")
                    else:
                        f.write("qna_gemma")
                else:
                    f.write("default gemma")
                f.write("\n")

                f.write(f"RAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")

                f.write(f"Related Concepts: {len(related_concepts)}\n")
                for k, concept in enumerate(related_concepts[:5]):
                    concept_name = concept.get('concept', 'Unknown') if isinstance(concept, dict) else str(concept)
                    f.write(f"  {k+1}. {concept_name}\n")

        except Exception as log_e:
            print(f"⚠️ Gemma 로그 저장 실패 (generate_complete): {log_e}")

        try:
            # 토큰 사용량 분석
            def count_tokens(text):
                return self.local_model.count_tokens(text)

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
            print(f"  - 컨텍스트 한계: 131,072 토큰")

            if total_prompt_tokens > 131072:
                print(f"❌ 토큰 초과! ({total_prompt_tokens:,} > 131,072)")

                # 참조 문서별 토큰 분석
                if source_files and context:
                    print(f"📄 참조 문서별 토큰 분석:")
                    context_lines = context.split('\n')
                    current_doc = ""
                    current_tokens = 0
                    doc_count = 0

                    for line in context_lines:
                        if line.strip().startswith('📄 문서 #'):
                            if current_doc and current_tokens > 0:
                                print(f"  - {current_doc}: {current_tokens:,} 토큰")
                            current_doc = line.strip()
                            current_tokens = 0
                            doc_count += 1
                        else:
                            current_tokens += count_tokens(line + '\n')

                    if current_doc and current_tokens > 0:
                        print(f"  - {current_doc}: {current_tokens:,} 토큰")

                    print(f"  총 {doc_count}개 참조 문서, 평균 {context_tokens//max(doc_count,1):,} 토큰/문서")

                return {
                    "answer": f"프롬프트가 너무 깁니다. ({total_prompt_tokens:,} 토큰 > 131,072 한계)",
                    "intent": None,
                    "sources": [],
                    "response_time": 0,
                    "question_type": question_type
                }

            start_model_time = time.time()
            print(f"🚀 Gemma 응답 생성 시작 (generate_complete)...")
            response = await asyncio.to_thread(
                self.local_model.generate_raw,
                prompt=formatted_prompt,
                max_tokens=4096,           # mode collapse 시 폭주 방지 cap
                temperature=0.3,
                top_p=0.9,
                top_k=40,
                min_p=0.05,                # 분포 꼬리 cut (이상 토큰 회피)
                repeat_penalty=1.18,
                repeat_last_n=128,
                frequency_penalty=0.5,
                presence_penalty=0.3,
                stop=[
                    # 안전망 — 2026-05-21 실측 50건에서 출현 0건이지만 모델 drift 대비 유지
                    "<turn|>",         # 공식 Gemma 4 turn 종료
                    "<end_of_turn>",   # 호환: 모델이 옛 토큰 출력해도 대응
                    "<eos>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )
            end_model_time = time.time()
            self._usage_tracker.track(response)
            print(f"✅ Gemma 응답 생성 완료 {end_model_time - start_model_time:.2f}초 (generate_complete)")

            full_response = response["choices"][0]["text"].strip()

            # ── PII 복원: restore=True 면 응답의 토큰을 원본값으로 역치환 (handler_base 공통) ──
            full_response = self._pii_unmask(full_response)

            print(f"📝 전체 응답 (generate_complete): {full_response}")

            # 누수 검증용: strip 전 raw 보존 (think 모드 검증 — task 1)
            raw_model_output = full_response

            # Clarify 응답 처리 (ACTION 타입에서 추가 정보 요청시)
            if (request.get("type", "").upper() == "ACTION" and
                ('"action": "clarify"' in full_response or '"action":"clarify"' in full_response)):
                original_intent = request.get("type", "ACTION")
                print(f"🔄 Clarify 응답 감지 (Gemma): {original_intent} 의도 유지")

                try:
                    clarify_data = json.loads(full_response)
                    if isinstance(clarify_data, dict) and "message" in clarify_data:
                        full_response = clarify_data["message"]
                        print(f"📝 Clarify 메시지 추출 (Gemma): {full_response}")
                except json.JSONDecodeError:
                    match = re.search(r'"message":\s*"([^"]*)"', full_response)
                    if match:
                        full_response = match.group(1)
                        print(f"📝 Clarify 메시지 추출 (정규식, Gemma): {full_response}")

            # Gemma 토큰 정리
            final_response = self._get_clean_message(full_response)
            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                final_response = self._remove_markdown_formatting(final_response)

            # ACTION 응답이 pipe 로 시작하는 fragment 면 history 의 base 쿼리 자동 prepend
            if request.get("type", "").upper() == "ACTION":
                final_response = self._repair_query_fragment(final_response, history)

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
            is_action = (request_type == "ACTION" or selected_intent_upper == "ACTION")


            if is_post_action:
                print(f"📝 POST_ACTION 감지: 쿼리 결과 분석 모드")
                print(f"🔧 일반 POST-ACTION: 전체 응답 사용")

            processed_response = self._fix_unicode_response(final_response)

            show_metadata_config = self.config_manager.get_option_config().get('show_metadata')
            print(f"🔍 Gemma 메타데이터 조건 체크: request_type='{request_type}', selected_intent='{selected_intent_upper}', show_metadata={show_metadata_config}")

            if (show_metadata_config and not is_post_action):
                detected_intent = selected_intent or (intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown')
                if detected_intent != 'unknown':
                    intent_text = f"🎯 **의도:** **{detected_intent.upper()}**"
                    if intent_text not in processed_response:
                        processed_response += f"\n\n{intent_text}"

                if source_files:
                    refs_text = f"📚 **참조 문서** ({len(source_files)}개):"

                    if 'reference_summary' in locals() and reference_summary and reference_summary.get('is_incremental_mode'):
                        refs_text += f"\n📊 전체: {reference_summary['total_documents']}개 | 기존: {reference_summary['existing_count']}개 | 신규: {reference_summary['temporary_count']}개"

                        for ref in reference_summary['detailed_references']:
                            status_icon = "🆕" if ref['is_new'] else "✅"
                            refs_text += f"\n{ref['rank']}. {status_icon} `{ref['file']}` (점수: {ref['score']})"
                    else:
                        refs_text += ''.join([f"\n{i+1}. `{src}`" for i, src in enumerate(source_files)])
                else:
                    refs_text = "📚 **참조 문서:** 없음"

                if refs_text not in processed_response:
                    processed_response += f"\n\n{refs_text}"

                time_text = f"⏱️ 답변 시간: {end_time - start_time:.2f}초"
                if time_text not in processed_response:
                    processed_response += f"\n\n{time_text}"

            self.system_prompt = original_prompt

            is_plan = (request_type == "PLAN" or selected_intent_upper == "PLAN")

            if is_plan:
                print(f"🔒 [Gemma] PLAN 응답 생성 완료 - 워크플로우는 aibot_llm_module.py에서 처리됨")

            return {
                "answer": processed_response,
                "raw_answer": raw_model_output,
                "intent": (selected_intent or intent_info.get('primary_intent', 'unknown') if intent_info else 'unknown').upper(),
                "intent_info": intent_info,
                "sources": source_files,
                "response_time": end_time - start_time,
                "question_type": question_type,
                "usage": self._usage_tracker.to_dict(),
            }

        except Exception as e:
            self.system_prompt = original_prompt

            logger.error(f"Gemma generate_complete [{type(e).__name__}]: {e}")
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
            yield "Gemma 모델을 사용할 수 없습니다."
            return

        # ── PII 마스킹: 사용자 입력을 RAG·프롬프트 조립 전에 단방향 치환 ──
        query = self._pii_mask_input(request, query)

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

        if self.config_manager.get_option_config().get('user_intent_prompt') == 'True' and selected_intent:
            intent_upper = selected_intent.upper()

            if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                post_action_path = self.config_manager.get_prompt_config().get('post_action')
                print(f"🔍 일반 POST-ACTION 프롬프트 경로: {post_action_path}")
                if post_action_path:
                    gpt_prompt = self.load_prompt_from_file(post_action_path)
                    self.system_prompt = self._clean_prompt_content(gpt_prompt)
                    print(f"🔧 일반 POST-ACTION 프롬프트 로드 완료: {post_action_path}")
                else:
                    print(f"❌ POST-ACTION 프롬프트 경로가 None입니다!")
                    fallback_path = "./prompts/system/post_action.yaml"
                    print(f"🔧 임시 경로 사용: {fallback_path}")
                    gpt_prompt = self.load_prompt_from_file(fallback_path)
                    self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'ACTION':
                prompts_cfg = self.config_manager.get_prompt_config()
                gemma_prompt_path = prompts_cfg.get('action_gemma') or prompts_cfg.get('action_gpt')
                gpt_prompt = self.load_prompt_from_file(gemma_prompt_path)
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'PLAN':
                print(f"🔧 [PLAN Gemma] 일반 계획 수립 프롬프트 사용")
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('plan_gpt'))
                print(f"✅ [PLAN Gemma] 일반 계획 프롬프트 로드 완료")

                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'TITLE':
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('title'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            else:
                prompts_cfg = self.config_manager.get_prompt_config()
                gemma_prompt_path = prompts_cfg.get('qna_gemma') or prompts_cfg.get('qna_gpt')
                gpt_prompt = self.load_prompt_from_file(gemma_prompt_path)
                self.system_prompt = self._clean_prompt_content(gpt_prompt)

        # locale 지시문 주입
        locale = request.get("locale", "") if isinstance(request, dict) else ""
        if locale:
            self.system_prompt += f"\n\n[LANGUAGE RULE]\nYou MUST respond in the language corresponding to locale '{locale}'. This overrides all other language rules."

        chat_session.add_message("user", query)

        formatted_prompt = self.create_clean_prompt(query, context, "", selected_intent.upper() if selected_intent else "", response_format, question_type, request)

        """전체 Gemma 메시지 구조 로깅"""
        import time as time_module
        import json
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/gemma/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE Gemma PROMPT STRUCTURE\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"📝 Formatted Prompt:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{formatted_prompt}\n")
                f.write("\n" + "=" * 60 + "\n\n")

                f.write(f"System Prompt Length: {len(self.system_prompt)} chars\n")
                f.write(f"Context Length: {len(context)} chars\n")
                f.write(f"Full Prompt Length: {len(formatted_prompt)} chars\n")

                f.write(f"\nPrompt File Used: ")
                if selected_intent:
                    intent_upper = selected_intent.upper()
                    if intent_upper in ['POST_ACTION', 'POST-ACTION']:
                        f.write("post_action")
                    elif intent_upper == 'ACTION':
                        f.write("action_gemma")
                    elif intent_upper == 'PLAN':
                        f.write("plan_gemma")
                    else:
                        f.write("qna_gemma")
                else:
                    f.write("default gemma")
                f.write("\n")

                f.write(f"RAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")

                f.write(f"Related Concepts: {len(related_concepts)}\n")
                for k, concept in enumerate(related_concepts[:5]):
                    concept_name = concept.get('concept', 'Unknown') if isinstance(concept, dict) else str(concept)
                    f.write(f"  {k+1}. {concept_name}\n")

        except Exception as log_e:
            print(f"⚠️ Gemma 로그 저장 실패: {log_e}")

        try:
            full_response = ""
            message_started = False

            # ── PII 복원(스트리밍) — 구현은 handler_base._pii_stream_restorer (S-6 공통화) ──
            _restore_emit = self._pii_stream_restorer()

            for chunk in self.local_model.generate_stream(
                prompt=formatted_prompt,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.95,
                repeat_penalty=1.15,
                top_k=64,
                frequency_penalty=0.5
            ):
                if chunk:
                    full_response += chunk

                    if not message_started:
                        # Gemma 는 채널 구조 없음 — 응답 시작부에서 thinking 토큰만 제거
                        if True:
                            content_after_tag = _strip_gemma_tokens(full_response)
                            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                                content_after_tag = self._remove_markdown_formatting(content_after_tag)
                            out = _restore_emit(content_after_tag)
                            if out:
                                yield out
                            message_started = True
                    else:
                        if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                            processed_chunk = self._remove_markdown_formatting(chunk)
                            out = _restore_emit(processed_chunk)
                        else:
                            out = _restore_emit(chunk)
                        if out:
                            yield out

            # 보류 중이던 꼬리 버퍼 flush
            tail = _restore_emit("", final=True)
            if tail:
                yield tail

            chat_session.add_message("assistant", full_response)

            request_type_stream = request.get("type", "").upper()
            question_type_upper = question_type.upper() if question_type else ""
            show_metadata_stream = self.config_manager.get_option_config().get('show_metadata')
            print(f"🔍 Gemma 스트리밍 메타데이터 조건: request_type='{request_type_stream}', question_type='{question_type_upper}', show_metadata={show_metadata_stream}")

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

            self.system_prompt = original_prompt

        except Exception as e:
            self.system_prompt = original_prompt

            error_message = f"Gemma 스트리밍 생성 중 오류: {str(e)}"
            logger.error(error_message)
            yield error_message

    def _get_clean_message(self, response: str) -> str:
        """Gemma 응답에서 순수 텍스트 추출 — 모듈 레벨 _strip_gemma_tokens 위임.

        실제 관찰된 thinking 토큰 패턴 (Gemma 4):
            <|channel>thought
<channel|>실제 답변
        (closing tag 가 `<channel|>` 인 비대칭 케이스 포함)
        """
        # turn 종료 후 잘리도록 한 번 split
        content = response
        for end_tag in ["<end_of_turn>", "<eos>"]:
            if end_tag in content:
                content = content.split(end_tag)[0]
        return _strip_gemma_tokens(content)

    def _repair_query_fragment(self, response: str, history: list) -> str:
        """follow-up ACTION 응답이 pipe-fragment (예: `| search ...`) 면
        history 의 마지막 AI 쿼리에서 base scan 명령을 추출해 자동 prepend.

        시스템 prompt 의 가이드(완전한 쿼리 생성)에도 모델이 가끔 fragment 만
        생성하므로 안전망으로 두는 보정 로직.
        """
        if not response or not history:
            return response
        try:
            parsed = json.loads(response)
        except Exception:
            return response
        if not isinstance(parsed, dict):
            return response

        query = parsed.get("query")
        if not isinstance(query, str):
            return response

        # 주석/공백 정리 후 pipe 로 시작하는지 검사
        cleaned_query = re.sub(r"^\s*#.*$", "", query, flags=re.MULTILINE).strip()
        if not cleaned_query.startswith("|"):
            return response  # 정상 완성된 쿼리

        # history 의 마지막 AI 쿼리 응답에서 base scan 추출
        base = None
        for msg in reversed(history):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not content:
                continue
            try:
                prev = json.loads(content) if isinstance(content, str) else content
            except Exception:
                continue
            prev_q = prev.get("query") if isinstance(prev, dict) else None
            if not isinstance(prev_q, str):
                continue
            # 첫 pipe 이전 (base scan command) 만 분리
            first_pipe = prev_q.find("|")
            base_candidate = prev_q[:first_pipe].strip() if first_pipe >= 0 else prev_q.strip()
            if re.match(r"^(log|table|fulltext|search|set)\b", base_candidate, re.IGNORECASE):
                base = base_candidate
                break

        if not base:
            print("⚠️ [fragment-repair] history 에서 base 쿼리 찾기 실패 — 원본 유지")
            return response

        # base + " " + cleaned_query (pipe 포함) 결합
        repaired_query = f"{base} {cleaned_query}".strip()
        parsed["query"] = repaired_query
        # msg 에 사용자에게 알리는 보충 — 원본 보존
        repaired_response = json.dumps(parsed, ensure_ascii=False)
        print(f"🔧 [fragment-repair] base 자동 prepend: '{base}' + '{cleaned_query[:60]}...'")
        return repaired_response


    # _clean_prompt_content: BaseModelHandler default 사용.

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

        # TITLE은 간결한 제목 요약이므로 부가 블록 생략, locale을 맨 끝에 재강조
        if question_type and question_type.upper() == "TITLE":
            locale = request.get("locale", "") if request and isinstance(request, dict) else ""
            if locale:
                system_content += f"\n\n[CRITICAL LANGUAGE RULE]\nYou MUST write the title in the language of locale '{locale}'. This is the highest priority rule."
            user_content = f"Question: {query}"
            if locale:
                user_content += f"\n\n(Respond in locale '{locale}' only)"
            prompt = f"""<start_of_turn>user
                {system_content}

                {user_content}<end_of_turn>
                <start_of_turn>model
"""
            return prompt

        # 공통 최소 hygiene — system prompt 에 이미 [Output Hygiene] 섹션이 있으면 skip.
        # 상세 hygiene 는 각 system prompt 파일의 [Output Hygiene] 참조
        # (qna_gemma.yaml / action_gemma.yaml / gemma_prompt.yaml).
        # PLAN/PLAYBOOK/POST_ACTION 등 hygiene 섹션 없는 경로 만 안전망 추가.
        if "[Output Hygiene]" not in system_content:
            system_content += "\n\n[Output Hygiene]"
            system_content += "\nOutput only the final answer. Never emit thinking markers or chat-template tokens (<think>, <start_of_turn>, <end_of_turn>, <bos>, <eos>)."

        cur_hms = now.strftime('%H%M%S')
        cur_dt14 = f"{current_date}{cur_hms}"
        d_yesterday = (now - timedelta(days=1)).strftime('%Y%m%d')
        d_7 = (now - timedelta(days=6)).strftime('%Y%m%d')
        d_30 = (now - timedelta(days=29)).strftime('%Y%m%d')
        system_content += f"\n\n[CURRENT DATE & TIME]"
        system_content += f"\n📅 Today: {current_date} ({current_readable})"
        system_content += f"\n🕒 NOW: {cur_dt14}  (= $CUR_DATE$$CUR_TIME$ — current moment, second-precision)"
        system_content += f"\n"
        system_content += f"\n⚠️ Time-range rule (CRITICAL — follow the Date & Time Rules section above):"
        system_content += f"\n - Relative time ('최근 N시간/일/주/달', '지금까지', '현재까지') → to = NOW = {cur_dt14}, from = to − N (preserve second-precision)."
        system_content += f"\n     · '최근 7일'   → from={d_7}{cur_hms}, to={cur_dt14}"
        system_content += f"\n     · '최근 30일'  → from={d_30}{cur_hms}, to={cur_dt14}"
        system_content += f"\n     · '최근 48시간' → from=(NOW − 48h), to={cur_dt14}"
        system_content += f"\n - Standalone date ('오늘', '어제', '5월 1일') → day boundary: from=YYYYMMDD000000, to=YYYYMMDD235959 (NEVER NOW, even with '오전/오후/새벽/추이/패턴')."
        system_content += f"\n - '오늘까지' → to={current_date}235959   |   '어제까지' → to={d_yesterday}235959"
        system_content += f"\n"
        system_content += f"\n🚫 Do NOT use future dates unless explicitly asked for future periods!"

        # 대화 히스토리가 있을 때 추가 지침
        if chat_history.strip():
            system_content += """

[CONVERSATION HISTORY CONTEXT]
⚠️ CRITICAL: The user is continuing from previous conversation.
- When user says "마지막" (last), "이전" (previous), "위에서" (above), refer to YOUR previous responses
- When user mentions "그룹" (group), "항목" (item), check what you listed in previous answers
- Prioritize conversation context over keyword-based document search
- If user refers to something from previous messages, DO NOT search for new information unless explicitly asked
- Example: If you previously mentioned "Lazarus Group" as the last item, and user asks about "마지막 그룹", you should explain Lazarus Group, NOT search for "last" function
"""

        locale = request.get("locale", "") if request and isinstance(request, dict) else ""
        if locale:
            user_content = f"[IMPORTANT: You MUST respond in locale '{locale}']\n\nQuestion: {query}"
        else:
            user_content = f"Question: {query}"

        if request and request.get("app_context") and question_type and question_type.upper() in ['ACTION', 'PLAN']:
            user_content += f"\n\n## 시스템 컨텍스트\n현재 설치된 앱: {request['app_context']}"

        # history 를 RAG 앞에 배치 — multi-turn (특히 clarify follow-up) 에서 모델이
        # RAG-우세 system prompt 에 휘둘리지 않도록 history 의 pending state 우선 노출.
        if chat_history.strip():
            user_content += f"\n\n이전 대화:\n{chat_history}"

        if context.strip():
            user_content += f"\n\n관련 문서:\n{context}"

        is_post_action_check = (question_type and question_type.upper() in ["POST_ACTION", "POST-ACTION"])

        if re.search(r'[가-힣]', query) and not is_post_action_check:
            if self.config_manager.get_option_config().get('translation') == 'True':
                translated_question, _ = self.translate_query_for_search(query, intent)
                if translated_question and translated_question != query:
                    user_content += f"\n(English: {translated_question})"
                    print(f"🔍 번역 추가됨: {translated_question}")
        elif is_post_action_check:
            print(f"🔍 POST_ACTION이므로 번역 건너뜀")

        # locale 지시를 user prompt 끝에도 추가 (RAG 컨텍스트에 묻히지 않도록)
        if locale:
            user_content += f"\n\n[REMINDER: Respond in locale '{locale}' only]"

        # Gemma 4 공식 chat template 토큰 사용 (GGUF tokenizer.chat_template 기준)
        # - turn 시작 <|turn>{role}\n
        # - turn 종료 <turn|>\n
        # - thinking 모드 (enable_thinking=True 상당): <|channel>thought\n 까지만 열어두고
        #   모델이 reasoning 출력 → <channel|> 으로 종료 → 답변 본문 출력. 답변 추출 시 thinking 부분 cut.
        prompt = f"""<|turn>user
                {system_content}

                {user_content}<turn|>
                <|turn>model
<|channel>thought
"""

        return prompt

    # ── Agent (completions2) — Gemma-4 공식 chat template 캡슐화 ──
    agent_model_name = "gemma-4-26b-a4b"

    # completions2 위생 규칙 — 채널 닫기만으론 특정 프롬프트에서 'thought\nthought…' mode
    # collapse 가 간헐 재발하므로(QnA 경로는 시스템 프롬프트 위생으로 이를 차단), 시스템
    # 내용 맨 앞에 명시적으로 주입한다. ReAct 의 content-level 'Thought:' 추론은 막지 않고
    # native thinking 채널의 반복/필러 토큰만 금지하는 표현으로 한정.
    _AGENT_HYGIENE = (
        "[Response Rules]\n"
        "- Respond in the same language as the user's input (use '해요체' style for Korean). "
        "No greetings or self-introductions; start with the core content immediately.\n"
        "- Output Hygiene: never output internal thinking markers or chat-template tokens "
        "(e.g. <think>, <|channel>, <|turn>, <start_of_turn>, <end_of_turn>, <bos>, <eos>). "
        "Never repeat a token, word, or line (e.g. the word \"thought\" over and over). "
        "Output only the final response content in the requested format."
        # NOTE: api/ai/chats(qna_gemma.yaml) 의 [Language Rule]+[Output Hygiene] 와 정렬.
        # 단 '추론/중간단계 표시 금지'(Step 1 / 먼저 등) 규칙은 제외 — completions2 는
        # ReAct 에이전트(ExCyTIn 등)가 content 에 'Thought:/Action:' 을 출력해야 하므로.
    )

    def build_agent_prompt(self, messages: List[Dict], options: Dict = None) -> str:
        """OpenAI messages → Gemma-4 chat template (GGUF tokenizer.chat_template 기준).
        create_clean_prompt 과 동일 토큰: <|turn>{role} / <turn|> / thinking 채널 <|channel>thought.
        Gemma 는 system role 이 없어 첫 user 에 system 내용을 prepend 한다.
        시스템 맨 앞에 _AGENT_HYGIENE 를 항상 주입(시스템 메시지 유무와 무관).
        options['locale'] 가 있으면 api/ai/chats 와 동일하게 응답 언어를 강제한다."""
        options = options or {}
        sys_text = self._AGENT_HYGIENE
        locale = options.get("locale")
        if locale:
            # api/ai/chats(create_clean_prompt) 와 동일한 강제 규칙 — 다른 언어규칙 override.
            sys_text += (f"\n- [LANGUAGE RULE] You MUST respond in the language corresponding "
                         f"to locale '{locale}'. This overrides all other language rules.")
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                sys_text = (sys_text + "\n\n" + content) if sys_text else content
            elif role == "user":
                if sys_text and not parts:
                    parts.append(f"<|turn>user\n{sys_text}\n\n{content}<turn|>\n")
                    sys_text = ""
                else:
                    parts.append(f"<|turn>user\n{content}<turn|>\n")
            elif role == "assistant":
                parts.append(f"<|turn>model\n{content}<turn|>\n")
        prompt = "".join(parts)
        # model turn 을 열고 thinking 채널 개시 — create_clean_prompt(QnA 경로)와 동일.
        # thought 루프(mode collapse)는 채널 close 가 아니라 위 _AGENT_HYGIENE(시스템 프롬프트
        # 위생 규칙)로 막는다. QnA 경로가 open 채널 + 위생 규칙으로 안정 동작하는 것과 동일 구성.
        if prompt.rstrip().endswith("<turn|>") or not prompt:
            prompt += "<|turn>model\n<|channel>thought\n"
        return prompt

    def _agent_generate(self, prompt: str) -> str:
        # PII 치환/복원은 handler_base.agent_complete 가 이 호출을 감싸서 처리한다 (S-6 공통화).
        # max_tokens 4096: mode-collapse 폭주 cap. (prompt는 이미 포맷 완료)
        return self.local_model.generate_agent(prompt, 4096, 0.3, 0.95)

    def agent_stream(self, messages, options=None):
        """completions2 스트리밍(stream:true 전용, additive). build_agent_prompt ->
        local_model.generate_stream -> 앞 <channel|> thinking 프리픽스 제거 후 텍스트 조각 yield.
        pii_mode=False 전제(config). 비스트림 경로(agent_complete)는 무변경."""
        if not getattr(self, "available", False) or getattr(self, "local_model", None) is None:
            raise RuntimeError(f"{self.registry_key} 모델을 사용할 수 없습니다. (핸들러 미로드)")
        prompt = self.build_agent_prompt(messages, options or {})
        MARK = "<channel|>"
        prefix_done = False
        head = ""
        for token in self.local_model.generate_stream(prompt, 4096, 0.3, 0.95):
            if not token:
                continue
            if prefix_done:
                t = token.replace("�", "")
                if t:
                    yield t
                continue
            head += token
            idx = head.find(MARK)
            if idx >= 0:
                rest = head[idx + len(MARK):].replace("�", "")
                prefix_done = True
                if rest:
                    yield rest
            elif len(head) > 256:
                prefix_done = True
                out = head.replace("�", "")
                if out:
                    yield out
        if not prefix_done and head:
            out = head.replace("�", "")
            if out:
                yield out

    # process_question: BaseModelHandler default 사용 (generate_stream wrap).
    # process_question_error: BaseModelHandler default 사용.

    async def generate_complete_error(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> str:
        try:
            if not self.available:
                return "Gemma 모델을 사용할 수 없습니다."

            if isinstance(request, dict):
                query = request.get("question", "")
                system_prompt = request.get("system", "")
                user_prompt = request.get("user", "")
            else:
                query = str(request)
                system_prompt = ""
                user_prompt = query

            if system_prompt and user_prompt:
                formatted_prompt = f"""<start_of_turn>user
{system_prompt}


{user_prompt}
<end_of_turn>
<start_of_turn>model
"""
            else:
                formatted_prompt = f"""<start_of_turn>user
You are a helpful assistant.


{query}
<end_of_turn>
<start_of_turn>model
"""

            response = await asyncio.to_thread(
                self.local_model.generate_raw,
                prompt=formatted_prompt,
                max_tokens=4096,
                temperature=0.3,
                top_p=0.95,
                repeat_penalty=1.15,
                top_k=64,
                frequency_penalty=0.5,
                stop=[
                    "<end_of_turn>",
                    "<eos>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )

            if hasattr(self, '_usage_tracker'):
                self._usage_tracker.track(response)

            full_response = response["choices"][0]["text"].strip()

            # Gemma 응답 정리 — thinking 토큰 + 챗 템플릿 토큰 모두 제거
            final_response = _strip_gemma_tokens(full_response)

            return final_response

        except Exception as e:
            error_message = f"Gemma 에러 생성 중 오류: {str(e)}"
            logger.error(error_message)
            return error_message


    def create_workflow_from_plan(self, plan_response: str, user_guid: str) -> Dict[str, Any]:
        try:
            plan_steps = json.loads(plan_response)
            if not isinstance(plan_steps, list):
                raise ValueError("PLAN 응답이 JSON 배열이 아닙니다")

            workflow_id = f"workflow_{user_guid}_{int(time.time())}"
            workflow = {
                "workflow_id": workflow_id,
                "user_guid": user_guid,
                "created_at": time.time(),
                "status": "pending",
                "current_step": 0,
                "total_steps": len(plan_steps),
                "steps": [
                    {
                        "step_id": i,
                        "msg": step.get("msg", ""),
                        "category": step.get("category", "action"),
                        "status": "pending",
                        "result": None,
                        "error": None,
                        "executed_at": None,
                        "execution_time": None
                    }
                    for i, step in enumerate(plan_steps)
                ],
                "results": [],
                "next_action": "execute_next"
            }

            if not hasattr(self, 'workflows'):
                self.workflows = {}
            self.workflows[workflow_id] = workflow

            print(f"🔧 워크플로우 생성 완료: {workflow_id}")
            print(f"   - 단계 수: {len(plan_steps)}")
            print(f"   - 첫 번째 단계: {plan_steps[0].get('msg', '')[:50]}...")

            return workflow

        except json.JSONDecodeError as e:
            logger.error(f"PLAN 응답 JSON 파싱 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"워크플로우 생성 중 오류: {e}")
            return None

    def get_next_workflow_step(self, workflow_id: str) -> Optional[Dict[str, Any]]:


        try:
            if not hasattr(self, 'workflows') or workflow_id not in self.workflows:
                logger.error(f"워크플로우를 찾을 수 없음: {workflow_id}")
                return None

            workflow = self.workflows[workflow_id]
            current_step = workflow["current_step"]

            if current_step >= workflow["total_steps"]:
                workflow["status"] = "completed"
                workflow["next_action"] = "complete"
                print(f"✅ 워크플로우 완료: {workflow_id}")
                return None

            next_step = workflow["steps"][current_step]
            print(f"🎯 다음 단계 준비: {current_step + 1}/{workflow['total_steps']}")
            print(f"   - 메시지: {next_step['msg'][:50]}...")

            return {
                "workflow_id": workflow_id,
                "step": next_step,
                "step_index": current_step,
                "total_steps": workflow["total_steps"]
            }

        except Exception as e:
            logger.error(f"다음 워크플로우 단계 가져오기 실패: {e}")
            return None

    def execute_workflow_step(self, workflow_id: str, step_result: Dict[str, Any]) -> Dict[str, Any]:


        try:
            if not hasattr(self, 'workflows') or workflow_id not in self.workflows:
                logger.error(f"워크플로우를 찾을 수 없음: {workflow_id}")
                return {"action": "error", "message": "워크플로우를 찾을 수 없음"}

            workflow = self.workflows[workflow_id]
            current_step_index = workflow["current_step"]
            current_step = workflow["steps"][current_step_index]

            current_step["status"] = "completed"
            current_step["result"] = step_result
            current_step["executed_at"] = time.time()

            should_continue = self._analyze_step_result(step_result)

            if should_continue:
                workflow["current_step"] += 1

                if workflow["current_step"] >= workflow["total_steps"]:
                    workflow["status"] = "completed"
                    workflow["next_action"] = "complete"
                    print(f"🎉 전체 워크플로우 완료: {workflow_id}")
                    return {"action": "complete", "workflow": workflow}
                else:
                    workflow["next_action"] = "execute_next"
                    next_step_info = self.get_next_workflow_step(workflow_id)
                    print(f"➡️ 다음 단계로 진행: {workflow['current_step']}/{workflow['total_steps']}")
                    return {"action": "continue", "next_step": next_step_info}
            else:
                workflow["status"] = "requires_alternative"
                workflow["next_action"] = "wait_user"

                print(f"⚠️ 단계 실행 결과 부족, 대안 필요: {workflow_id}")
                return {
                    "action": "requires_alternative",
                    "current_result": step_result,
                    "suggestion": "다른 기간이나 조건으로 재시도 필요"
                }

        except Exception as e:
            logger.error(f"워크플로우 단계 실행 처리 실패: {e}")
            return {"action": "error", "message": str(e)}

    def _analyze_step_result(self, step_result: Dict[str, Any]) -> bool:


        try:
            if isinstance(step_result, dict):
                answer_text = step_result.get("answer_text", "")

                no_result_indicators = [
                    "조회 결과가 없습니다",
                    "결과가 없습니다",
                    "데이터가 없습니다",
                    "No results found",
                    "Empty result"
                ]

                for indicator in no_result_indicators:
                    if indicator in answer_text:
                        print(f"🚫 결과 없음 감지: {indicator}")
                        return False

                if any(keyword in answer_text.lower() for keyword in ["bytes", "count", "traffic", "app", "source"]):
                    print(f"✅ 유효한 결과 감지")
                    return True

            print(f"🤔 결과 분석 불명확, 기본적으로 진행")
            return True

        except Exception as e:
            logger.error(f"단계 결과 분석 실패: {e}")
            return True

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:


        try:
            if not hasattr(self, 'workflows') or workflow_id not in self.workflows:
                return None

            workflow = self.workflows[workflow_id]

            completed_steps = sum(1 for step in workflow["steps"] if step["status"] == "completed")
            progress = (completed_steps / workflow["total_steps"]) * 100 if workflow["total_steps"] > 0 else 0

            return {
                "workflow_id": workflow_id,
                "status": workflow["status"],
                "progress": f"{progress:.1f}%",
                "current_step": workflow["current_step"],
                "total_steps": workflow["total_steps"],
                "completed_steps": completed_steps,
                "next_action": workflow["next_action"],
                "created_at": workflow["created_at"]
            }

        except Exception as e:
            logger.error(f"워크플로우 상태 조회 실패: {e}")
            return None

    def _handle_plan_workflow(self, plan_response: str, request: Dict[str, Any], user_id: str) -> Optional[str]:


        try:
            user_guid = request.get("user_guid", user_id or "unknown")
            workflow = self.create_workflow_from_plan(plan_response, user_guid)

            if not workflow:
                print(f"❌ 워크플로우 생성 실패")
                return None

            first_step_info = self.get_next_workflow_step(workflow["workflow_id"])

            if not first_step_info:
                print(f"❌ 첫 번째 단계를 찾을 수 없음")
                return None

            first_step = first_step_info["step"]
            step_msg = first_step["msg"]

            print(f"🚀 첫 번째 단계 실행: {step_msg[:50]}...")

            action_request = {
                "type": "ACTION",
                "question": step_msg,
                "user_guid": user_guid,
                "workflow_id": workflow["workflow_id"],
                "step_index": first_step_info["step_index"]
            }


            workflow_response = {
                "workflow_created": True,
                "workflow_id": workflow["workflow_id"],
                "total_steps": workflow["total_steps"],
                "first_step": {
                    "step_index": first_step_info["step_index"],
                    "msg": step_msg,
                    "category": first_step.get("category", "action")
                },
                "next_action": "execute_first_step"
            }

            print(f"📋 워크플로우 정보:")
            print(f"   - ID: {workflow['workflow_id']}")
            print(f"   - 총 단계: {workflow['total_steps']}")
            print(f"   - 첫 단계: {step_msg[:30]}...")

            return json.dumps(workflow_response, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"PLAN 워크플로우 처리 실패: {e}")
            print(f"❌ PLAN 워크플로우 처리 중 오류: {e}")
            return None

    async def execute_workflow_first_step(self, workflow_id: str) -> Optional[Dict[str, Any]]:


        try:
            first_step_info = self.get_next_workflow_step(workflow_id)

            if not first_step_info:
                return None

            first_step = first_step_info["step"]
            step_msg = first_step["msg"]

            print(f"⚡ 워크플로우 첫 단계 실행: {step_msg}")

            action_request = {
                "type": "ACTION",
                "question": step_msg,
                "workflow_context": True
            }


            return {
                "step_executed": True,
                "workflow_id": workflow_id,
                "step_index": first_step_info["step_index"],
                "step_msg": step_msg,
                "status": "executed"
            }

        except Exception as e:
            logger.error(f"워크플로우 첫 단계 실행 실패: {e}")
            return None

    async def get_complete_answer(self, request, user_id=None, channel_id=None, source='API', thread_ts=None):
        try:
            result = await self.generate_complete(request, user_id, channel_id, source, thread_ts)

            return {
                "answer": result.get("answer", ""),
                "raw_answer": result.get("raw_answer", ""),
                "sources": result.get("sources", []),
                "intent": result.get("intent", "unknown")
            }

        except Exception as e:
            logger.error(f"Gemma get_complete_answer [{type(e).__name__}]: {e}")

            return {
                "answer": classify_error(e),
                "raw_answer": "",
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
            print(f"Gemma 검색 상태 확인 오류: {str(e)}")
            return {
                "rag_system_available": False,
                "knowledge_graph_enabled": self.use_kg,
                "search_config": {"search_strategy": "unknown", "cache_enabled": False},
                "total_documents": 0,
                "cache_size": 0
            }

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
        print(f"🔥 [DEBUG Gemma] _apply_response_format called with response_format: {response_format}")

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

            text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

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

    def cleanup_resources(self):
        try:
            if hasattr(self, 'chat_session'):
                self.chat_session.clear()

            if hasattr(self, 'local_model') and self.local_model:
                self.local_model = None

            logger.info("✅ Gemma 핸들러 리소스 정리 완료")

        except Exception as e:
            logger.warning(f"⚠️ Gemma 핸들러 리소스 정리 중 오류: {e}")

    def __del__(self):
        try:
            self.cleanup_resources()
        except:
            pass

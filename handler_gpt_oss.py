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

_global_server_manager = None
_global_translation_server_manager = None


def get_server_manager():
    """qa_llm.py shutdown 시 접근용"""
    return _global_server_manager


def get_translation_server_manager():
    """qa_llm.py shutdown 시 접근용"""
    return _global_translation_server_manager


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
            # KV 캐시 재사용 — 레거시 기본은 미전달(꺼짐/서버 기본). config cache_reuse 설정 시에만 활성
            # (gemma spawn과의 비대칭 승계 — T2 A/B 후 기본 승격 검토, docs/kv-cache-reuse.md)
            _cr = cfg.get('cache_reuse')
            _sps = cfg.get('slot_prompt_similarity')
            if _cr:
                cmd += ['--cache-reuse', str(_cr)]
            if _sps is not None:
                cmd += ['--slot-prompt-similarity', str(_sps)]
            # 추가 서버 인자 — 카트리지/설치기가 정의 (예: --cpu-moe, --flash-attn)
            if cfg.get('extra_args'):
                cmd += shlex.split(cfg['extra_args'])

            logger.info(f"llama-server 시작: {' '.join(cmd)}")
            # DEVNULL이면 OOM/alloc 실패 원인이 어디에도 안 남는다 — 파일로 보존 (verification-log #11)
            os.makedirs('logs', exist_ok=True)
            _srv_log = open(os.path.join('logs', 'llama-server-gpt-oss.log'), 'ab')
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

    LocalGPTOSSModel.llm 을 이 객체로 교체하면 기존 코드가 그대로 동작한다.
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
                          stop: list = None, echo: bool = False, stream: bool = False, **kwargs) -> Any:
        self._mgr.ensure_running()

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
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

class LocalGPTOSSModel:
    def __init__(self, llm, server_manager: LlamaServerManager = None):
        self._server_manager = server_manager
        self.llm = llm
        # llama-cpp-python 모드에서는 동시 접근 방지용 Lock
        self._lock = threading.Lock() if server_manager is None else None
        logger.info(f"✅ LocalGPTOSSModel 초기화 완료 ({'llama-server' if server_manager else 'llama-cpp-python'} 모드)")

    @classmethod
    def create_with_config(cls) -> 'LocalGPTOSSModel':
        global _global_server_manager

        config_manager = ConfigManager()
        paths_config = config_manager.get_paths_config()
        server_config = config_manager.get_llama_server_config()

        is_external = bool(server_config.get('external_server', False))

        model_path = str(Path(paths_config.get('gpt_oss_model_path', '')).resolve())
        # external_server 면 모델 파일은 외부 컨테이너가 보유 — 앱 측 존재 체크 skip
        if not is_external and not os.path.exists(model_path):
            raise FileNotFoundError(f"GPT-OSS 모델 파일을 찾을 수 없습니다: {model_path}")

        if server_config.get('use_server_mode', False):
            # llama-server HTTP 모드
            if not is_external and not os.path.exists(server_config['binary_path']):
                raise FileNotFoundError(f"llama-server 바이너리를 찾을 수 없습니다: {server_config['binary_path']}")

            mgr = LlamaServerManager(server_config, model_path)
            mgr.start()
            _global_server_manager = mgr
            return cls(_LlamaServerShim(mgr), server_manager=mgr)
        else:
            # 기존 llama-cpp-python 직접 로드 모드
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python 라이브러리가 필요합니다.")

            context_size = int(os.getenv("LLAMA_CONTEXT_SIZE", "32768"))
            logger.info(f"🚀 로컬 GPT-OSS 모델 로드 중: {model_path} (컨텍스트={context_size})")

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

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.01, top_p: float = 0.9) -> Dict:
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.1,
            top_k=40,
            stop=[
                "<|end|>",
                "<|return|>",
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
        # llama-cpp-python 모드에서는 Lock으로 직렬화
        if self._lock:
            self._lock.acquire()
        try:
            yield from self._generate_stream_inner(prompt, max_tokens, temperature, top_p, repeat_penalty, top_k)
        finally:
            if self._lock:
                self._lock.release()

    def _generate_stream_inner(self, prompt, max_tokens, temperature, top_p, repeat_penalty, top_k):
        buffer = ""
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stop=[
                "<|end|>",
                "<|return|>",
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
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.1,
                top_k=40,
                stop=[
                    "<|end|>",
                    "<|return|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )
            result = response["choices"][0]["text"].strip()

            # Harmony 태그 정리
            unwanted_tokens = ["<|end|>", "<|return|>", "<|endoftext|>", "<|message|>"]
            for token in unwanted_tokens:
                result = result.replace(token, "")

            return result.strip()

        except Exception as e:
            logger.error(f"GPT-OSS 생성 오류: {e}")
            return ""


    def generate_agent(self, prompt: str, max_tokens: int = 131072, temperature: float = 0.01,
                           top_p: float = 0.9) -> str:
        """Agent용 응답 생성. prompt는 호출측(build_agent_prompt)이 이미 Harmony 포맷 완료."""
        try:
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.1,
                top_k=40,
                stop=[
                    "<|end|>",
                    "<|return|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )

            result = response["choices"][0]["text"].strip()

            # Harmony 태그 정리
            unwanted_tokens = ["<|endoftext|>", "<|end|>", "<|return|>", "<|message|>"]
            for token in unwanted_tokens:
                result = result.replace(token, "")

            return result.strip()

        except Exception as e:
            logger.error(f"GPT-OSS Agent 생성 오류: {e}")
            return ""


class GPTOSSHandler(BaseModelHandler):
    registry_key = "gpt-oss"  # handler_registry.HANDLER_CLASSES 키와 일치
    is_local = True            # 로컬 GGUF / llama-server

    def __init__(self, rag_system=None, use_kg: bool = True, intent_analyzer=None):
        super().__init__(rag_system, use_kg, intent_analyzer)

        self.config_manager = ConfigManager()

        self.handler_type = "gpt-oss"
        # 실제 서빙 GGUF 파일명에서 라벨 파생 (gemma 핸들러와 동일 규칙) — 프리셋과 라벨 불일치 방지
        try:
            _gguf_stem = Path(self.config_manager.get_paths_config().get('gpt_oss_model_path', '')).stem
        except Exception:
            _gguf_stem = ''
        self.model_name = _gguf_stem or 'gpt-oss-20b'
        self.agent_model_name = self.model_name
        self.local_model = None
        self.available = False

        prompt_path = self.config_manager.get_prompt_config().get('gpt_oss')
        if not prompt_path:
            prompt_path = self.config_manager.get_prompt_config().get('gpt')

        self.system_prompt = self.load_prompt_from_file(prompt_path)

        self.logger = ChatLogger(self.model_name)

        self._initialize_model()

        self.translation_llama = None
        self._init_translation_llama()

    def _init_translation_llama(self):
        """번역 모델 선택 — config 의 [llama_server_translation] use_server_mode 로 토글.

        - True: 별도 llama-server (llama-3.1 8B 등) 띄워서 번역. 기존 동작.
        - False: self.local_model (GPT-OSS) 을 번역에도 재사용 — 별도 모델 미사용.
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

        # 모드 B: self.local_model (GPT-OSS) 재사용 (Harmony chat template)
        if not self.available or self.local_model is None:
            logger.warning("⚠️ GPT-OSS local_model 미로드 — 번역도 사전 fallback 사용")
            self.translation_llama = None
            return

        class _GPTOSSTranslationWrapper:
            def __init__(self, gptoss_local):
                self._model = gptoss_local

            def generate_complete(self, prompt, max_tokens=200, temperature=0.1, **kwargs):
                kwargs.setdefault('stop', ['<|end|>', '<|return|>', '<|endoftext|>'])
                r = self._model.generate_raw(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return r["choices"][0]["text"].strip()

        self.translation_llama = _GPTOSSTranslationWrapper(self.local_model)
        logger.info("✅ 번역도 응답 모델(GPT-OSS)로 처리 — 별도 모델 미사용 (use_server_mode=False)")

    def _initialize_model(self):
        try:
            self.local_model = LocalGPTOSSModel.create_with_config()
            self.available = True
            mode = "llama-server" if self.local_model._server_manager else "llama-cpp-python"
            logger.info(f"✅ 로컬 GPT-OSS 모델 초기화 완료 ({mode})")
        except Exception as e:
            logger.error(f"❌ GPT-OSS 모델 초기화 실패: {str(e)}")
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
                # self.local_model 재사용 (GPT-OSS Harmony chat template)
                prompt = f"<|start|>system\n{system_instructions}\n<|end|>\n<|start|>user\n{query}\n<|end|>\n<|start|>assistant<|channel|>final<|message|>"

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
                "answer": "GPT-OSS 모델을 사용할 수 없습니다.",
                "intent": None,
                "sources": [],
                "response_time": 0,
                "question_type": question_type
            }

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
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('action_gpt'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'PLAN':
                print(f"🔧 [PLAN GPT-OSS] 일반 계획 수립 프롬프트 사용")
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('plan_gpt'))
                print(f"✅ [PLAN GPT-OSS] 일반 계획 프롬프트 로드 완료")

                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'PLAYBOOK':
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('playbook'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'TITLE':
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('title'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            else:
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('qna_gpt'))
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

        """전체 GPT-OSS 메시지 구조 로깅 (generate_complete)"""
        import time as time_module
        import json
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/gpt-oss/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write(f"Method: generate_complete\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE GPT-OSS PROMPT STRUCTURE\n")
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
                        f.write("action_gpt_oss")
                    elif intent_upper == 'PLAN':
                        f.write("plan_gpt_oss")
                    else:
                        f.write("qna_gpt_oss")
                else:
                    f.write("default gpt_oss")
                f.write("\n")

                f.write(f"RAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")

                f.write(f"Related Concepts: {len(related_concepts)}\n")
                for k, concept in enumerate(related_concepts[:5]):
                    concept_name = concept.get('concept', 'Unknown') if isinstance(concept, dict) else str(concept)
                    f.write(f"  {k+1}. {concept_name}\n")

        except Exception as log_e:
            print(f"⚠️ GPT-OSS 로그 저장 실패 (generate_complete): {log_e}")

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
            print(f"🚀 GPT-OSS 응답 생성 시작 (generate_complete)...")
            response = await asyncio.to_thread(
                self.local_model.generate_raw,
                prompt=formatted_prompt,
                max_tokens=131072,
                temperature=0.1,
                top_p=0.8,
                repeat_penalty=1.2,
                top_k=20,
                stop=[
                    "<|end|>",
                    "<|return|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )
            end_model_time = time.time()
            self._usage_tracker.track(response)
            print(f"✅ GPT-OSS 응답 생성 완료 {end_model_time - start_model_time:.2f}초 (generate_complete)")

            full_response = response["choices"][0]["text"].strip()
            print(f"📝 전체 응답 (generate_complete): {full_response}")

            # Clarify 응답 처리 (ACTION 타입에서 추가 정보 요청시)
            if (request.get("type", "").upper() == "ACTION" and
                ('"action": "clarify"' in full_response or '"action":"clarify"' in full_response)):
                original_intent = request.get("type", "ACTION")
                print(f"🔄 Clarify 응답 감지 (GPT-OSS): {original_intent} 의도 유지")

                try:
                    clarify_data = json.loads(full_response)
                    if isinstance(clarify_data, dict) and "message" in clarify_data:
                        full_response = clarify_data["message"]
                        print(f"📝 Clarify 메시지 추출 (GPT-OSS): {full_response}")
                except json.JSONDecodeError:
                    match = re.search(r'"message":\s*"([^"]*)"', full_response)
                    if match:
                        full_response = match.group(1)
                        print(f"📝 Clarify 메시지 추출 (정규식, GPT-OSS): {full_response}")

            # Harmony 태그 정리
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
            is_action = (request_type == "ACTION" or selected_intent_upper == "ACTION")


            if is_post_action:
                print(f"📝 POST_ACTION 감지: 쿼리 결과 분석 모드")
                print(f"🔧 일반 POST-ACTION: 전체 응답 사용")

            processed_response = self._fix_unicode_response(final_response)

            show_metadata_config = self.config_manager.get_option_config().get('show_metadata')
            print(f"🔍 GPT-OSS 메타데이터 조건 체크: request_type='{request_type}', selected_intent='{selected_intent_upper}', show_metadata={show_metadata_config}")

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
                print(f"🔒 [GPT-OSS] PLAN 응답 생성 완료 - 워크플로우는 aibot_llm_module.py에서 처리됨")

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

            logger.error(f"GPT-OSS generate_complete [{type(e).__name__}]: {e}")
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
            yield "GPT-OSS 모델을 사용할 수 없습니다."
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
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('action_gpt'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'PLAN':
                print(f"🔧 [PLAN GPT-OSS] 일반 계획 수립 프롬프트 사용")
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('plan_gpt'))
                print(f"✅ [PLAN GPT-OSS] 일반 계획 프롬프트 로드 완료")

                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            elif intent_upper == 'TITLE':
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('title'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)
            else:
                gpt_prompt = self.load_prompt_from_file(self.config_manager.get_prompt_config().get('qna_gpt'))
                self.system_prompt = self._clean_prompt_content(gpt_prompt)

        # locale 지시문 주입
        locale = request.get("locale", "") if isinstance(request, dict) else ""
        if locale:
            self.system_prompt += f"\n\n[LANGUAGE RULE]\nYou MUST respond in the language corresponding to locale '{locale}'. This overrides all other language rules."

        chat_session.add_message("user", query)

        formatted_prompt = self.create_clean_prompt(query, context, "", selected_intent.upper() if selected_intent else "", response_format, question_type, request)

        """전체 GPT-OSS 메시지 구조 로깅"""
        import time as time_module
        import json
        timestamp = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime())
        filename = f"./logs/full_messages/gpt-oss/prompt_contents_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Selected Intent: {selected_intent}\n")
                f.write(f"Question Type: {question_type}\n")
                f.write("=" * 80 + "\n")
                f.write("COMPLETE GPT-OSS PROMPT STRUCTURE\n")
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
                        f.write("action_gpt_oss")
                    elif intent_upper == 'PLAN':
                        f.write("plan_gpt_oss")
                    else:
                        f.write("qna_gpt_oss")
                else:
                    f.write("default gpt_oss")
                f.write("\n")

                f.write(f"RAG Sources: {len(source_files)}\n")
                for j, source in enumerate(source_files):
                    f.write(f"  {j+1}. {source}\n")

                f.write(f"Related Concepts: {len(related_concepts)}\n")
                for k, concept in enumerate(related_concepts[:5]):
                    concept_name = concept.get('concept', 'Unknown') if isinstance(concept, dict) else str(concept)
                    f.write(f"  {k+1}. {concept_name}\n")

        except Exception as log_e:
            print(f"⚠️ GPT-OSS 로그 저장 실패: {log_e}")

        try:
            full_response = ""
            message_started = False

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

                    if not message_started:
                        # final 채널의 <|message|>가 나올 때까지 대기
                        if "<|channel|>final<|message|>" in full_response:
                            message_pos = full_response.rfind("<|channel|>final<|message|>")
                            content_after_tag = full_response[message_pos + len("<|channel|>final<|message|>"):]
                            if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                                content_after_tag = self._remove_markdown_formatting(content_after_tag)
                            yield content_after_tag
                            message_started = True
                    else:
                        if question_type and question_type.upper() in ["QNA", "POST-ACTION"] and response_format is False:
                            processed_chunk = self._remove_markdown_formatting(chunk)
                            yield processed_chunk
                        else:
                            yield chunk

            chat_session.add_message("assistant", full_response)

            request_type_stream = request.get("type", "").upper()
            question_type_upper = question_type.upper() if question_type else ""
            show_metadata_stream = self.config_manager.get_option_config().get('show_metadata')
            print(f"🔍 GPT-OSS 스트리밍 메타데이터 조건: request_type='{request_type_stream}', question_type='{question_type_upper}', show_metadata={show_metadata_stream}")

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

            error_message = f"GPT-OSS 스트리밍 생성 중 오류: {str(e)}"
            logger.error(error_message)
            yield error_message

    def _get_clean_message(self, response: str) -> str:
        """Harmony 태그에서 순수 응답 추출 (final 채널 우선)"""

        # final 채널의 <|message|> 태그 찾기
        if "<|channel|>final<|message|>" in response:
            final_pos = response.rfind("<|channel|>final<|message|>")
            content = response[final_pos + len("<|channel|>final<|message|>"):].strip()

            # Harmony 종료 태그로 자르기
            for end_tag in ["<|end|>", "<|return|>", "<|endoftext|>"]:
                if end_tag in content:
                    content = content.split(end_tag)[0]

            content = content.strip()
            print(f"✅ final 채널 메시지 추출: '{content[:50]}...'")
            return content

        # 기존 방식 (assistant 채널) - 하위 호환성
        elif "<|message|>" in response:
            last_message_pos = response.rfind("<|message|>")
            content = response[last_message_pos + len("<|message|>"):].strip()

            # Harmony 종료 태그로 자르기
            for end_tag in ["<|end|>", "<|return|>", "<|endoftext|>"]:
                if end_tag in content:
                    content = content.split(end_tag)[0]

            content = content.strip()
            print(f"✅ <|message|> 태그 발견! 추출: '{content[:50]}...'")
            return content

        # 태그 없으면 Harmony 태그들 정리
        clean = response
        harmony_tags = ["<|start|>", "<|end|>", "<|return|>", "<|message|>",
                       "<|channel|>", "<|endoftext|>", "analysis", "final"]
        for tag in harmony_tags:
            clean = clean.replace(tag, "")

        return clean.strip()

    # _clean_prompt_content: BaseModelHandler default 사용 (superset 토큰 strip).
    # 기존 unwanted_tags 는 base 의 UNWANTED_PROMPT_TAGS (12개) 의 부분집합이라
    # 동일 결과 보장됨.

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
            prompt = f"""<|start|>system
                {system_content}<|end|>
                <|start|>user
                {user_content}<|end|>
                <|start|>assistant<|channel|>analysis<|message|><|end|>
                <|start|>assistant<|channel|>final<|message|>"""
            return prompt

        system_content += "\n\n[RESPONSE STYLE]"
        system_content += "\nDo not show your thinking process or intermediate steps."
        system_content += "\nDo not use phrases like 'Let me think...', 'First, I need to...', 'Step 1:', etc."
        system_content += "\nProvide the final answer directly without showing how you arrived at it."
        system_content += "\nKeep the answer comprehensive but skip the reasoning chain."

        system_content += f"\n\n[CURRENT DATE & CALCULATION GUIDE]"
        system_content += f"\n📅 Current date: {current_date} ({current_readable})"
        system_content += f"\n"
        system_content += f"\n⚠️ IMPORTANT: When user asks for 'recent X days' or 'last X days', calculate BACKWARDS from today:"
        system_content += f"\n- Recent 7 days = {(now - timedelta(days=6)).strftime('%Y%m%d')} to {current_date}"
        system_content += f"\n- Recent 30 days = {(now - timedelta(days=29)).strftime('%Y%m%d')} to {current_date}"
        system_content += f"\n- Yesterday = {(now - timedelta(days=1)).strftime('%Y%m%d')}"
        system_content += f"\n- Last week = {(now - timedelta(days=6)).strftime('%Y%m%d')} to {current_date}"
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

        if context.strip():
            user_content += f"\n\n관련 문서:\n{context}"

        if chat_history.strip():
            user_content += f"\n\n이전 대화:\n{chat_history}"

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

        prompt = f"""<|start|>system
                {system_content}<|end|>
                <|start|>user
                {user_content}<|end|>
                <|start|>assistant<|channel|>analysis<|message|><|end|>
                <|start|>assistant<|channel|>final<|message|>"""

        return prompt

    # ── Agent (completions2) — gpt-oss Harmony 포맷 캡슐화 ──
    agent_model_name = "gpt-oss-20b"

    def build_agent_prompt(self, messages: List[Dict], options: Dict = None) -> str:
        """OpenAI messages → Harmony 포맷 프롬프트 (gpt-oss 전용).
        options['locale'] 가 있으면 응답 언어를 강제(api/ai/chats 와 동일 의도)."""
        options = options or {}
        locale = options.get("locale")
        locale_rule = (f"\nYou MUST respond in the language corresponding to locale '{locale}'. "
                       f"This overrides all other language rules." if locale else "")
        prompt = ""
        has_system = False
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                if not has_system:
                    prompt += f"<|start|>system\n{content}{locale_rule}<|end|>\n"
                    has_system = True
                else:
                    prompt = prompt.replace("<|end|>\n", f"\n{content}<|end|>\n", 1)
            elif role == "user":
                prompt += f"<|start|>user\n{content}<|end|>\n"
            elif role == "assistant":
                prompt += (f"<|start|>assistant<|channel|>analysis<|message|><|end|>\n"
                           f"<|start|>assistant<|channel|>final<|message|>{content}")
        if prompt.rstrip().endswith("<|end|>"):
            prompt += ("<|start|>assistant<|channel|>analysis<|message|><|end|>\n"
                       "<|start|>assistant<|channel|>final<|message|>")
        return prompt

    def _agent_generate(self, prompt: str) -> str:
        return self.local_model.generate_agent(prompt)

    # process_question: BaseModelHandler default 사용 (generate_stream wrap).
    # process_question_error: BaseModelHandler default 사용 (generate_complete_error
    # 보유 시 {"answer": ..., "context_sources": []} wrap).

    async def generate_complete_error(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> str:
        try:
            if not self.available:
                return "GPT-OSS 모델을 사용할 수 없습니다."

            if isinstance(request, dict):
                query = request.get("question", "")
                system_prompt = request.get("system", "")
                user_prompt = request.get("user", "")
            else:
                query = str(request)
                system_prompt = ""
                user_prompt = query

            if system_prompt and user_prompt:
                formatted_prompt = f"""<|start|>system
{system_prompt}
<|end|>
<|start|>user
{user_prompt}
<|end|>
<|start|>assistant<|channel|>analysis<|message|><|end|>
<|start|>assistant<|channel|>final<|message|>"""
            else:
                formatted_prompt = f"""<|start|>system
You are a helpful assistant.
<|end|>
<|start|>user
{query}
<|end|>
<|start|>assistant<|channel|>analysis<|message|><|end|>
<|start|>assistant<|channel|>final<|message|>"""

            response = await asyncio.to_thread(
                self.local_model.generate_raw,
                prompt=formatted_prompt,
                max_tokens=4096,
                temperature=0.1,
                stop=[
                    "<|end|>",
                    "<|return|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False
            )

            if hasattr(self, '_usage_tracker'):
                self._usage_tracker.track(response)

            full_response = response["choices"][0]["text"].strip()

            final_response = full_response

            # Harmony 태그가 응답에 포함된 경우 추출
            if "<|message|>" in full_response:
                message_start = full_response.rfind("<|message|>") + len("<|message|>")
                final_response = full_response[message_start:].strip()
                print(f"✅ message 태그 발견! 추출 완료")

            # Harmony 종료 태그 정리
            unwanted_tokens = ["<|end|>", "<|return|>", "<|endoftext|>", "<|message|>"]
            for token in unwanted_tokens:
                final_response = final_response.replace(token, "")

            return final_response.strip()

        except Exception as e:
            error_message = f"GPT-OSS 에러 생성 중 오류: {str(e)}"
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
                "sources": result.get("sources", []),
                "intent": result.get("intent", "unknown")
            }

        except Exception as e:
            logger.error(f"GPT-OSS get_complete_answer [{type(e).__name__}]: {e}")

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
            print(f"GPT-OSS 검색 상태 확인 오류: {str(e)}")
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
        print(f"🔥 [DEBUG GPT-OSS] _apply_response_format called with response_format: {response_format}")

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

            logger.info("✅ GPT-OSS 핸들러 리소스 정리 완료")

        except Exception as e:
            logger.warning(f"⚠️ GPT-OSS 핸들러 리소스 정리 중 오류: {e}")

    def __del__(self):
        try:
            self.cleanup_resources()
        except:
            pass

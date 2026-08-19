import os
from pathlib import Path
import configparser
from typing import Dict, Any, Optional
import base64
import json
import hashlib
import hmac
import secrets

class ConfigManager:
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()

    def load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ 설정 파일을 찾을 수 없습니다: {self.config_path}")

        self.config.read(self.config_path, encoding='utf-8')

        required_keys = {
            'openai': ['api_key'],
            'bedrock': ['aws_region'],
            'paths': ['rfp_docs_dir', 'rfp_embedding_dir',
                    'aibot_docs_dir', 'system_docs_dir','aibot_embedding_dir', 'system_embedding_dir',
                    'prompt_dir', 'api_keys_dir', 'local_llama_model_path','gpt_oss_model_path','qwen_model_path','gemma_model_path'],
            'model': ['model', 'model_name', 'temperature', 'max_tokens', 'chunk_size', 'chunk_overlap'],
            'prompts': ['gpt', 'gpt_oss', 'gemma', 'llama', 'qwen', 'qna_cve', 'intent', 'action', 'plan', 'playbook', 'qna_gpt', 'plan_gpt','action_gpt','focus_query','post_action','ticket_memo'],
            'search': ['search_strategy', 'max_parallel_searches', 'cache_enabled'],
            'search_weights': ['filename_question', 'exact_keyword', 'semantic_vector'],
            'option': ['use_hnsw', 'focus', 'response', 'translation','user_intent_prompt','show_metadata'],
            'server': ['port']
        }

        for section, keys in required_keys.items():
            if section not in self.config:
                raise ValueError(f"❌ 설정 파일에 필수 섹션이 없습니다: [{section}]")

            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"❌ 설정 파일의 [{section}] 섹션에서 필수 키 '{key}'가 누락되었습니다.")

        self.validate_search_config()

    def validate_search_config(self) -> None:
        if 'search' in self.config:
            strategy = self.config.get('search', 'search_strategy', fallback='basic')
            valid_strategies = ['basic', 'hybrid']
            if strategy not in valid_strategies:
                print(f"⚠️ 잘못된 search_strategy 값: {strategy}. 기본값 'basic' 사용")

        if 'search_weights' in self.config:
            weights = self.get_search_weights_config()
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.1:
                print(f"⚠️ 검색 가중치 합계가 1.0이 아닙니다: {total_weight:.2f}")

    def get_bedrock_config(self) -> Dict[str, str]:
        # aws_access_key_id/secret/session_token 은 선택 — 셋 다 비면 boto3 default chain
        # (env / ~/.aws / IAM role) 사용. STS 임시 토큰 운영 시엔 이 세 키를 채워 넣는다.
        return {
            'aws_region': self.config.get('bedrock', 'aws_region'),
            'aws_access_key_id': self.config.get('bedrock', 'aws_access_key_id', fallback='').strip(),
            'aws_secret_access_key': self.config.get('bedrock', 'aws_secret_access_key', fallback='').strip(),
            'aws_session_token': self.config.get('bedrock', 'aws_session_token', fallback='').strip(),
            'claude_model_id': self.config.get('bedrock', 'claude_model_id', fallback='').strip(),
            'claude_translate_model_id': self.config.get('bedrock', 'claude_translate_model_id', fallback='').strip(),
            'claude_max_tokens': self.config.getint('bedrock', 'claude_max_tokens', fallback=4096),
            'claude_temperature': self.config.getfloat('bedrock', 'claude_temperature', fallback=0.01),
        }

    def get_openai_config(self) -> Dict[str, str]:
        return {
            'api_key': self.config.get('openai', 'api_key'),
            # base_url 비우면 OpenAI 기본 endpoint, 채우면 OpenAI 호환 백엔드(예: Furiosa 평가 endpoint) 로 라우팅
            'base_url': self.config.get('openai', 'base_url', fallback=''),
        }

    def get_paths_config(self) -> Dict[str, Path]:
        return {
            'rfp_docs_dir': Path(self.config.get('paths', 'rfp_docs_dir')).resolve(),
            'rfp_embedding_dir': Path(self.config.get('paths', 'rfp_embedding_dir')).resolve(),
            'aibot_docs_dir': Path(self.config.get('paths', 'aibot_docs_dir')).resolve(),
            'system_docs_dir': Path(self.config.get('paths', 'system_docs_dir')).resolve(),
            'aibot_embedding_dir': Path(self.config.get('paths', 'aibot_embedding_dir')).resolve(),
            'system_embedding_dir': Path(self.config.get('paths', 'system_embedding_dir')).resolve(),
            'prompt_dir': Path(self.config.get('paths', 'prompt_dir')).resolve(),
            'api_keys_dir': Path(self.config.get('paths', 'api_keys_dir')).resolve(),
            'local_llama_model_path': Path(self.config.get('paths', 'local_llama_model_path')).resolve(),
            'gpt_oss_model_path': Path(self.config.get('paths', 'gpt_oss_model_path')).resolve(),
            'qwen_model_path': Path(self.config.get('paths', 'qwen_model_path')).resolve(),
            'gemma_model_path': Path(self.config.get('paths', 'gemma_model_path')).resolve()
        }

    def get_model_config(self) -> Dict[str, Any]:
        return {
            'model': self.config.get('model', 'model', fallback='llama'),
            'model_name': self.config.get('model', 'model_name', fallback='gpt-4o'),
            'temperature': self.config.getfloat('model', 'temperature', fallback=0.1),
            'max_tokens': self.config.getint('model', 'max_tokens', fallback=1024),
            'chunk_size': self.config.getint('model', 'chunk_size', fallback=1000),
            'chunk_overlap': self.config.getint('model', 'chunk_overlap', fallback=200)
        }

    def get_prompt_config(self) -> Dict[str, str]:
        return {
            'gpt': self.config.get('prompts', 'gpt'),
            'gpt_oss': self.config.get('prompts', 'gpt_oss'),
            'gemma': self.config.get('prompts', 'gemma'),
            'llama': self.config.get('prompts', 'llama'),
            'qwen': self.config.get('prompts', 'qwen'),
            'qna_cve': self.config.get('prompts', 'qna_cve'),
            'intent': self.config.get('prompts', 'intent'),
            'action': self.config.get('prompts', 'action'),
            'plan': self.config.get('prompts', 'plan'),
            'playbook': self.config.get('prompts', 'playbook'),
            'qna_gpt': self.config.get('prompts', 'qna_gpt'),
            'plan_gpt': self.config.get('prompts', 'plan_gpt'),
            'action_gpt': self.config.get('prompts', 'action_gpt'),
            'qna_gemma': self.config.get('prompts', 'qna_gemma', fallback=self.config.get('prompts', 'qna_gpt')),
            'action_gemma': self.config.get('prompts', 'action_gemma', fallback=self.config.get('prompts', 'action_gpt')),
            'claude': self.config.get('prompts', 'claude', fallback=self.config.get('prompts', 'gpt')),
            'qna_claude': self.config.get('prompts', 'qna_claude', fallback=self.config.get('prompts', 'qna_gpt')),
            'action_claude': self.config.get('prompts', 'action_claude', fallback=self.config.get('prompts', 'action_gpt')),
            'plan_claude': self.config.get('prompts', 'plan_claude', fallback=self.config.get('prompts', 'plan_gpt')),
            'title_claude': self.config.get('prompts', 'title_claude', fallback=self.config.get('prompts', 'title', fallback=None)),
            'focus_query': self.config.get('prompts', 'focus_query'),
            'post_action': self.config.get('prompts', 'post_action'),
            'ticket_memo': self.config.get('prompts', 'ticket_memo'),
            'slack': self.config.get('prompts', 'slack', fallback=None),
            'title': self.config.get('prompts', 'title', fallback=None),
            'domain_keywords': self.config.get('prompts', 'domain_keywords', fallback=None)
        }

    def get_regression_config(self) -> Dict[str, str]:
        return {
            'intent_analysis': self.config.get('regression', 'intent_analysis'),
            'validation': self.config.get('regression', 'validation')
        }

    def get_option_config(self) -> Dict[str, str]:
        return {
            'use_hnsw': self.config.get('option','use_hnsw'),
            'focus': self.config.get('option', 'focus'),
            'response': self.config.get('option', 'response'),
            'translation': self.config.get('option','translation'),
            'user_intent_prompt': self.config.get('option','user_intent_prompt'),
            'show_metadata': self.config.get('option','show_metadata').strip().lower() == 'true'
        }


    def get_pii_config(self) -> Dict[str, bool]:
        return {
            'pii_mode': self.config.get('pii', 'pii_mode', fallback='False').strip().lower() == 'true',
            'verbose': self.config.get('pii', 'verbose', fallback='False').strip().lower() == 'true',
            'restore': self.config.get('pii', 'restore', fallback='True').strip().lower() == 'true',
        }

    def get_validation_config(self):
        return {
            'base_url': self.config.get('validation', 'base_url'),
            'api_key': self.config.get('validation', 'api_key'),
            'llama_val_action': self.config.get('validation', 'llama_val_action'),
            'llama_val_qna': self.config.get('validation', 'llama_val_qna'),
            'gpt_val': self.config.get('validation', 'gpt_val'),
            'queue_path': self.config.get('validation', 'queue_path',
                                          fallback='/api/ai/{chat_guid}/queue'),
            'apps_path': self.config.get('validation', 'apps_path', fallback='/api/apps')
        }

    def get_server_config(self) -> Dict[str, int]:
        return {
            'port' : int(self.config.get('server','port')),
            'saas_mode' : self.config.get('server', 'saas_mode', fallback='False').lower() == 'true'
        }

    def get_db_config(self):
        return {
            'url' : self.config.get('database','url'),
            'port' : int(self.config.get('database','port')),
            'user' : self.config.get('database','user'),
            'password' : self.config.get('database','password'),
            'database' : self.config.get('database','database'),
            'use_db_mode': self.config.get('database','use_db_mode'),
        }

    def get_search_config(self) -> Dict[str, Any]:
        return {
            'search_strategy': self.config.get('search', 'search_strategy', fallback='basic'),
            'max_parallel_searches': self.config.getint('search', 'max_parallel_searches', fallback=4),
            'cache_enabled': self.config.getboolean('search', 'cache_enabled', fallback=True),
            'cache_expiry_seconds': self.config.getint('search', 'cache_expiry_seconds', fallback=3600),
            'search_timeout_seconds': self.config.getint('search', 'search_timeout_seconds', fallback=10)
        }

    def get_search_weights_config(self) -> Dict[str, float]:
        return {
            'filename_question': self.config.getfloat('search_weights', 'filename_question', fallback=0.35),
            'exact_keyword': self.config.getfloat('search_weights', 'exact_keyword', fallback=0.25),
            'semantic_vector': self.config.getfloat('search_weights', 'semantic_vector', fallback=0.20),
            'translated_search': self.config.getfloat('search_weights', 'translated_search', fallback=0.15),
            'knowledge_graph': self.config.getfloat('search_weights', 'knowledge_graph', fallback=0.25),
            'contextual_expansion': self.config.getfloat('search_weights', 'contextual_expansion', fallback=0.10)
        }

    def get_domain_expansion_config(self) -> Dict[str, str]:
        if 'domain_expansion' not in self.config:
            return {}

        return {
            key: value for key, value in self.config['domain_expansion'].items()
        }

    def validate_paths(self) -> None:
        paths = self.get_paths_config()

        for name, path in paths.items():
            if not path.exists():
                print(f"📂 {name.replace('_', ' ').title()} directory not found. Creating: {path}")
                path.mkdir(parents=True, exist_ok=True)

    def _get_or_create_key(self) -> bytes:
        utils_dir = Path('utils')
        if not utils_dir.exists():
            utils_dir.mkdir(parents=True, exist_ok=True)

        key_file = utils_dir / '.encryption_key'

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = secrets.token_bytes(32)
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
            return key

    def _xor_encrypt_decrypt(self, data: bytes, key: bytes) -> bytes:
        extended_key = key * (len(data) // len(key) + 1)
        extended_key = extended_key[:len(data)]

        result = bytes(a ^ b for a, b in zip(data, extended_key))
        return result

    def encrypt_value(self, value: str) -> str:
        key = self._get_or_create_key()

        salt = secrets.token_bytes(16)
        salted_value = salt + value.encode()

        encrypted = self._xor_encrypt_decrypt(salted_value, key)

        return f"ENC:{base64.b64encode(encrypted).decode()}"

    def decrypt_value(self, encrypted_value: str) -> str:
        if not encrypted_value.startswith("ENC:"):
            return encrypted_value

        key = self._get_or_create_key()

        encrypted_data = base64.b64decode(encrypted_value[4:])

        decrypted = self._xor_encrypt_decrypt(encrypted_data, key)

        original_value = decrypted[16:]

        return original_value.decode()

    def check_and_setup_db_credentials(self) -> bool:

        import getpass
        import pymysql

        while True:
            db_section = 'database'
            required_keys = ['url', 'port', 'user', 'password', 'database']
            missing_keys = []

            if not self.config.has_section(db_section):
                self.config.add_section(db_section)
                missing_keys = required_keys
            else:
                for key in required_keys:
                    if not self.config.has_option(db_section, key) or not self.config.get(db_section, key).strip():
                        missing_keys.append(key)

            if missing_keys:
                print("\n🔐 데이터베이스 접속 정보가 누락되었습니다. 설정을 진행합니다.")

                for key in missing_keys:
                    if key == 'password':
                        value = getpass.getpass(f"   {key}: ")
                    elif key == 'port':
                        value = input(f"   {key} (default: 3306): ") or "3306"
                    elif key == 'url':
                        value = input(f"   {key} (default: localhost): ") or "localhost"
                    else:
                        value = input(f"   {key}: ")

                    encrypted = self.encrypt_value(value)
                    self.config.set(db_section, key, encrypted)

                with open(self.config_path, 'w', encoding='utf-8') as f:
                    self.config.write(f)
                print("✅ 데이터베이스 접속 정보가 암호화되어 저장되었습니다.")
            else:
                needs_update = False
                for key in required_keys:
                    value = self.config.get(db_section, key)
                    if value and not value.startswith("ENC:"):
                        encrypted = self.encrypt_value(value)
                        self.config.set(db_section, key, encrypted)
                        needs_update = True

                if needs_update:
                    with open(self.config_path, 'w', encoding='utf-8') as f:
                        self.config.write(f)
                    print("✅ 기존 데이터베이스 접속 정보가 암호화되었습니다.")

            print("\n🔍 데이터베이스 연결 테스트 중...")
            try:
                db_config = self.get_db_config()
                connection = pymysql.connect(
                    host=db_config.get('url', 'localhost'),
                    port=db_config.get('port', 3306),
                    user=db_config.get('user'),
                    password=db_config.get('password'),
                    database=db_config.get('database'),
                    charset='utf8mb4'
                )
                connection.close()
                print("✅ 데이터베이스 연결 성공!\n")
                return True

            except Exception as e:
                print(f"❌ 데이터베이스 연결 실패: {e}")
                print("다시 입력해주세요.\n")

                for key in required_keys:
                    self.config.set(db_section, key, '')

                with open(self.config_path, 'w', encoding='utf-8') as f:
                    self.config.write(f)

                continue

    def get_db_config(self):
        port_value = self.config.get('database', 'port', fallback='')
        if port_value:
            port_decrypted = self.decrypt_value(port_value)
            port = int(port_decrypted) if port_decrypted else 3306
        else:
            port = 3306

        return {
            'url': self.decrypt_value(self.config.get('database', 'url', fallback='')) or 'localhost',
            'port': port,
            'user': self.decrypt_value(self.config.get('database', 'user', fallback='')),
            'password': self.decrypt_value(self.config.get('database', 'password', fallback='')),
            'database': self.decrypt_value(self.config.get('database', 'database', fallback='')),
            'use_db_mode': self.config.get('database', 'use_db_mode', fallback='False'),
        }

    def get_llama_server_config(self) -> Dict[str, Any]:
        return {
            'use_server_mode': self.config.getboolean('llama_server', 'use_server_mode', fallback=False),
            'external_server': self.config.getboolean('llama_server', 'external_server', fallback=False),
            'binary_path': self.config.get('llama_server', 'binary_path', fallback='/data/ai-agent/llama-cpp-python/vendor/llama.cpp/build/bin/llama-server'),
            'host': self.config.get('llama_server', 'host', fallback='127.0.0.1'),
            'port': self.config.getint('llama_server', 'port', fallback=8181),
            'n_gpu_layers': self.config.getint('llama_server', 'n_gpu_layers', fallback=99),
            'n_ctx': self.config.getint('llama_server', 'n_ctx', fallback=32768),
            'n_parallel': self.config.getint('llama_server', 'n_parallel', fallback=4),
            'n_batch': self.config.getint('llama_server', 'n_batch', fallback=1024),
            'extra_args': self.config.get('llama_server', 'extra_args', fallback=''),
            # KV 캐시 재사용 — None이면 핸들러의 레거시 기본값을 따름 (docs/kv-cache-reuse.md)
            'cache_reuse': self.config.getint('llama_server', 'cache_reuse', fallback=None),
            'slot_prompt_similarity': self.config.getfloat('llama_server', 'slot_prompt_similarity', fallback=None),
        }

    def get_llama_server_translation_config(self) -> Dict[str, Any]:
        return {
            'use_server_mode': self.config.getboolean('llama_server_translation', 'use_server_mode', fallback=False),
            'external_server': self.config.getboolean('llama_server_translation', 'external_server', fallback=False),
            'binary_path': self.config.get('llama_server_translation', 'binary_path', fallback='/data/ai-agent/llama-cpp-python/vendor/llama.cpp/build/bin/llama-server'),
            'host': self.config.get('llama_server_translation', 'host', fallback='127.0.0.1'),
            'port': self.config.getint('llama_server_translation', 'port', fallback=8182),
            'n_gpu_layers': self.config.getint('llama_server_translation', 'n_gpu_layers', fallback=99),
            'n_ctx': self.config.getint('llama_server_translation', 'n_ctx', fallback=16384),
            'n_parallel': self.config.getint('llama_server_translation', 'n_parallel', fallback=4),
            'n_batch': self.config.getint('llama_server_translation', 'n_batch', fallback=512),
            'extra_args': self.config.get('llama_server_translation', 'extra_args', fallback=''),
            'cache_reuse': self.config.getint('llama_server_translation', 'cache_reuse', fallback=None),
            'slot_prompt_similarity': self.config.getfloat('llama_server_translation', 'slot_prompt_similarity', fallback=None),
        }

    def get_llama_server_gemma_config(self) -> Dict[str, Any]:
        return {
            'use_server_mode': self.config.getboolean('llama_server_gemma', 'use_server_mode', fallback=False),
            'external_server': self.config.getboolean('llama_server_gemma', 'external_server', fallback=False),
            'binary_path': self.config.get('llama_server_gemma', 'binary_path', fallback='/data/ai-agent/llama.cpp-gemma4/build/bin/llama-server'),
            'host': self.config.get('llama_server_gemma', 'host', fallback='127.0.0.1'),
            'port': self.config.getint('llama_server_gemma', 'port', fallback=8183),
            'n_gpu_layers': self.config.getint('llama_server_gemma', 'n_gpu_layers', fallback=99),
            'n_ctx': self.config.getint('llama_server_gemma', 'n_ctx', fallback=262144),
            'n_parallel': self.config.getint('llama_server_gemma', 'n_parallel', fallback=2),
            'n_batch': self.config.getint('llama_server_gemma', 'n_batch', fallback=2048),
            'extra_args': self.config.get('llama_server_gemma', 'extra_args', fallback=''),
            'cache_reuse': self.config.getint('llama_server_gemma', 'cache_reuse', fallback=None),
            'slot_prompt_similarity': self.config.getfloat('llama_server_gemma', 'slot_prompt_similarity', fallback=None),
        }

    def get_slack_config(self) -> Dict[str, str]:
        if 'slack' not in self.config:
            return {'enabled': 'false'}
        return {
            'enabled': self.config.get('slack', 'enabled', fallback='false'),
            'bot_token': self.config.get('slack', 'bot_token', fallback=''),
            'app_token': self.config.get('slack', 'app_token', fallback=''),
        }

# ── 카트리지 프롬프트 로더 (공유 유틸) ─────────────────────────────
_PROMPT_FILE_CACHE = {}

def load_prompt_file(key: str, fallback: str) -> str:
    """[prompts] 키의 경로에서 프롬프트 로드(캐시). 미설정/미존재 시 범용 폴백.
    카트리지가 도메인 프롬프트를 교체하는 표준 통로."""
    if key not in _PROMPT_FILE_CACHE:
        text = None
        try:
            path = ConfigManager().config.get('prompts', key, fallback=None)
            if path and os.path.exists(path):
                text = open(path, encoding='utf-8').read().strip()
        except Exception:
            pass
        _PROMPT_FILE_CACHE[key] = text or fallback
    return _PROMPT_FILE_CACHE[key]

TRANSLATION_PROMPT_KEYS = {'SUMMARY': 'translation_summary', 'NORMAL': 'translation_normal', 'CVE': 'translation_cve'}

def load_translation_prompt(mode: str) -> str:
    return load_prompt_file(
        TRANSLATION_PROMPT_KEYS.get(mode, 'translation_normal'),
        "You are a Korean-to-English translator. Translate the user's Korean text into natural "
        "English. Return only the translation.")


# ─────────────────────────────────────────────────────────────
# Qdrant 지식 컬렉션명 — 단일 소스 ([qdrant] collection, 기본 "bge")
# 코드 리터럴 20곳에 흩어져 있던 고정값을 여기로 모았다. 인스턴스별로 값을 달리 주면
# Qdrant 1대를 공유해도 지식이 격리된다 (docs/multi-instance.md §Qdrant 공유).
# ─────────────────────────────────────────────────────────────
_QDRANT_COLLECTION: Optional[str] = None

def qdrant_collection() -> str:
    global _QDRANT_COLLECTION
    if _QDRANT_COLLECTION is None:
        try:
            cp = configparser.ConfigParser()
            cp.read("config.ini", encoding='utf-8')
            _QDRANT_COLLECTION = (cp.get('qdrant', 'collection', fallback='bge') or 'bge').strip() or 'bge'
        except Exception:
            _QDRANT_COLLECTION = 'bge'
    return _QDRANT_COLLECTION

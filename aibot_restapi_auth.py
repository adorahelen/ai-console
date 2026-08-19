
import os
import json
import secrets
import uuid
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import HTTPException, Request, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from config_utils import ConfigManager
from ipaddress import ip_address, ip_network
import re
import threading

from aibot_db_manager import AibotDBManager
from aibot_db_command import SQL_QUERIES
from aibot_validation import Query_validation

config = ConfigManager()

DEFAULT_MODEL = config.get_model_config().get('model')

API_KEYS_FOLDER = config.get_paths_config().get('api_keys_dir', 'api_keys')

# 관리 평면 전용 키 파일. 사용자 Bearer 키 스캔(_verify_file_api_key)에서 제외되며
# verify_admin_key 만 읽는다 (security-review.md S-2/S-7).
ADMIN_KEY_FILE = 'admin.key'
    
class ApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="사용자 이름")
    account: str = Field(..., min_length=2, max_length=50, description="계정 이름")
    description: Optional[str] = Field(None, max_length=200, description="설명")
    acl: Optional[str] = Field(None, max_length=500, description="IP 화이트리스트")

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('이름은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @validator('account')
    def validate_account(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('계정명은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @validator('description')
    def validate_description(cls, v):
        if v and ('<' in v or '>' in v or 'script' in v.lower()):
            raise ValueError('설명에 HTML 태그나 스크립트는 허용되지 않습니다')
        return v

    @validator('acl')
    def validate_acl(cls, v):
        if v:
            cidrs = [cidr.strip() for cidr in v.split(',')]
            for cidr in cidrs:
                if cidr:
                    try:
                        ip_network(cidr, strict=False)
                    except ValueError:
                        raise ValueError(f'잘못된 IP CIDR 형식: {cidr}')
        return v

class ApiKeyUpdateRequest(BaseModel):
    api_key: str = Field(..., min_length=36, max_length=36, description="API 키")
    name: str = Field(..., min_length=2, max_length=50, description="사용자 이름")
    account: str = Field(..., min_length=2, max_length=50, description="계정 이름")
    description: Optional[str] = Field(None, max_length=200, description="설명")
    acl: Optional[str] = Field(None, max_length=500, description="IP 화이트리스트")

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('이름은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @validator('account')
    def validate_account(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('계정명은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @validator('description')
    def validate_description(cls, v):
        if v and ('<' in v or '>' in v or 'script' in v.lower()):
            raise ValueError('설명에 HTML 태그나 스크립트는 허용되지 않습니다')
        return v

    @validator('acl')
    def validate_acl(cls, v):
        if v:
            cidrs = [cidr.strip() for cidr in v.split(',')]
            for cidr in cidrs:
                if cidr:
                    try:
                        ip_network(cidr, strict=False)
                    except ValueError:
                        raise ValueError(f'잘못된 IP CIDR 형식: {cidr}')
        return v
    
class ApiKeyRenewRequest(BaseModel):
    api_key: str = Field(..., min_length=36, max_length=36, description="API 키")
    duration: int = Field(1, gt=0, le=100, description="연장 기간")
    unit: str = Field("year", pattern="^(day|month|year)$", description="기간 단위")

# 구독 설정 enum (qa_llm.py 의 VALID_LENGTHS / VALID_REASONING_EFFORTS 와 동기 유지)
_SETTINGS_VALID_LENGTHS = ('low', 'medium', 'high')
_SETTINGS_VALID_REASONING = ('minimal', 'low', 'medium', 'high')

class SubscriptionSettingsRequest(BaseModel):
    """구독 단위 model/length/reasoning_effort 부분 갱신. 명시한 필드만 갱신되며, null 명시는 폴백 모드(NULL)로 되돌리기."""
    api_key: str = Field(..., min_length=36, max_length=36, description="API 키")
    model: Optional[str] = Field(None, max_length=64, description="모델명 (null=폴백 복귀)")
    length: Optional[str] = Field(None, description="low/medium/high 또는 null")
    reasoning_effort: Optional[str] = Field(None, description="minimal/low/medium/high 또는 null")

    @validator('length')
    def validate_length(cls, v):
        if v is not None and v not in _SETTINGS_VALID_LENGTHS:
            raise ValueError(f'length는 {"/".join(_SETTINGS_VALID_LENGTHS)} 중 하나여야 합니다')
        return v

    @validator('reasoning_effort')
    def validate_reasoning_effort(cls, v):
        if v is not None and v not in _SETTINGS_VALID_REASONING:
            raise ValueError(f'reasoning_effort는 {"/".join(_SETTINGS_VALID_REASONING)} 중 하나여야 합니다')
        return v

class ApiKeyVerify(BaseModel):
    api_key: str = Field(..., min_length=36, max_length=36, description="API 키")

class ApiKeyDelete(BaseModel):
    api_key: str = Field(..., min_length=36, max_length=36, description="API 키")

class AdminAuthRequest(BaseModel):
    admin_key: str = Field(..., min_length=32, max_length=64, description="관리자 키 (api_keys/admin.key)")

class AdminListRequest(BaseModel):
    admin_key: str = Field(..., min_length=32, max_length=64, description="관리자 키 (api_keys/admin.key)")
    name: Optional[str] = Field(None, min_length=2, max_length=50, description="검색할 이름")
    account: Optional[str] = Field(None, min_length=2, max_length=50, description="검색할 계정")

    @validator('name')
    def validate_name(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('이름은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @validator('account')
    def validate_account(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('계정명은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50, description="사용자 이름")
    password: str = Field(..., min_length=1, max_length=100, description="패스워드 (마스터 계정 또는 API 키)")

class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=100, description="현재 비밀번호")
    new_password: str = Field(..., min_length=8, max_length=100, description="새 비밀번호")

    @validator('new_password')
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('비밀번호는 최소 8자 이상이어야 합니다')
        if not re.search(r'[A-Z]', v):
            raise ValueError('비밀번호는 최소 1개의 대문자를 포함해야 합니다')
        if not re.search(r'[a-z]', v):
            raise ValueError('비밀번호는 최소 1개의 소문자를 포함해야 합니다')
        if not re.search(r'[0-9]', v):
            raise ValueError('비밀번호는 최소 1개의 숫자를 포함해야 합니다')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('비밀번호는 최소 1개의 특수문자를 포함해야 합니다')
        return v

class AdminSessionResponse(BaseModel):
    token: str
    username: str
    expires_at: str
    first_login: bool = False  # 최초 로그인 여부

class GenerateApiKeyRequest(BaseModel):
    admin_key: str = Field(..., min_length=32, max_length=64, description="관리자 키 (api_keys/admin.key)")
    name: str = Field(..., min_length=2, max_length=50, description="사용자 이름")
    account: str = Field(..., min_length=2, max_length=50, description="계정 이름")
    description: Optional[str] = Field(None, max_length=200, description="설명")
    acl: Optional[str] = Field(None, max_length=500, description="IP 화이트리스트")

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('이름은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @validator('account')
    def validate_account(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('계정명은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @validator('description')
    def validate_description(cls, v):
        if v and ('<' in v or '>' in v or 'script' in v.lower()):
            raise ValueError('설명에 HTML 태그나 스크립트는 허용되지 않습니다')
        return v

    @validator('acl')
    def validate_acl(cls, v):
        if v:
            cidrs = [cidr.strip() for cidr in v.split(',')]
            for cidr in cidrs:
                if cidr:
                    try:
                        ip_network(cidr, strict=False)
                    except ValueError:
                        raise ValueError(f'잘못된 IP CIDR 형식: {cidr}')
        return v

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

db_manager = AibotDBManager(
    config=config,
    query_properties=SQL_QUERIES
)

# Session management without JWT
SESSION_EXPIRATION_HOURS = 24

# Session storage: token -> session_data
# In production, consider using Redis or database
session_tokens = {}

# Master account credentials file path
MASTER_CREDENTIALS_FILE = "ssl/.master_credentials.json"

def generate_salt():
    """Generate a random salt for password hashing"""
    return secrets.token_hex(32)

def hash_password_with_salt(password: str, salt: str) -> str:
    """Hash password with salt using SHA256"""
    salted_password = f"{salt}{password}".encode()
    return hashlib.sha256(salted_password).hexdigest()

def load_master_credentials():
    """Load master credentials from file or create default if not exists"""
    default_credentials = {
        "username": "admin",
        "salt": generate_salt(),
        "password_hash": None,  # Will be set below
        "first_login": True,
        "last_changed": None,
        "password_history": [],
        "failed_attempts": 0,
        "last_failed_attempt": None,
        "locked_until": None
    }

    # Set default password hash with salt
    default_credentials["password_hash"] = hash_password_with_salt(
        "admin", default_credentials["salt"]
    )

    try:
        if os.path.exists(MASTER_CREDENTIALS_FILE):
            with open(MASTER_CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
                credentials = json.load(f)
                # Validate required fields
                required_fields = ["username", "salt", "password_hash"]
                if all(field in credentials for field in required_fields):
                    return credentials
                else:
                    print("Invalid credentials file format. Creating new default credentials.")
                    save_master_credentials(default_credentials)
                    return default_credentials
        else:
            # Create new credentials file with default values
            os.makedirs(os.path.dirname(MASTER_CREDENTIALS_FILE), exist_ok=True)
            save_master_credentials(default_credentials)
            print(f"Created new master credentials file: {MASTER_CREDENTIALS_FILE}")
            return default_credentials
    except Exception as e:
        print(f"Error loading credentials file: {e}. Using default credentials.")
        return default_credentials

def save_master_credentials(credentials: dict) -> bool:
    """Save master credentials to file"""
    try:
        os.makedirs(os.path.dirname(MASTER_CREDENTIALS_FILE), exist_ok=True)
        with open(MASTER_CREDENTIALS_FILE, 'w', encoding='utf-8') as f:
            json.dump(credentials, f, indent=2, ensure_ascii=False)
        # Set file permissions to 600 (owner read/write only)
        os.chmod(MASTER_CREDENTIALS_FILE, 0o600)
        return True
    except Exception as e:
        print(f"Failed to save master credentials: {e}")
        return False

# Load master credentials on startup
master_credentials = load_master_credentials()

def verify_master_account(username: str, password: str) -> tuple:
    """마스터 계정 인증
    Returns: (is_valid, first_login)
    """
    if username != master_credentials["username"]:
        # Record failed attempt
        master_credentials["failed_attempts"] = master_credentials.get("failed_attempts", 0) + 1
        master_credentials["last_failed_attempt"] = datetime.now().isoformat()
        save_master_credentials(master_credentials)
        return False, False

    # Check if account is locked
    locked_until = master_credentials.get("locked_until")
    if locked_until:
        lock_time = datetime.fromisoformat(locked_until)
        if datetime.now() < lock_time:
            return False, False

    # Hash input password with stored salt
    salt = master_credentials.get("salt", "")
    input_password_hash = hash_password_with_salt(password, salt)

    if input_password_hash != master_credentials["password_hash"]:
        # Record failed attempt
        master_credentials["failed_attempts"] = master_credentials.get("failed_attempts", 0) + 1
        master_credentials["last_failed_attempt"] = datetime.now().isoformat()

        # Lock account after 5 failed attempts for 30 minutes
        if master_credentials["failed_attempts"] >= 5:
            master_credentials["locked_until"] = (datetime.now() + timedelta(minutes=30)).isoformat()

        save_master_credentials(master_credentials)
        return False, False

    # Reset failed attempts on successful login
    if master_credentials.get("failed_attempts", 0) > 0:
        master_credentials["failed_attempts"] = 0
        master_credentials["last_failed_attempt"] = None
        master_credentials["locked_until"] = None
        save_master_credentials(master_credentials)

    return True, master_credentials.get("first_login", False)

def validate_password_policy(password: str) -> tuple:
    """비밀번호 정책 검증
    Returns: (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "비밀번호는 최소 8자 이상이어야 합니다"

    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    if not (has_upper and has_lower and has_digit):
        return False, "비밀번호는 대문자, 소문자, 숫자를 포함해야 합니다"

    return True, ""

def update_master_password(new_password: str, skip_validation: bool = False) -> tuple:
    """마스터 계정 비밀번호 업데이트
    Returns: (success, error_message)
    """
    try:
        # Validate password policy
        if not skip_validation:
            is_valid, error_msg = validate_password_policy(new_password)
            if not is_valid:
                return False, error_msg

        # Generate new salt
        new_salt = generate_salt()
        new_hash = hash_password_with_salt(new_password, new_salt)

        # Check password history (prevent reusing last 3 passwords)
        password_history = master_credentials.get("password_history", [])
        for old_entry in password_history[-3:]:
            old_salt = old_entry.get("salt")
            if old_salt:
                test_hash = hash_password_with_salt(new_password, old_salt)
                if old_entry.get("password_hash") == test_hash:
                    return False, "새 비밀번호는 최근 3개의 이전 비밀번호와 달라야 합니다"

        # Save current password to history
        if master_credentials.get("password_hash"):
            password_history.append({
                "password_hash": master_credentials["password_hash"],
                "salt": master_credentials["salt"],
                "changed_at": master_credentials.get("last_changed") or datetime.now().isoformat()
            })
            # Keep only last 5 passwords in history
            password_history = password_history[-5:]

        # Update credentials
        master_credentials["salt"] = new_salt
        master_credentials["password_hash"] = new_hash
        master_credentials["first_login"] = False
        master_credentials["last_changed"] = datetime.now().isoformat()
        master_credentials["password_history"] = password_history
        master_credentials["failed_attempts"] = 0
        master_credentials["last_failed_attempt"] = None
        master_credentials["locked_until"] = None

        # Save to file
        if save_master_credentials(master_credentials):
            return True, "비밀번호가 성공적으로 변경되었습니다"
        else:
            return False, "비밀번호 저장 중 오류가 발생했습니다"

    except Exception as e:
        print(f"Failed to update master password: {e}")
        return False, f"비밀번호 업데이트 실패: {str(e)}"

embedding_jobs = {}

class EmbeddingJobStatus(BaseModel):
    sub_id: int
    status: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

def process_embedding_background(sub_id: int, model_type: str):
    try:
        embedding_jobs[sub_id] = {
            "status": "processing",
            "started_at": datetime.now().isoformat(),
            "error": None
        }

        from aibot_embedding import EmbeddingGenerator

        local_model_path = "../models/bge-m3"
        generator = None

        if model_type == 'llama':
            generator = EmbeddingGenerator(
                save_db=True,
                use_local_model=True,
                local_model_path=local_model_path,
                batch_db_operations=True
            )
        elif model_type == 'gpt':
            generator = EmbeddingGenerator(
                save_db=True,
                use_local_model=False,
                local_model_path=None,
                batch_db_operations=True
            )
        elif model_type == 'gpt-oss':
            generator = EmbeddingGenerator(
                save_db=True,
                use_local_model=True,
                local_model_path=local_model_path,
                batch_db_operations=True
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {model_type}")

        if generator is not None:
            generator.set_performance_profile("ultra")
            print(f"✅ [{model_type.upper()}] 임베딩 제너레이터 초기화 완료 (sub_id: {sub_id})")
        else:
            raise Exception(f"{model_type} Generator 초기화 실패")

        ctx = generator.embed_changes_only(sub_id)
        generator.postprocess_with_faiss_and_kg(sub_id, ctx, True)

        embedding_jobs[sub_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        })

        print(f"✅ 임베딩 생성 완료 (sub_id: {sub_id})")

    except Exception as e:
        embedding_jobs[sub_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)[:200]
        })
        print(f"❌ 임베딩 생성 실패 (sub_id: {sub_id}): {e}")

async def start_embedding_background(sub_id: int, model_type: str):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, process_embedding_background, sub_id, model_type)

async def copy_base_embedding_to_subscription(sub_id: int):
    """기본 구독(default)의 FAISS 인덱스와 지식그래프를 새로운 구독에 복사"""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, process_copy_base_embedding, sub_id)

def process_copy_base_embedding(sub_id: int):
    """기본 구독의 임베딩 데이터를 복사하는 백그라운드 작업"""
    try:
        print(f"📋 Starting copy process for subscription {sub_id}")

        # 기본 구독 ID 찾기 (name='default')
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id FROM ai_subscriptions WHERE name = %s', ('default',))
                result = cursor.fetchone()
                if not result:
                    print(f'❌ Default subscription not found for copying to sub_id {sub_id}')
                    return
                default_sub_id = result['id'] if isinstance(result, dict) else result[0]
                print(f'📋 Default subscription ID: {default_sub_id}')

        # 기본 구독의 데이터를 새 구독에 복사
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # 1. FAISS 인덱스 복사
                print(f'📋 Copying FAISS indices from {default_sub_id} to {sub_id}')
                cursor.execute('''
                    INSERT INTO ai_faiss_indices (subscription_id, index_name, index_data, metadata, artifact_metadata, created_at, updated_at)
                    SELECT %s, index_name, index_data, metadata, artifact_metadata, NOW(), NOW()
                    FROM ai_faiss_indices
                    WHERE subscription_id = %s
                ''', (sub_id, default_sub_id))

                # 2. FAISS 인덱스 파트 복사
                print(f'📋 Copying FAISS index parts from {default_sub_id} to {sub_id}')
                cursor.execute('''
                    INSERT INTO ai_faiss_index_parts (subscription_id, index_name, part_no, chunk, created_at)
                    SELECT %s, index_name, part_no, chunk, NOW()
                    FROM ai_faiss_index_parts
                    WHERE subscription_id = %s
                ''', (sub_id, default_sub_id))

                # 3. 지식그래프 복사 및 ID 매핑
                print(f'📋 Copying knowledge graphs from {default_sub_id} to {sub_id}')
                cursor.execute('''
                    SELECT id, data_name, metadata
                    FROM ai_knowledge_graph
                    WHERE subscription_id = %s
                ''', (default_sub_id,))
                default_kg_entries = cursor.fetchall()

                for kg_entry in default_kg_entries:
                    old_kg_id = kg_entry['id'] if isinstance(kg_entry, dict) else kg_entry[0]
                    data_name = kg_entry['data_name'] if isinstance(kg_entry, dict) else kg_entry[1]
                    metadata = kg_entry['metadata'] if isinstance(kg_entry, dict) else kg_entry[2]

                    # 새로운 지식그래프 엔트리 생성
                    cursor.execute('''
                        INSERT INTO ai_knowledge_graph (subscription_id, data_name, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, NOW(), NOW())
                    ''', (sub_id, data_name, metadata))

                    new_kg_id = cursor.lastrowid

                    # 4. 해당 지식그래프 파트 복사
                    cursor.execute('''
                        INSERT INTO ai_knowledge_graph_parts (kg_id, part_no, codec, chunk, checksum, created_at)
                        SELECT %s, part_no, codec, chunk, checksum, NOW()
                        FROM ai_knowledge_graph_parts
                        WHERE kg_id = %s
                    ''', (new_kg_id, old_kg_id))

            conn.commit()
            print(f'✅ Successfully copied base embedding data to subscription {sub_id}')

            # 메모리 로딩된 임베딩 데이터에 새 구독 정보를 즉시 반영
            reload_embedding_for_subscription(sub_id)

    except Exception as e:
        print(f'❌ Failed to copy base embedding to subscription {sub_id}: {e}')

def reload_embedding_for_subscription(sub_id: int):
    """새로 생성된 구독의 임베딩 데이터를 메모리에 로드"""
    try:
        # qa_llm.py의 reload_subscription_embedding 함수 호출
        import qa_llm
        return qa_llm.reload_subscription_embedding(sub_id)
    except Exception as e:
        print(f'❌ Failed to import or call qa_llm.reload_subscription_embedding: {e}')
        import traceback
        traceback.print_exc()
        return False

async def get_bearer_api_key_user(
    request: Request,
    api_key: str = Depends(oauth2_scheme)
) -> Dict:
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API 키가 필요합니다",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    return verify_bearer_api_key(api_key, request)


def _verify_file_api_key(api_key: str, request: Request) -> Dict:
    """파일 기반 API 키 검증 — DB 없는 로컬 설치용 ([server] auth_mode = file).
    api_keys/ 디렉토리의 각 *.key 파일 내용(첫 줄)이 유효한 키다."""
    try:
        folder = str(API_KEYS_FOLDER)
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if not fname.endswith('.key'):
                    continue
                if fname == ADMIN_KEY_FILE:
                    # 관리자 키는 **관리 평면 전용**이다 — 사용자 Bearer 키로 통과시키면
                    # 관리 권한과 질의 권한이 한 값으로 합쳐진다 (security-review.md S-7).
                    continue
                try:
                    with open(os.path.join(folder, fname), encoding='utf-8') as f:
                        stored = f.readline().strip()
                except OSError:
                    continue
                if stored and secrets.compare_digest(stored, api_key):
                    name = fname[:-4]
                    return {"id": 1, "user_id": name, "guid": f"file-{name}",
                            # 로컬 설치: 전 핸들러 키 + general 를 허용 (query_type 게이트가
                            # 특정 모델에 막히지 않도록). DB 모드의 구독별 ACL 과 달리 단일 사용자.
                            "permissions": ["general", "gpt", "gpt-oss", "qwen", "gemma",
                                            "llama", "claude"],
                            "model": None, "length": None, "reasoning_effort": None}
    except Exception:
        pass
    raise HTTPException(status_code=401, detail="유효하지 않은 API 키")


def verify_bearer_api_key(api_key: str, request: Request) -> Dict:
    # 인증 모드 — db(기본, 기존 동작) | file(DB 없는 로컬 설치)
    auth_mode = config.config.get('server', 'auth_mode', fallback='db')
    if auth_mode == 'file':
        return _verify_file_api_key(api_key, request)
    try:
       with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, name, guid, acl, expires_at,
                           model, length, reasoning_effort
                    FROM ai_subscriptions
                    WHERE api_key = %s
                """, (api_key,))
                row = cursor.fetchone()

                if not row:
                    raise HTTPException(status_code=401, detail="유효하지 않은 API 키")

                now = datetime.now()
                expires_at = row["expires_at"]

                if expires_at and now > expires_at:
                    raise HTTPException(status_code=401, detail="만료된 API 키")

                client_ip = get_client_ip(request)
                if not is_ip_allowed(client_ip, row.get("acl")):
                    raise HTTPException(status_code=403, detail="허용되지 않은 IP 주소")

                return {
                    "id": row["id"],
                    "user_id": row["name"],
                    "guid": row["guid"],
                    "permissions": ["gpt", "llama", "gpt-oss"],
                    "model": row.get("model"),
                    "length": row.get("length"),
                    "reasoning_effort": row.get("reasoning_effort"),
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip()
    return request.client.host

def is_ip_allowed(client_ip: str, acl_str: str | None) -> bool:
    if not acl_str:
        return True

    try:
        ip = ip_address(client_ip)
    except ValueError:
        return False

    for token in acl_str.split(","):
        cidr = token.strip()
        if not cidr:
            continue
        try:
            net = ip_network(cidr, strict=False)
        except ValueError:
            continue
        if ip in net:
            return True

    return False

def verify_admin_key(admin_key: str) -> bool:
    """관리자 키 검증 — api_keys/admin.key 와 타이밍 안전 비교.

    2026-07-27 복구(security-review.md S-2). 이전 구현은 무조건 True 를 돌려줘
    관리 평면이 통째로 무인증이었다. 되살리면서 정한 규칙:
      - 키 파일이 없거나 비어 있으면 **fail-closed** (관리 API 전면 거부).
        "키가 없으면 통과"는 예전 상태로 조용히 되돌아가는 길이다.
      - 사용자 Bearer 키와 다른 평면 — _verify_file_api_key 는 이 파일을 건너뛴다.
    키 발급은 install.sh(10-b), 사용 측은 run.sh get_admin_key.
    """
    if not admin_key:
        return False
    path = os.path.join(str(API_KEYS_FOLDER), ADMIN_KEY_FILE)
    try:
        with open(path, encoding='utf-8') as f:
            stored = f.readline().strip()
    except OSError:
        logging.getLogger(__name__).warning(
            "관리자 키 파일 없음(%s) — 관리 API 를 거부합니다. "
            "./install.sh --config-only 로 발급하세요.", path)
        return False
    if not stored:
        return False
    return secrets.compare_digest(stored, admin_key)

def setup_subscription_logger():
    logger = logging.getLogger('subscription_audit')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_filename = f"logs/subscription/subscription_audit_{datetime.now().strftime('%Y%m%d')}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    handler = logging.FileHandler(log_filename, encoding='utf-8')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger

def log_subscription_audit(action: str, details: dict, client_ip: str = None, success: bool = True):
    logger = setup_subscription_logger()

    log_data = {
        "action": action,
        "success": success,
        "client_ip": client_ip,
        "timestamp": datetime.now().isoformat(),
        "details": details
    }

    logger.info(json.dumps(log_data, ensure_ascii=False, separators=(',', ':'), indent=None))

def get_client_ip_from_request(request: Request) -> str:
    return get_client_ip(request)

def create_session_token(username: str, sub_id: int) -> str:
    """Create a simple UUID session token"""
    token = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(hours=SESSION_EXPIRATION_HOURS)

    # Store session data
    session_tokens[token] = {
        "username": username,
        "sub_id": sub_id,
        "expires_at": expires_at,
        "created_at": datetime.now()
    }

    # Clean up expired tokens periodically
    cleanup_expired_sessions()

    return token

def cleanup_expired_sessions():
    """Remove expired sessions from memory"""
    now = datetime.now()
    expired_tokens = [
        token for token, data in session_tokens.items()
        if data['expires_at'] < now
    ]
    for token in expired_tokens:
        del session_tokens[token]

def verify_session_token(token: str) -> Dict:
    """Verify session token and return session data"""
    if token not in session_tokens:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

    session_data = session_tokens[token]

    # Check expiration
    if datetime.now() > session_data['expires_at']:
        del session_tokens[token]
        raise HTTPException(status_code=401, detail="세션이 만료되었습니다")

    return session_data

async def get_current_admin(request: Request, token: str = Depends(oauth2_scheme)) -> Dict:
    """Get current admin from session token"""
    try:
        session_data = verify_session_token(token)
        # sub_id가 0 (마스터) 또는 1 (일반 관리자)인 경우 허용
        if session_data['sub_id'] not in [0, 1]:
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다")
        return session_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

async def get_current_master(request: Request, token: str = Depends(oauth2_scheme)) -> Dict:
    """Get current master admin from session token (sub_id=0 only)"""
    try:
        session_data = verify_session_token(token)
        # sub_id가 0 (마스터)인 경우만 허용
        if session_data['sub_id'] != 0:
            raise HTTPException(status_code=403, detail="마스터 관리자 권한이 필요합니다")
        return session_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

def register_auth_routes(app):
    register_auth_routes_original(app)
    register_web_routes(app)

def register_web_routes(app):
    """웹 인터페이스 관련 라우트들"""

    # 웹 페이지 관련 라우트는 테스트 전용 서버로 이동됨
    # utils/test_only/web_server.py 참조

def register_auth_routes_original(app):
    @app.get("/api/embedding-status/{sub_id}",
             description="임베딩 생성 상태 확인",
             tags=["인증"])
    def get_embedding_status(request: Request, sub_id: int):
        if sub_id not in embedding_jobs:
            raise HTTPException(
                status_code=404,
                detail="해당 구독 ID의 임베딩 작업을 찾을 수 없습니다."
            )

        job_info = embedding_jobs[sub_id]
        return {
            "sub_id": sub_id,
            "status": job_info["status"],
            "started_at": job_info["started_at"],
            "completed_at": job_info.get("completed_at"),
            "error": job_info.get("error"),
            "message": {
                "processing": "임베딩 생성이 진행 중입니다. 잠시 후 다시 확인해주세요.",
                "completed": "임베딩 생성이 완료되었습니다. API 키를 사용할 수 있습니다.",
                "failed": "임베딩 생성이 실패했습니다. 관리자에게 문의하세요."
            }.get(job_info["status"], "알 수 없는 상태입니다.")
        }

    @app.post("/api/embedding-status-all",
             description="모든 임베딩 작업 상태 조회 (관리자 전용)",
             tags=["인증"])
    def get_all_embedding_status(request: Request, admin_auth: AdminAuthRequest):
        if not verify_admin_key(admin_auth.admin_key):
            raise HTTPException(
                status_code=401,
                detail="관리자 인증이 필요합니다."
            )

        return {
            "total_jobs": len(embedding_jobs),
            "jobs": [
                {
                    "sub_id": sub_id,
                    "status": job_info["status"],
                    "started_at": job_info["started_at"],
                    "completed_at": job_info.get("completed_at"),
                    "error": job_info.get("error")
                }
                for sub_id, job_info in embedding_jobs.items()
            ]
        }

    @app.post("/api/list",
              description="API 키 목록 조회 (관리자 권한 필요) - name 또는 account로 특정 검색 가능",
              tags=["인증"])
    def list_subscriptions(request: Request, admin_request: AdminListRequest):
        client_ip = get_client_ip_from_request(request)

        # 2026-07-27 복구 (security-review.md S-1) — 이 블록이 통째로 ''' 로 주석 처리돼
        # 있어 무인증으로 전 구독의 평문 API 키가 나갔다.
        if not verify_admin_key(admin_request.admin_key):
            log_subscription_audit(
                action="READ_ATTEMPT",
                details={
                    "search_name": admin_request.name,
                    "search_account": admin_request.account,
                    "error": "Invalid admin key"
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(
                status_code=401,
                detail="관리자 인증이 필요합니다. 유효하지 않은 관리자 키이거나 검증 서버 연결에 실패했습니다."
            )

        base_sql = "SELECT s.id, s.guid, s.name, s.description, s.account, s.api_key, s.acl, s.model, s.length, s.reasoning_effort, s.expires_at, s.created_at, s.updated_at FROM ai_subscriptions s"
        params = []
        where_conditions = []

        if admin_request.name:
            where_conditions.append("s.name = %s")
            params.append(admin_request.name)

        if admin_request.account:
            where_conditions.append("s.account = %s")
            params.append(admin_request.account)

        if where_conditions:
            base_sql += " WHERE " + " AND ".join(where_conditions)

        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(base_sql, params)
                    results = cursor.fetchall()

                    # 평문 키를 그대로 돌려주지 않는다 (security-review.md S-1).
                    # 운영자가 어떤 키인지 식별할 수 있을 만큼만 남긴다 — 뒤 4자.
                    # 실키가 필요하면 발급 시점의 응답이나 api_keys/ 파일을 쓴다.
                    for _row in results:
                        _k = _row.get("api_key") or ""
                        _row["api_key"] = f"****{_k[-4:]}" if len(_k) >= 4 else "****"

                    is_specific_search = bool(admin_request.name or admin_request.account)

                    log_subscription_audit(
                        action="READ",
                        details={
                            "search_mode": "specific" if is_specific_search else "all",
                            "search_name": admin_request.name,
                            "search_account": admin_request.account,
                            "results_count": len(results),
                            "api_key_masked": True
                        },
                        client_ip=client_ip,
                        success=True
                    )

                    return {
                        "count": len(results),
                        "subscriptions": results,
                        "search_mode": "specific" if is_specific_search else "all",
                        "api_key_masked": True
                    }
        except Exception as e:
            log_subscription_audit(
                action="READ",
                details={
                    "search_name": admin_request.name,
                    "search_account": admin_request.account,
                    "error": "Database error",
                    "error_detail": str(e)[:200]
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")


    @app.post("/api/verify",
             description="API 키 유효성 검증",
             tags=["인증"])
    async def verify_api_key(request: Request, key: ApiKeyVerify):
        try:
            user_info = verify_bearer_api_key(key.api_key, request)
            return {
                "valid": True,
                "id": user_info["id"],
                "user_id": user_info["user_id"],
                "guid": user_info["guid"],
                "permissions": user_info["permissions"]
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")
        
    @app.put("/api/update",
             description="API 키 정보 수정",
             tags=["인증"])
    async def update_api_key(request: Request, update_request: ApiKeyUpdateRequest):
        client_ip = get_client_ip_from_request(request)
        now = datetime.now()
        updated_at = now.strftime("%Y-%m-%d %H:%M:%S.000")

        update_sql = """
            UPDATE ai_subscriptions
            SET name = %s,
                account = %s,
                description = %s,
                acl = %s,
                updated_at = %s
            WHERE api_key = %s
        """

        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(update_sql, (
                        update_request.name,
                        update_request.account,
                        update_request.description,
                        update_request.acl,
                        updated_at,
                        update_request.api_key
                    ))
                    if cursor.rowcount == 0:
                        log_subscription_audit(
                            action="UPDATE",
                            details={
                                "api_key_masked": f"{update_request.api_key[:8]}****{update_request.api_key[-8:]}",
                                "error": "API key not found"
                            },
                            client_ip=client_ip,
                            success=False
                        )
                        raise HTTPException(status_code=404, detail="해당 API 키에 대한 구독 정보가 존재하지 않습니다.")
                conn.commit()

                log_subscription_audit(
                    action="UPDATE",
                    details={
                        "api_key_masked": f"{update_request.api_key[:8]}****{update_request.api_key[-8:]}",
                        "name": update_request.name,
                        "account": update_request.account,
                        "description": update_request.description,
                        "acl": update_request.acl,
                        "updated_at": updated_at
                    },
                    client_ip=client_ip,
                    success=True
                )

                return {
                    "message": "구독 정보가 성공적으로 갱신되었습니다.",
                    "updated_at": updated_at,
                    "api_key": update_request.api_key
                }

        except HTTPException:
            raise
        except Exception as e:
            log_subscription_audit(
                action="UPDATE",
                details={
                    "api_key_masked": f"{update_request.api_key[:8]}****{update_request.api_key[-8:]}",
                    "error": "Database error",
                    "error_detail": str(e)[:200]
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")
        
    @app.patch("/api/subscriptions/settings",
               description="구독 단위 model/length/reasoning_effort 설정 갱신 (명시한 필드만, null=폴백)",
               tags=["인증"])
    async def update_subscription_settings(request: Request, settings_request: SubscriptionSettingsRequest):
        client_ip = get_client_ip_from_request(request)
        fields_set = settings_request.model_fields_set - {'api_key'}

        if not fields_set:
            raise HTTPException(status_code=400, detail="갱신할 필드를 하나 이상 지정해주세요 (model/length/reasoning_effort)")

        set_clauses = []
        params = []
        applied = {}
        for field in ('model', 'length', 'reasoning_effort'):
            if field in fields_set:
                value = getattr(settings_request, field)
                set_clauses.append(f"{field} = %s")
                params.append(value)
                applied[field] = value

        now = datetime.now()
        updated_at = now.strftime("%Y-%m-%d %H:%M:%S.000")
        set_clauses.append("updated_at = %s")
        params.append(updated_at)
        params.append(settings_request.api_key)

        update_sql = f"UPDATE ai_subscriptions SET {', '.join(set_clauses)} WHERE api_key = %s"

        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(update_sql, tuple(params))
                    if cursor.rowcount == 0:
                        log_subscription_audit(
                            action="UPDATE_SETTINGS",
                            details={
                                "api_key_masked": f"{settings_request.api_key[:8]}****{settings_request.api_key[-8:]}",
                                "error": "API key not found"
                            },
                            client_ip=client_ip,
                            success=False
                        )
                        raise HTTPException(status_code=404, detail="해당 API 키를 찾을 수 없습니다.")
                conn.commit()

                log_subscription_audit(
                    action="UPDATE_SETTINGS",
                    details={
                        "api_key_masked": f"{settings_request.api_key[:8]}****{settings_request.api_key[-8:]}",
                        "applied": applied,
                        "updated_at": updated_at
                    },
                    client_ip=client_ip,
                    success=True
                )

                return {
                    "message": "구독 설정이 갱신되었습니다.",
                    "applied": applied,
                    "updated_at": updated_at,
                    "api_key": settings_request.api_key
                }

        except HTTPException:
            raise
        except Exception as e:
            log_subscription_audit(
                action="UPDATE_SETTINGS",
                details={
                    "api_key_masked": f"{settings_request.api_key[:8]}****{settings_request.api_key[-8:]}",
                    "error": "Database error",
                    "error_detail": str(e)[:200]
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")

    @app.put("/api/renew", description="API 키 만료일 연장", tags=["인증"])
    async def renew_expires_at(request: Request, renew_request: ApiKeyRenewRequest):
        client_ip = get_client_ip_from_request(request)

        # Calculate new expiration based on unit
        now = datetime.now()
        if renew_request.unit == "day":
            delta = timedelta(days=renew_request.duration)
        elif renew_request.unit == "month":
            delta = timedelta(days=renew_request.duration * 30)  # Approximate month
        elif renew_request.unit == "year":
            delta = timedelta(days=renew_request.duration * 365)
        else:
            raise HTTPException(status_code=400, detail="잘못된 기간 단위입니다")

        new_expires_at = (now + delta).strftime("%Y-%m-%d %H:%M:%S.000")
        updated_at = now.strftime("%Y-%m-%d %H:%M:%S.000")

        update_sql = """
            UPDATE ai_subscriptions
            SET expires_at = %s,
                updated_at = %s
            WHERE api_key = %s
        """

        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(update_sql, (
                        new_expires_at,
                        updated_at,
                        renew_request.api_key
                    ))
                    if cursor.rowcount == 0:
                        log_subscription_audit(
                            action="RENEW",
                            details={
                                "api_key_masked": f"{renew_request.api_key[:8]}****{renew_request.api_key[-8:]}",
                                "error": "API key not found"
                            },
                            client_ip=client_ip,
                            success=False
                        )
                        raise HTTPException(status_code=404, detail="해당 API 키를 찾을 수 없습니다.")
                conn.commit()

                log_subscription_audit(
                    action="RENEW",
                    details={
                        "api_key_masked": f"{renew_request.api_key[:8]}****{renew_request.api_key[-8:]}",
                        "duration": renew_request.duration,
                        "unit": renew_request.unit,
                        "new_expires_at": new_expires_at,
                        "updated_at": updated_at
                    },
                    client_ip=client_ip,
                    success=True
                )

                unit_text = {"day": "일", "month": "개월", "year": "년"}.get(renew_request.unit, renew_request.unit)
                return {
                    "message": f"만료일이 {renew_request.duration}{unit_text} 연장되었습니다.",
                    "expires_at": new_expires_at,
                    "api_key": renew_request.api_key
                }

        except HTTPException:
            raise
        except Exception as e:
            log_subscription_audit(
                action="RENEW",
                details={
                    "api_key_masked": f"{renew_request.api_key[:8]}****{renew_request.api_key[-8:]}",
                    "error": "Database error",
                    "error_detail": str(e)[:200]
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")

    @app.delete("/api/delete",
             description="API 키 삭제",
             tags=["인증"])
    async def delete_api_key(request: Request, key: ApiKeyDelete):
        client_ip = get_client_ip_from_request(request)
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM ai_subscriptions WHERE api_key = %s", (key.api_key,))
                    row = cursor.fetchone()

                    if not row:
                        log_subscription_audit(
                            action="DELETE",
                            details={
                                "api_key_masked": f"{key.api_key[:8]}****{key.api_key[-8:]}",
                                "error": "API key not found"
                            },
                            client_ip=client_ip,
                            success=False
                        )
                        raise HTTPException(status_code=404, detail="해당 API 키가 존재하지 않습니다.")

                    sub_id = row["id"]

                    # 관련된 데이터 삭제 (외래키 제약조건 순서대로)

                    # 1. 지식그래프 파트 삭제
                    cursor.execute("""
                        DELETE kgp FROM ai_knowledge_graph_parts kgp
                        INNER JOIN ai_knowledge_graph kg ON kgp.kg_id = kg.id
                        WHERE kg.subscription_id = %s
                    """, (sub_id,))
                    deleted_kg_parts = cursor.rowcount

                    # 2. 지식그래프 삭제
                    cursor.execute("DELETE FROM ai_knowledge_graph WHERE subscription_id = %s", (sub_id,))
                    deleted_kg = cursor.rowcount

                    # 3. FAISS 인덱스 파트 삭제
                    cursor.execute("DELETE FROM ai_faiss_index_parts WHERE subscription_id = %s", (sub_id,))
                    deleted_faiss_parts = cursor.rowcount

                    # 4. FAISS 인덱스 삭제
                    cursor.execute("DELETE FROM ai_faiss_indices WHERE subscription_id = %s", (sub_id,))
                    deleted_faiss = cursor.rowcount

                    # 5. OpenAI 프롬프트 삭제
                    cursor.execute("DELETE FROM openai_prompts WHERE subscription_id = %s", (sub_id,))
                    deleted_prompts = cursor.rowcount

                    # 6. 구독 삭제
                    cursor.execute("DELETE FROM ai_subscriptions WHERE api_key = %s", (key.api_key,))

                    conn.commit()

                    log_subscription_audit(
                        action="DELETE",
                        details={
                            "sub_id": sub_id,
                            "api_key_masked": f"{key.api_key[:8]}****{key.api_key[-8:]}",
                            "deleted_at": datetime.now().isoformat(),
                            "cleaned_data": {
                                "knowledge_graph_parts": deleted_kg_parts,
                                "knowledge_graphs": deleted_kg,
                                "faiss_index_parts": deleted_faiss_parts,
                                "faiss_indices": deleted_faiss,
                                "openai_prompts": deleted_prompts
                            }
                        },
                        client_ip=client_ip,
                        success=True
                    )

                    return {
                        "result": "success",
                        "message": "API 키와 관련 데이터가 모두 삭제되었습니다.",
                        "cleaned_data": {
                            "knowledge_graph_parts": deleted_kg_parts,
                            "knowledge_graphs": deleted_kg,
                            "faiss_index_parts": deleted_faiss_parts,
                            "faiss_indices": deleted_faiss,
                            "openai_prompts": deleted_prompts
                        }
                    }

        except HTTPException:
            raise
        except Exception as e:
            log_subscription_audit(
                action="DELETE",
                details={
                    "api_key_masked": f"{key.api_key[:8]}****{key.api_key[-8:]}",
                    "error": "Database error",
                    "error_detail": str(e)[:200]
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")

    @app.post("/api/admin/login",
             description="관리자 로그인 (마스터 계정 또는 sub_id=1 사용)",
             tags=["인증"])
    async def admin_login(request: Request, login_request: AdminLoginRequest):
        client_ip = get_client_ip_from_request(request)

        try:
            # 1. 먼저 마스터 계정으로 인증 시도
            is_valid, is_first_login = verify_master_account(login_request.username, login_request.password)
            if is_valid:
                # 마스터 계정 로그인 성공
                token = create_session_token(login_request.username, 0)  # sub_id=0 for master
                expires_at = (datetime.now() + timedelta(hours=SESSION_EXPIRATION_HOURS)).isoformat()

                log_subscription_audit(
                    action="ADMIN_LOGIN",
                    details={
                        "username": login_request.username,
                        "sub_id": 0,
                        "login_type": "master",
                        "first_login": is_first_login,
                        "token_prefix": token[:8]
                    },
                    client_ip=client_ip,
                    success=True
                )

                return AdminSessionResponse(
                    token=token,
                    username=login_request.username,
                    expires_at=expires_at,
                    first_login=is_first_login  # 최초 로그인 여부 전달
                )

            # 2. 마스터 계정이 아니면 기존 방식으로 sub_id=1 확인
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, name, api_key, guid
                        FROM ai_subscriptions
                        WHERE id = 1
                    """)
                    admin_row = cursor.fetchone()

                    if not admin_row:
                        log_subscription_audit(
                            action="ADMIN_LOGIN",
                            details={
                                "error": "Admin account not found (sub_id=1)",
                                "username": login_request.username
                            },
                            client_ip=client_ip,
                            success=False
                        )
                        raise HTTPException(status_code=401, detail="관리자 계정이 설정되지 않았습니다")

                    # 사용자명과 패스워드(API 키) 확인
                    if admin_row['name'] != login_request.username or admin_row['api_key'] != login_request.password:
                        log_subscription_audit(
                            action="ADMIN_LOGIN",
                            details={
                                "error": "Invalid credentials",
                                "username": login_request.username
                            },
                            client_ip=client_ip,
                            success=False
                        )
                        raise HTTPException(status_code=401, detail="사용자명 또는 패스워드가 올바르지 않습니다")

                    # 세션 토큰 생성
                    token = create_session_token(admin_row['name'], admin_row['id'])
                    expires_at = (datetime.now() + timedelta(hours=SESSION_EXPIRATION_HOURS)).isoformat()

                    log_subscription_audit(
                        action="ADMIN_LOGIN",
                        details={
                            "username": admin_row['name'],
                            "sub_id": admin_row['id'],
                            "login_type": "subscription",
                            "token_prefix": token[:8]  # Log only token prefix for security
                        },
                        client_ip=client_ip,
                        success=True
                    )

                    return AdminSessionResponse(
                        token=token,
                        username=admin_row['name'],
                        expires_at=expires_at,
                        first_login=False  # 일반 관리자는 최초 로그인 플래그 사용 안 함
                    )

        except HTTPException:
            raise
        except Exception as e:
            log_subscription_audit(
                action="ADMIN_LOGIN",
                details={
                    "error": "Database error",
                    "error_detail": str(e)[:200]
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다")

    @app.post("/api/admin/change-password",
             description="마스터 계정 비밀번호 변경",
             tags=["인증"])
    async def change_master_password(
        request: Request,
        password_change: PasswordChangeRequest,
        current_master: Dict = Depends(get_current_master)
    ):
        client_ip = get_client_ip_from_request(request)

        try:
            # 현재 비밀번호 확인
            is_valid, _ = verify_master_account(
                current_master['username'],
                password_change.current_password
            )

            if not is_valid:
                log_subscription_audit(
                    action="PASSWORD_CHANGE",
                    details={
                        "username": current_master['username'],
                        "error": "Invalid current password"
                    },
                    client_ip=client_ip,
                    success=False
                )
                raise HTTPException(status_code=401, detail="현재 비밀번호가 올바르지 않습니다")

            # 새 비밀번호가 현재 비밀번호와 같은지 확인
            if password_change.current_password == password_change.new_password:
                raise HTTPException(status_code=400, detail="새 비밀번호는 현재 비밀번호와 달라야 합니다")

            # 비밀번호 업데이트 (정책 검증 포함)
            success, message = update_master_password(password_change.new_password)

            if success:
                log_subscription_audit(
                    action="PASSWORD_CHANGE",
                    details={
                        "username": current_master['username'],
                        "password_changed_at": datetime.now().isoformat()
                    },
                    client_ip=client_ip,
                    success=True
                )

                return {
                    "message": message,
                    "changed_at": datetime.now().isoformat()
                }
            else:
                log_subscription_audit(
                    action="PASSWORD_CHANGE",
                    details={
                        "username": current_master['username'],
                        "error": message
                    },
                    client_ip=client_ip,
                    success=False
                )
                raise HTTPException(status_code=400, detail=message)

        except HTTPException:
            raise
        except Exception as e:
            log_subscription_audit(
                action="PASSWORD_CHANGE",
                details={
                    "username": current_master['username'],
                    "error": "Unexpected error",
                    "error_detail": str(e)[:200]
                },
                client_ip=client_ip,
                success=False
            )
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다")

    @app.post("/api/admin/logout",
             description="관리자 로그아웃",
             tags=["인증"])
    async def admin_logout(request: Request, current_admin: Dict = Depends(get_current_admin)):
        client_ip = get_client_ip_from_request(request)

        # Find and remove the token
        token_to_remove = None
        for token, session_data in session_tokens.items():
            if session_data['username'] == current_admin['username']:
                token_to_remove = token
                break

        if token_to_remove:
            del session_tokens[token_to_remove]

        log_subscription_audit(
            action="ADMIN_LOGOUT",
            details={
                "username": current_admin['username'],
                "sub_id": current_admin['sub_id']
            },
            client_ip=client_ip,
            success=True
        )

        return {"message": "로그아웃되었습니다"}

    @app.get("/api/admin/session",
             description="현재 세션 확인",
             tags=["인증"])
    async def check_session(request: Request, current_admin: Dict = Depends(get_current_admin)):
        return {
            "valid": True,
            "username": current_admin['username'],
            "sub_id": current_admin['sub_id'],
            "expires_at": current_admin['expires_at'].isoformat() if hasattr(current_admin['expires_at'], 'isoformat') else str(current_admin['expires_at'])
        }
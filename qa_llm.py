import os
import time
import re
import json
import yaml
import asyncio
import threading
import uuid
import requests
import gc
import logging

logging.getLogger('httpx').setLevel(logging.WARNING)

class _HealthCheckFilter(logging.Filter):
    """헬스체크 엔드포인트의 반복 access log를 숨긴다."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/api/ai/hello" not in msg

logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())
from typing import Dict, List, Any, Optional, AsyncGenerator, Literal, Union
from fastapi import FastAPI, HTTPException, Request, Depends, Query, Body, UploadFile, File, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel
import uvicorn

from config_utils import ConfigManager, qdrant_collection
from aibot_llm_module import LLMHandler, ChatMessage
from handler_registry import HANDLER_CLASSES, LEGACY_ATTR_TO_KEY
from aibot_validation import Query_validation, translate_error_message
from aibot_restapi_auth import register_auth_routes, get_bearer_api_key_user, verify_admin_key
from aibot_logger import ChatLogger
from aibot_prompts_class import Record, LlmPrompt, OpenAiPrompt, StreamBufferRegistry
from aibot_prompts_functions import parse_prompt, create_prompt, remove_prompt, init_generator, toggle_prompt_bge, update_prompt_bge, _is_bge_mode

from aibot_db_manager import AibotDBManager
from aibot_db_command import SQL_QUERIES

# Slack 봇 (선택적)
try:
    from utils.slack_bot import init_slack_bot, register_slack_handlers, run_slack_bot
    SLACK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SLACK_AVAILABLE = False


query_validation = None
loop = None
llm_handler = None

def get_global_llm_handler():
    """전역 llm_handler 반환"""
    global llm_handler
    return llm_handler

def reload_subscription_embedding(sub_id: int):
    """새로 생성된 구독의 임베딩 데이터를 메모리에 리로드"""
    global llm_handler
    try:
        print(f'🔄 Starting memory reload for subscription {sub_id}...')

        if llm_handler is not None:
            print(f'   ✅ Found global llm_handler (ID: {id(llm_handler)}), reloading subscription data...')
            llm_handler.reload_embedding(sub_id=sub_id)
            print(f'   ✅ Successfully reloaded embedding data for subscription {sub_id}')
            return True
        else:
            print(f'   ⚠️ Global llm_handler is None, skipping memory reload')
            print(f'   💡 Hint: AI system may not be initialized yet')
            return False

    except Exception as e:
        print(f'   ❌ Failed to reload embedding for subscription {sub_id}: {e}')
        import traceback
        traceback.print_exc()
        return False


def reload_cartridge_prompts() -> dict:
    """카트리지 장착 후 [prompts] 배선을 런타임에 재적용 — 콘솔 재시작 불필요.

    reload_subscription_embedding 과 같은 패턴(전역 llm_handler 를 찾아 리로드)이다.
    핸들러마다 자기 ConfigManager 를 들고 있으므로 전부 순회해 재파싱시킨다.
    mount 는 [model] 을 건드리지 않으므로(cartridge_mount.plan_mount 주석) 모델
    재기동이 필요한 경우는 없다 — 프리셋 전환은 install 영역이다.

    알려진 한계 (T2 부하 테스트로 검증·개선 예정):
    - **동시성**: 요청이 self.system_prompt 를 intent 별로 swap 후 원복하는 사이에
      리로드가 끼면, 진행 중 요청의 원복이 리로드를 덮어쓸 수 있다(부하 중 mount 시).
      mount 는 드문 관리 작업이라 확률은 낮으나, 확실히 하려면 mount 후 ./run.sh restart.
    - **단일형 카트리지**: intent 모드(user_intent_prompt=True)에서는 요청마다
      qna/action/plan 프롬프트로 base 를 덮으므로, base('system') 슬롯만 정의한
      카트리지는 런타임 리로드로도 페르소나가 안 물린다 — 카트리지가 intent 키까지
      정의하거나 재시작이 필요하다.
    """
    global llm_handler
    if llm_handler is None:
        return {"reloaded": [], "ok": False, "detail": "llm_handler 미초기화 (콘솔 기동 전)"}

    reloaded, failed, applied = [], [], {}
    for key, h in (getattr(llm_handler, 'handlers', None) or {}).items():
        try:
            path = h.reload_system_prompt() if hasattr(h, 'reload_system_prompt') else ""
            if path:
                reloaded.append(key)
                # 무엇이 실제로 물렸는지 응답에 실어 검증 가능하게 한다 (경로 + 내용 지문).
                applied[key] = {
                    "prompt": path,
                    "fingerprint": h.system_prompt_fingerprint() if hasattr(h, 'system_prompt_fingerprint') else None,
                }
            else:
                failed.append(key)
        except Exception as e:
            print(f'   ⚠️ 핸들러 {key} 프롬프트 리로드 실패: {e}')
            failed.append(key)

    # 콘솔 자신의 ConfigManager 도 갱신 (경로·옵션을 읽는 모듈 전역)
    try:
        if config is not None:
            config.load_config()
    except Exception as e:
        print(f'   ⚠️ 전역 config 재파싱 실패: {e}')

    # ok 는 "실제로 요청에 답하는 활성 핸들러"(llm_handler.model)가 리로드됐는지로 판정한다.
    # 유휴 핸들러만 리로드되고 활성 핸들러가 실패해도 ok=True 로 오보하던 문제(리뷰 지적).
    active = getattr(llm_handler, 'model', None)
    ok = bool(active) and active in reloaded

    print(f'🔄 카트리지 프롬프트 리로드 — 성공 {len(reloaded)} / 실패 {len(failed)} / 활성={active} ok={ok}')
    return {"reloaded": reloaded, "failed": failed, "applied": applied, "active": active, "ok": ok}

# API 키 생성 관련 import 추가
from datetime import datetime, timedelta
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from ipaddress import ip_network
import uuid

# 구독 설정 enum (length, reasoning_effort)
VALID_LENGTHS = ('low', 'medium', 'high')
VALID_REASONING_EFFORTS = ('minimal', 'low', 'medium', 'high')

# API 키 생성 관련 모델
class GenerateApiKeyRequest(BaseModel):
    # 관리자 키 — 이 엔드포인트는 문서상 "관리자 권한 필요"인데 실제 검사가 없었다
    # (security-review.md S-3). 다른 관리 API(AdminAuthRequest)와 같은 방식으로 받는다.
    admin_key: str = Field(..., min_length=32, max_length=64, description="관리자 키 (api_keys/admin.key)")
    name: str = Field(..., min_length=1, max_length=50, description="구독 이름")
    account: str = Field(..., min_length=1, max_length=50, description="계정명")
    description: Optional[str] = Field(None, max_length=200, description="설명")
    acl: Optional[str] = Field(None, description="IP 접근 제어 (CIDR 형식, 콤마로 구분)")
    model: Optional[str] = Field(None, max_length=64, description="기본 모델 (NULL 이면 config DEFAULT_MODEL 폴백)")
    length: Optional[str] = Field(None, description="응답 길이 low/medium/high (NULL 이면 medium 폴백)")
    reasoning_effort: Optional[str] = Field(None, description="추론 깊이 minimal/low/medium/high (NULL 이면 config 폴백)")

    @field_validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9가-힣_\-\.\s]+$', v):
            raise ValueError('이름은 영문, 한글, 숫자, _, -, ., 공백만 허용됩니다')
        return v.strip()

    @field_validator('account')
    def validate_account(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('계정명은 영문, 숫자, _, -, . 만 허용됩니다')
        return v

    @field_validator('description')
    def validate_description(cls, v):
        if v and ('<' in v or '>' in v or 'script' in v.lower()):
            raise ValueError('설명에 HTML 태그나 스크립트는 허용되지 않습니다')
        return v

    @field_validator('acl')
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

    @field_validator('length')
    def validate_length(cls, v):
        if v is not None and v not in VALID_LENGTHS:
            raise ValueError(f'length는 {"/".join(VALID_LENGTHS)} 중 하나여야 합니다')
        return v

    @field_validator('reasoning_effort')
    def validate_reasoning_effort(cls, v):
        if v is not None and v not in VALID_REASONING_EFFORTS:
            raise ValueError(f'reasoning_effort는 {"/".join(VALID_REASONING_EFFORTS)} 중 하나여야 합니다')
        return v

config = ConfigManager()
DEFAULT_MODEL = config.get_model_config().get('model')

# length enum → 토큰 캡 매핑은 OpenAIHandler 로 이관 (OpenAI 전용 로직).
# agent/chat/completions2 는 핸들러의 agent_complete 로 위임한다.


def _resolve_setting(request_val, sub_val, default_val):
    """3-tier 우선순위: request body > 구독 설정 > config 디폴트. None/누락은 다음 tier 로 폴백."""
    if request_val is not None:
        return request_val
    if sub_val is not None:
        return sub_val
    return default_val


# 로컬 모델 식별 — handler_registry 에 등록된 모델 중 'gpt' 제외
# (OpenAI/호환 백엔드는 'gpt' 핸들러로 위임, 나머지는 자체 로컬 라우팅)
_LOCAL_MODELS = frozenset(HANDLER_CLASSES.keys()) - {"gpt"}


def _resolve_routing(requested_model: str) -> tuple[str, Optional[str]]:
    """요청 모델명 → (routing_target, model_override).

    - HANDLER_CLASSES 의 로컬 모델(gpt-oss/llama/qwen/gemma): 자체 라우팅, override 없음
    - 'gpt': OpenAI 핸들러 + 핸들러 기본 모델명 사용
    - 그 외 외부 모델명(예: openai/gpt-oss-120b): OpenAI 핸들러로 위임 + 모델명 override
    """
    if requested_model in _LOCAL_MODELS:
        return requested_model, None
    if requested_model == "gpt":
        return "gpt", None
    return "gpt", requested_model

def process_copy_base_embedding_with_reload(sub_id: int):
    """기본 구독의 임베딩 데이터를 복사하고 메모리에 즉시 리로드"""
    global llm_handler

    try:
        print(f"📋 Starting copy process for subscription {sub_id}")

        # DB 매니저 가져오기
        from aibot_db_manager import AibotDBManager
        from aibot_db_command import SQL_QUERIES

        db_manager = AibotDBManager(
            config=config,
            query_properties=SQL_QUERIES
        )

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
            reload_subscription_embedding(sub_id)

    except Exception as e:
        print(f'❌ Failed to copy base embedding to subscription {sub_id}: {e}')

async def copy_base_embedding_to_subscription(sub_id: int):
    """기본 구독(default)의 FAISS 인덱스와 지식그래프를 새로운 구독에 복사"""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process_copy_base_embedding_with_reload, sub_id)


api_app = FastAPI(
    title="RAG 질의응답 API",
    description="RAG 및 지식 그래프 기반 AI 질의응답 API",
    version="1.0"
)

# CORS (security-review.md S-4) — 와일드카드와 자격증명을 동시에 켜면 안 된다.
# starlette 는 allow_origins=["*"] + allow_credentials=True 조합에서 '*' 대신
# **요청 Origin 을 그대로 되비춘다**(CORSMiddleware.send). 즉 임의 사이트가 쿠키를 실은
# 교차출처 요청을 보내고 응답까지 읽을 수 있다 — 쿠키 세션(/api/admin/login)이 있는 이 콘솔에선
# 실질 위험이다. Bearer 키는 브라우저가 자동 첨부하지 않아 이 경로로는 새지 않는다.
#
#   [server] cors_origins 비움(기본) → 와일드카드 + 자격증명 OFF (동봉 web-UI 는 동일 출처라 무관)
#   [server] cors_origins 지정       → 그 오리진만 + 자격증명 ON (교차출처 관리 UI 를 붙일 때)
_cors_origins = [o.strip() for o in
                 config.config.get('server', 'cors_origins', fallback='').split(',') if o.strip()]
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=bool(_cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
print(f"🔒 CORS: {'화이트리스트 ' + ', '.join(_cors_origins) + ' (자격증명 허용)' if _cors_origins else '와일드카드 (자격증명 차단)'}")

# Gzip 압축 (싱크 엔드포인트 등 대용량 JSON 응답 최적화)
from starlette.middleware.gzip import GZipMiddleware
api_app.add_middleware(GZipMiddleware, minimum_size=1000)

register_auth_routes(api_app)

# ── 온보딩 위저드 (docs/onboarding-design.md) — API 라우터 + 단일 페이지 UI ──
from fastapi.responses import FileResponse
from aibot_wizard import create_wizard_router
# 위저드는 활성 핸들러(agent_complete 보유)가 필요 — LLMHandler 컨테이너가 아니라
# handlers[DEFAULT_MODEL] 을 넘긴다 (completions2 라우트와 동일 패턴).
def _active_handler():
    return llm_handler.handlers.get(DEFAULT_MODEL) if llm_handler else None
api_app.include_router(create_wizard_router(_active_handler, get_bearer_api_key_user))

@api_app.get("/wizard", include_in_schema=False)
async def wizard_page():
    return FileResponse("webui/wizard.html")

@api_app.get("/chat", include_in_schema=False)
async def chat_page():
    return FileResponse("webui/chat.html")


class QueryRequest(BaseModel):
    query: str
    query_type: str = "general"

class SummaryRequest(BaseModel):
    llm_model: Optional[str] = DEFAULT_MODEL
    question: str
    user_guid: str
    chat_guid: str
    locale: str

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, dict]

class AssistantRequest(BaseModel):
    user_guid: str
    type: str
    question: Optional[str] = None
    llm_model: Optional[str] = DEFAULT_MODEL
    prompt_count: int
    prompt_token: int
    locale: str
    installed_apps: Optional[List[Dict]] = None
    connect_profiles: Optional[List[Dict]] = None
    messages: Optional[List[Dict]] = None
    response_format: Optional[bool] = True

class QueueRequest(BaseModel):
    intent: str
    conversation_guid: str
    chat_guid: str
    post_action_response: Dict

class TicketRequest(BaseModel):
    user_guid: str
    type: str
    llm_model: Optional[str] = DEFAULT_MODEL
    locale: str
    reference_id: List[Dict]


BOT_USER_ID = None
BOT_NAME = "아로"


db_manager = AibotDBManager(
    config=config,
    query_properties=SQL_QUERIES
)



response_cache = {}
CACHE_EXPIRY = 3600

def get_cached_response(query: str, model_type: str) -> Dict[str, Any]:
    cache_key = f"{query.strip().lower()}:{model_type}"
    cached_item = response_cache.get(cache_key)

    if cached_item:
        if time.time() - cached_item['timestamp'] < CACHE_EXPIRY:
            print(f"✅ 캐시에서 응답 로드: {cache_key[:30]}...")

            response = cached_item['response'].copy()
            if 'search_metadata' not in response:
                response['search_metadata'] = {}
            response['search_metadata']['from_cache'] = True
            response['search_metadata']['cache_timestamp'] = cached_item['timestamp']

            return response

    return None

def store_in_cache(query: str, model_type: str, response: Dict[str, Any]) -> None:
    cache_key = f"{query.strip().lower()}:{model_type}"

    clean_answer = response.get('answer', '')
    clean_response = response.copy()
    clean_response['answer'] = clean_answer

    cache_entry = {
        'response': clean_response,
        'timestamp': time.time(),
        'search_metadata': response.get('search_metadata', {})
    }

    response_cache[cache_key] = cache_entry

    if len(response_cache) > 100:
        oldest_key = min(response_cache.keys(), key=lambda k: response_cache[k]['timestamp'])
        del response_cache[oldest_key]

    search_strategy = response.get('search_metadata', {}).get('search_strategy', 'unknown')
    print(f"✅ 응답을 캐시에 저장 (메타데이터 제거됨): {cache_key[:30]}... (검색전략: {search_strategy})")




async def process_query(query: str, model: str = "gpt", user_id: str = None, channel_id: str = None) -> Dict[str, Any]:
    cached_result = get_cached_response(query, model)
    if cached_result:
        if user_id:
            chat_session = llm_handler.get_chat_session(user_id, channel_id)
            chat_session.add_message("user", query)
            chat_session.add_message("assistant", cached_result["answer"])
        return cached_result

    llm_handler.set_model(model)

    search_start_time = time.time()

    print(f"질문 처리 중: '{query[:50]}...' (모델: {model})")
    start_time = time.time()

    result = await llm_handler.process_question(query, user_id, channel_id, 'API')

    search_end_time = time.time()
    search_duration = search_end_time - search_start_time

    full_response = ""
    async for chunk in result["answer_stream"]:
        full_response += chunk

    response_data = {
        "answer": full_response,
        "sources": result.get("context_sources", []),
        "search_metadata": {
            "search_duration": round(search_duration, 3),
            "total_sources": len(result.get("context_sources", [])),
            "model_used": model,
            "search_strategy": "unknown"
        }
    }

    try:
        # 알 수 없는 모델명은 llama 핸들러로 폴백 (원본 else 분기 보존)
        handler = llm_handler.handlers.get(model) or llm_handler.handlers.get("llama")
        search_status = handler.get_search_status()

        response_data["search_metadata"]["search_strategy"] = search_status["search_config"].get("search_strategy", "unknown")
    except Exception as e:
        print(f"검색 메타데이터 추가 실패: {e}")

    store_in_cache(query, model, response_data)

    print(f"응답 완료: 소요시간 {time.time() - start_time:.2f}초, 검색시간 {search_duration:.2f}초")
    return response_data

def extract_json_block(text: str) -> tuple[str, str, str]:
    match = re.search(r'(?s)(.*?)\n(\[\s*{.*?}\s*])\n(.*)', text)
    if match:
        return match.group(1), match.group(2), match.group(3)

    match = re.search(r'(?s)(.*?)(\[\s*{.*?}\s*])(.*)', text)
    if match:
        return match.group(1), match.group(2), match.group(3)

    raise ValueError("JSON 배열 블록을 찾을 수 없습니다.")

def extract_query_blocks(text: str) -> List[str]:
    """Extract query blocks from markdown formatted text"""

    query_blocks = []

    patterns = [
        r'```query\s*\n(.*?)\n```',      
        r'```query\s*\n(.*?)```',        
        r'```query(.*?)```',             
        r'`{3,}query\s*\n(.*?)\n`{3,}',  
        r'`{3,}query\s*\n(.*?)`{3,}',    
    ]

    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            query = match.strip()

            if query.lower().startswith('query'):
                query = re.sub(r'^query\s+', '', query, flags=re.IGNORECASE).strip()

            if query and query not in query_blocks:
                query_blocks.append(query)
                
    if not query_blocks:

        fallback_patterns = [
            r'(?i)```\s*query[^\n]*\n(.*?)```',              
            r'(?i)`{2,}query[^\n]*\n(.*?)`{2,}',             
            r'(?i)```\s*query[^\n]*\n(.*?)\n*```',           
        ]

        for i, pattern in enumerate(fallback_patterns):
            matches = re.findall(pattern, text, re.DOTALL)

            for match in matches:
                query = match.strip()

                if query and query not in query_blocks:
                    query_blocks.append(query)

    return query_blocks

def replace_query_blocks_in_response(original_response: str, corrected_response: str) -> str:

    original_query_blocks = extract_query_blocks(original_response)

    corrected_query_blocks = extract_query_blocks(corrected_response)

    if not original_query_blocks or not corrected_query_blocks:
        return original_response

    result_response = original_response

    import re
    
    if len(corrected_query_blocks) > 0:
        corrected_query = corrected_query_blocks[0]

        patterns = [
            r'```query\s*\n.*?\n```',
            r'```query\s*\n.*?```',
            r'`{3,}query\s*\n.*?\n`{3,}',
            r'`{3,}query\s*\n.*?`{3,}',
        ]

        replaced = False
        for pattern in patterns:
            if re.search(pattern, result_response, re.DOTALL):
                safe_replacement = f'```query\n{corrected_query}\n```'.replace('\\', '\\\\')
                result_response = re.sub(
                    pattern,
                    safe_replacement,
                    result_response,
                    count=1,
                    flags=re.DOTALL
                )
                replaced = True
                break

        if not replaced:
            return original_response

    return result_response

def validate_qna_queries(response_text: str) -> Dict[str, Any]:
    """Validate queries in QnA response"""
    query_blocks = extract_query_blocks(response_text)

    if not query_blocks:
        return {
            "has_queries": False,
            "queries": [],
            "validation_results": []
        }

    validation = Query_validation()
    results = []

    for query in query_blocks:
        validation_result = validation.validate_query(query)
        results.append({
            "query": query,
            "is_valid": not any('error' in r for r in validation_result),
            "errors": [r for r in validation_result if 'error' in r]
        })

    return {
        "has_queries": True,
        "queries": query_blocks,
        "validation_results": results
    }

def extract_years_from_question(question: str, cur_year: str) -> set:
    years = set()
    question_lower = question.lower()

    for match in re.findall(r'(19\d{2}|20\d{2})(?:년)?', question):
        years.add(match)

    for match in re.findall(r'(19\d{2}|20\d{2})[-/]\d{2}[-/]\d{2}', question):
        years.add(match)

    base_year = int(cur_year)

    mapping_ko = [
        ("재작년", str(base_year - 2)),
        ("작년",   str(base_year - 1)),
        ("올해",   str(base_year)),
    ]
    mapping_en = [
        ("the year before last", str(base_year - 2)),
        ("last year",            str(base_year - 1)),
        ("this year",            str(base_year)),
    ]

    q_ko = question
    q_en = question_lower

    def _mask(text: str, start: int, length: int) -> str:
        return text[:start] + (" " * length) + text[start+length:]

    for keyword, mapped in mapping_ko:
        i = 0
        while True:
            pos = q_ko.find(keyword, i)
            if pos == -1:
                break
            years.add(mapped)
            q_ko = _mask(q_ko, pos, len(keyword))
            i = pos + len(keyword)

    for keyword, mapped in mapping_en:
        i = 0
        while True:
            pos = q_en.find(keyword, i)
            if pos == -1:
                break
            years.add(mapped)
            q_en = _mask(q_en, pos, len(keyword))
            i = pos + len(keyword)

    return years

def patch_query_year_if_not_in_question(question: str, response_json: list, cur_year: str) -> list:
    question_years = extract_years_from_question(question, cur_year)
    patched = []

    for item in response_json:
        query = item.get("query", "")
        match = re.search(r'table\s+from=(\d{8,14})\s+to=([^\s]+)', query)
        if not match:
            patched.append(item)
            continue

        from_time = match.group(1)
        to_time = match.group(2)

        from_year = from_time[:4]

        to_year = to_time[:4] if re.match(r'^\d{8,14}$', to_time) else None

        to_is_variable = any(x in to_time for x in ["$CUR_DATE", "$CUR_TIME", "now", "latest", "today"])

        modified = False
        if len(question_years) == 0:
            new_from_time = f"{cur_year}{from_time[4:]}"
            query = query.replace(from_time, new_from_time, 1)
            modified = True

            if to_year and not to_is_variable and int(to_year) < int(cur_year):
                new_to_time = f"{cur_year}{to_time[4:]}"
                query = query.replace(to_time, new_to_time, 1)
                modified = True

        elif len(question_years) == 1:
            selected_year = list(question_years)[0]

            if from_year != selected_year:
                new_from_time = f"{selected_year}{from_time[4:]}"
                query = query.replace(from_time, new_from_time, 1)
                modified = True

            if to_year and not to_is_variable and int(to_year) < int(selected_year):
                new_to_time = f"{selected_year}{to_time[4:]}"
                query = query.replace(to_time, new_to_time, 1)
                modified = True

        elif len(question_years) == 2:
            sorted_years = sorted(int(y) for y in question_years)
            from_y, to_y = str(sorted_years[0]), str(sorted_years[1])

            if from_year != from_y:
                new_from_time = f"{from_y}{from_time[4:]}"
                query = query.replace(from_time, new_from_time, 1)
                modified = True

            if to_year and not to_is_variable and to_year != to_y:
                new_to_time = f"{to_y}{to_time[4:]}"
                query = query.replace(to_time, new_to_time, 1)
                modified = True


        if modified:
            item["query"] = query

        patched.append(item)

    return patched

def process_mixed_message(question: str, full_message: str, cur_year: str) -> str:
    try:
        prefix, json_text, suffix = extract_json_block(full_message)
        response_json = json.loads(json_text)
        patched_json = patch_query_year_if_not_in_question(question, response_json, cur_year)
        new_json_text = json.dumps(patched_json, ensure_ascii=False, indent=2)
        return f"{prefix}\n{new_json_text}\n{suffix}"
    except Exception as e:
        return full_message


def quick_syntax_fix_with_validation(response, validation_results):
    if not response:
        return response, False

    print("⏸️ 전처리 기능 비활성화 - JSON 파싱 에러 방지")

    has_json_error = False
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_text = json_match.group()
            json.loads(json_text)
            print("✅ JSON 구문 검사 통과")
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 구문 오류 발견: {e}")
        has_json_error = True

    return response, has_json_error

def has_json_list(text):
    candidates = re.findall(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return "action"
        except json.JSONDecodeError:
            continue

    return "qna"

async def get_corrected_query(llm_response, results, original_query, user=None, original_sources=None):
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    cur_year = current_time[:4]
    cur_date = current_time[:8]
    cur_time = current_time[8:]

    validation = Query_validation()
    query_validation = "### [쿼리 검증 결과 - 참고용, 절대 출력 금지]"
    
    for i, r in enumerate(results, 1):
        if 'error' in r:
            err = r['error']
            err_json = {}

            try:
                if not isinstance(err, str):
                    raise TypeError(f"error 필드는 문자열이어야 합니다: {err}")

                try:
                    err_json = json.loads(err)
                except json.JSONDecodeError:
                    if 'note=' in err:
                        note_part = err.split('note=')[-1]
                        err_json['note'] = note_part.strip()
                    else:
                        print(f"⚠️ note 키워드를 찾을 수 없습니다: {err}")

                query_str = r.get('query', '(쿼리 없음)')
                query_validation += f"\n{i}. * query: {query_str}"

                guide = translate_error_message(r)
                if guide:
                    query_validation += f"\n   * error: {guide}"
                else:
                    print(f"⚠️ 가이드 추가 실패: {r}")

            except TypeError as e:
                print(f"❌ Type 오류: {e}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
            except KeyError as e:
                print(f"❌ 키 누락 오류: {e}")
            except Exception as e:
                print(f"❌ 기타 처리 오류: {r}, 예외: {e}")

    print(query_validation)

    quick_fixed, has_json_error = quick_syntax_fix_with_validation(llm_response, results)
    if quick_fixed != llm_response:
        print(f"🔧 규칙 기반 빠른 수정 적용됨")
        print(f"🔍 [DEBUG] 수정 전: {llm_response[:100]}...")
        print(f"🔍 [DEBUG] 수정 후: {quick_fixed[:100]}...")

        has_json = '{' in quick_fixed and '}' in quick_fixed
        print(f"🔍 [DEBUG] JSON 블록 존재: {has_json}")

        if not has_json:
            print(f"❌ JSON 블록이 사라짐, 원본 응답 유지하고 LLM 수정으로 진행")
            quick_fixed = llm_response

        if has_json_error or not has_json:
            print(f"⚠️ JSON 구문 오류 또는 JSON 없음, LLM 수정으로 진행")
        else:
            validation_obj = Query_validation()
            quick_queries = validation_obj.extract_queries(quick_fixed)
            quick_results = []
            for query in quick_queries:
                quick_results.extend(validation_obj.validate_query(query))

            quick_errors = [r for r in quick_results if 'error' in r]
            if not quick_errors:
                print(f"✅ 규칙 기반 수정으로 모든 오류 해결 완료!")
                print(f"🔍 [DEBUG] 전달될 수정된 응답:\n{quick_fixed[:200]}...")

                has_final_json = '{' in quick_fixed and '}' in quick_fixed
                if not has_final_json:
                    print(f"❌ [CRITICAL] 최종 응답에 JSON 없음, 원본 유지")
                    quick_fixed = llm_response

                return {
                    "corrected_response": quick_fixed,
                    "fixed": True
                }
            else:
                print(f"⚠️ 일부 쿼리 오류 남음, LLM 수정으로 진행 ({len(quick_errors)}개 오류)")

    # llama 외 전 핸들러(gpt/gpt-oss/gemma/claude/qwen)는 gpt_val + system/user 포맷 공용.
    # 모델명별 {model}_val 키를 요구하면 새 핸들러마다 open(None) 크래시가 재발한다(qwen 사례).
    if DEFAULT_MODEL == "llama":
        prompt_name = f"{DEFAULT_MODEL}_val_{has_json_list(llm_response)}"
    else:
        prompt_name = "gpt_val"

    val_prompt_path = config.get_validation_config().get(prompt_name)

    import yaml
    with open(val_prompt_path, 'r', encoding='utf-8') as f:
        prompt_yaml = yaml.safe_load(f)
        if 'prompt' not in prompt_yaml:
            prompt_content = "You are a helpful assistant."
        prompt_content = prompt_yaml['prompt']

    if DEFAULT_MODEL == "llama":
        fix_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>"""
        fix_prompt += prompt_content

    else:
        system_prompt = prompt_content

        user_prompt = f"""다음 오류들을 수정해주세요:
ACTION 응답 원본:\n
{llm_response}

{query_validation}

** 중요 규칙 **:
- 위에 명시된 오류가 있는 쿼리만 수정하세요
- 오류가 없는 다른 쿼리나 설명은 절대 변경하지 마세요
"""
        fix_prompt = {"system": system_prompt, "user": user_prompt}


    current_model = llm_handler.model
    corrected_result = await llm_handler.process_question_error(
        fix_prompt, user, None, "INTERNAL", None
    )

    corrected_response = ""

    if corrected_result is None:
        print("❌ corrected_result가 None입니다.")
        return {
            "corrected_response": llm_response,
            "fixed": False
        }

    if not isinstance(corrected_result, dict):
        print(f"❌ 예상하지 못한 타입: {type(corrected_result)}")
        return {
            "corrected_response": llm_response,
            "fixed": False
        }
    if corrected_result is not None and isinstance(corrected_result, dict) and "answer_stream" in corrected_result:
        async for chunk in corrected_result["answer_stream"]:
            corrected_response += chunk
    elif "answer" in corrected_result:
        corrected_response = corrected_result["answer"]
    else:
        corrected_response = "⚠️ 응답 수정 중 오류가 발생했습니다. 원본 응답을 사용합니다."

    def validate_json_structure(response_text):
        try:
            json_matches = re.findall(r'\{[^{}]*"action"[^{}]*\}', response_text, re.DOTALL)
            if json_matches:
                for json_str in json_matches:
                    json.loads(json_str)
                return True

            return False
        except (json.JSONDecodeError, Exception):
            return False

    if not validate_json_structure(corrected_response):
        print("⚠️ 수정된 응답이 유효한 JSON을 포함하지 않습니다. 원본 응답을 사용합니다.")
        return {
            "corrected_response": llm_response,
            "fixed": False
        }

    return {
        "corrected_response": corrected_response,
        "fixed": True
    }


async def get_corrected_query_for_qna(llm_response, results, original_query, user=None, original_sources=None):
    """QnA 모드를 위한 쿼리 수정 함수 - get_corrected_query 기반, JSON 검증 제외"""
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    cur_year = current_time[:4]
    cur_date = current_time[:8]
    cur_time = current_time[8:]

    validation = Query_validation()
    query_validation = "### [쿼리 검증 결과 - 참고용, 절대 출력 금지]"

    for i, r in enumerate(results, 1):
        if 'error' in r:
            err = r['error']
            err_json = {}

            try:
                if not isinstance(err, str):
                    raise TypeError(f"error 필드는 문자열이어야 합니다: {err}")

                try:
                    err_json = json.loads(err)
                except json.JSONDecodeError:
                    if 'note=' in err:
                        note_part = err.split('note=')[-1]
                        err_json['note'] = note_part.strip()
                    else:
                        print(f"⚠️ note 키워드를 찾을 수 없습니다: {err}")

                query_str = r.get('query', '(쿼리 없음)')
                query_validation += f"\n{i}. * query: {query_str}"

                guide = translate_error_message(r)
                if guide:
                    query_validation += f"\n   * error: {guide}"
                else:
                    print(f"⚠️ 가이드 추가 실패: {r}")

            except TypeError as e:
                print(f"❌ Type 오류: {e}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
            except KeyError as e:
                print(f"❌ 키 누락 오류: {e}")
            except Exception as e:
                print(f"❌ 기타 처리 오류: {r}, 예외: {e}")
    
    print(query_validation)
    
    # action 쪽과 동일: llama 외 전 핸들러는 gpt_val 공용 (모델명별 키 크래시 방지)
    if DEFAULT_MODEL == "llama":
        prompt_name = "llama_val_qna"
    else:
        prompt_name = "gpt_val"

    val_prompt_path = config.get_validation_config().get(prompt_name)

    import yaml
    with open(val_prompt_path, 'r', encoding='utf-8') as f:
        prompt_yaml = yaml.safe_load(f)
        prompt_content = prompt_yaml.get('prompt', "You are a helpful assistant.")

    if DEFAULT_MODEL == "llama":
        fix_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>"""
        fix_prompt += prompt_content
    else:
        system_prompt = prompt_content
        user_prompt = f"""QnA 응답의 쿼리 블록에서 다음 오류들을 수정해주세요:
QnA 응답 원본:\n
{extract_query_blocks(llm_response)}

{query_validation}

** 중요 규칙 **:
- 위에 명시된 오류가 있는 쿼리만 수정하세요
- 오류가 없는 다른 쿼리나 설명은 절대 변경하지 마세요
- 쿼리 블록의 형식(```query ... ```)은 그대로 유지하세요"""
        fix_prompt = {"system": system_prompt, "user": user_prompt}
    
    current_model = llm_handler.model
    corrected_result = await llm_handler.process_question_error(
        fix_prompt, user, None, "INTERNAL", None
    )

    corrected_response = ""
    if corrected_result is None:
        print("❌ corrected_result가 None입니다.")
        return {
            "corrected_response": llm_response,
            "fixed": False
        }

    if not isinstance(corrected_result, dict):
        print(f"❌ 예상하지 못한 타입: {type(corrected_result)}")
        return {
            "corrected_response": llm_response,
            "fixed": False
        }

    if corrected_result is not None and isinstance(corrected_result, dict) and "answer_stream" in corrected_result:
        async for chunk in corrected_result["answer_stream"]:
            corrected_response += chunk
    elif "answer" in corrected_result:
        corrected_response = corrected_result["answer"]
    else:
        corrected_response = "⚠️ 응답 수정 중 오류가 발생했습니다. 원본 응답을 사용합니다."

    def validate_query_structure(response_text):
        try:
            query_blocks = extract_query_blocks(response_text)
            return len(query_blocks) > 0
        except Exception:
            return False

    if not validate_query_structure(corrected_response):
        print(corrected_response)
        print("⚠️ 수정된 응답이 유효한 query 블록을 포함하지 않습니다. 원본 응답을 사용합니다.")
        return {
            "corrected_response": llm_response,
            "fixed": False
        }

    return {
        "corrected_response": corrected_response,
        "fixed": True
    }


def process_time_variables(text, original_question=None):
    """시간 관련 변수를 처리하는 공통 함수"""
    if DEFAULT_MODEL not in ["llama", "gpt-oss", "gpt", "gemma"]:
        return text

    now = datetime.now()
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    cur_year = current_time[:4]

    # 시간 변수 치환
    text = re.sub(
        r"\$CUR_DATE\$\s*\$CUR_TIME\$|\$CURDATE\$\s*\$CURTIME\$|\$CUR_DATE\$|\$NOW|\${NOW}|\$\(now\)|<CURRENT_TIME>|now\(\)|\bnow\b",
        current_time,
        text
    )

    def replace_minus_nd(match):
        n = int(match.group(1))
        return (now - timedelta(days=n)).strftime('%Y%m%d') + '000000'

    text = re.sub(
        r"\$CUR_DATE_MINUS_(\d+)D\$",
        replace_minus_nd,
        text
    )
    # T가 포함된 날짜 형식 (예: 20260113T000000) -> T 제거 (20260113000000)
    text = re.sub(
        r'(\d{8})T(\d{6})',
        r'\1\2',
        text
    )

    # original_question이 제공되면 연도 처리도 수행
    if original_question:
        text = process_mixed_message(original_question, text, cur_year)

    return text

async def verify_and_fix_query_for_qna(
    query_response,
    original_question,
    qa_result=None,
    user=None,
    max_iterations=3
):
    # 시간 변수 처리 (공통 함수 사용)
    query_response = process_time_variables(query_response, original_question)

    original_response = query_response
    current_response = query_response

    validation = Query_validation()

    queries = extract_query_blocks(current_response)

    results = []
    all_errors = []
    for query in queries:
        result = validation.validate_query(query)
        results.extend(result)
        for item in result:
            if 'error' in item:
                command = item.get('command', item.get('query', ''))
                if command == '' or '[' in command or ']' in command:
                    continue
                all_errors.append({
                    "query": query,
                    "command": command,
                    "error": item.get('error', '알 수 없는 오류')
                })

    if not all_errors:
        print("✅ QnA 초기 검증 결과 오류 없음. 종료.")
        return {
            "fixed": False,
            "original_response": original_response,
            "corrected_response": current_response,
            "errors": []
        }

    print(f"🚨 QnA 오류 발생 쿼리 수정 진행")
    for iteration in range(max_iterations):
        print(f"🔁 QnA 쿼리 수정 요청 - 반복 {iteration + 1}/{max_iterations}")

        correction = await get_corrected_query_for_qna(
            current_response,
            results,
            original_question,
            user
        )

        if not correction.get("fixed", False):
            break

        current_response = replace_query_blocks_in_response(
            current_response,
            correction["corrected_response"]
        )

        queries = extract_query_blocks(current_response)

        results = []
        all_errors = []

        for query in queries:
            result = validation.validate_query(query)
            results.extend(result)
            for item in result:
                if 'error' in item:
                    all_errors.append({
                        "query": query,
                        "command": item.get('command', ''),
                        "error": item.get('error', '알 수 없는 오류')
                    })

        if not all_errors:
            print(f"✅ QnA 쿼리 수정 완료 - 반복 {iteration + 1}")
            return {
                "fixed": True,
                "original_response": original_response,
                "corrected_response": current_response,
                "errors": []
            }

    print(f"⚠️ {max_iterations}회 시도 후에도 QnA 쿼리 오류 존재")
    return {
        "fixed": False,
        "original_response": original_response,
        "corrected_response": current_response,
        "errors": all_errors
    }

async def verify_and_fix_query(
    llm_response,
    original_query,
    qa_result=None,
    user=None,
    only_time=False,
    max_iterations=3
):
    if original_query and ("[자동 조사]" in original_query or "워크플로우" in original_query):
        print("🔒 워크플로우 자동 조사 단계 - 쿼리 검증 건너뛰기")
        return {
            "fixed": False,
            "original_response": llm_response,
            "corrected_response": llm_response,
            "errors": []
        }

    # 시간 변수 처리 (공통 함수 사용)
    llm_response = process_time_variables(llm_response, original_query)

    if only_time:
        return {
        "fixed": False,
        "original_response": llm_response,
        "corrected_response": llm_response,
        "errors": []
    }

    original_response = llm_response
    current_response = llm_response

    validation = Query_validation()
    queries = validation.extract_queries(current_response)
    results = []
    all_errors = []

    for query in queries:
        result =  validation.validate_query(query)
        results.extend(result)
        for item in result:
            if 'error' in item:
                command = item.get('command', item.get('query', ''))
                if command == '' or '[' in command or ']' in command:
                    continue
                all_errors.append({
                    "query": query,
                    "command": command,
                    "error": item.get('error', '알 수 없는 오류')
                })

    if not all_errors:
        print("✅ 초기 검증 결과 오류 없음. 종료.")
        return {
            "fixed": False,
            "original_response": original_response,
            "corrected_response": current_response,
            "errors": []
        }

    print(f"🚨 오류 발생 쿼리 수정 진행")
    for iteration in range(max_iterations):
        print(f"🔁 쿼리 수정 요청 - 반복 {iteration + 1}/{max_iterations}")

        correction = await get_corrected_query(
            current_response,
            results,
            original_query,
            user
        )
        current_response = correction["corrected_response"]
        try:
            data = json.loads(current_response)
            q = data.get("query", original_query)
        except json.JSONDecodeError as e:
            q = original_query

        queries = validation.extract_queries(q)
        results = []
        all_errors = []

        for query in queries:
            result = validation.validate_query(query)
            results.extend(result)
            for item in result:
                if 'error' in item:
                    all_errors.append({
                        "query": query,
                        "command": item.get('command', ''),
                        "error": item.get('error_command', '알 수 없는 오류')
                    })

        if not all_errors:
            print("✅ 오류 모두 수정됨. 종료.")

            return {
                "fixed": True,
                "original_response": original_response,
                "corrected_response": current_response,
                "errors": []
            }

    print(f"⛔ 최대 반복({max_iterations}) 후에도 오류 남음.")

    return {
        "fixed": True,
        "original_response": original_response,
        "corrected_response": current_response,
        "errors": all_errors
    }


def verify_query(llm_response):
    validation = Query_validation()
    queries = validation.extract_queries(llm_response)
    for i, query in enumerate(queries, 1):
        result = validation.validate_query(query)
        errors = [item for item in result if 'error' in item]

        for error_item in errors:
            print(f"오류 내용: {error_item['error']}")
            print("-" * 50)


async def clean_cache():
    while True:
        try:
            current_time = time.time()
            expired_keys = []

            for key, item in response_cache.items():
                # timestamp 키가 있는 항목만 만료 체크
                if isinstance(item, dict) and 'timestamp' in item:
                    if current_time - item['timestamp'] > CACHE_EXPIRY:
                        expired_keys.append(key)

            for key in expired_keys:
                del response_cache[key]

            if expired_keys:
                print(f"캐시에서 {len(expired_keys)}개 항목 제거됨")

        except Exception as e:
            print(f"캐시 정리 중 오류: {str(e)}")

        await asyncio.sleep(3600)

async def clean_chat_sessions():
    while True:
        try:
            if hasattr(llm_handler, "clean_old_sessions"):
                result = llm_handler.clean_old_sessions()
                if asyncio.iscoroutine(result):
                    removed_count = await result
                else:
                    removed_count = result

                if removed_count and removed_count > 0:
                    print(f"오래된 채팅 세션 {removed_count}개 제거됨")
            else:
                print("🧹 정리할 세션이 없습니다")

        except Exception as e:
            print(f"채팅 세션 정리 중 오류: {str(e)}")

        await asyncio.sleep(10800)


@api_app.post("/api/query/stream", description="질의응답 스트리밍 API",tags=["질의응답"])
async def handle_query_stream(request: QueryRequest, user_info: Dict = Depends(get_bearer_api_key_user)):
    if request.query_type not in user_info["permissions"]:
        raise HTTPException(status_code=403, detail=f"{request.query_type} 쿼리에 대한 권한이 없습니다.")

    model = DEFAULT_MODEL
    llm_handler.set_model(model)

    cached_result = get_cached_response(request.query, model)

    async def generate():
        try:
            if cached_result:
                logger = ChatLogger(model)
                logger.log(
                    user_info['user_id'],
                    request.query,
                    cached_result['answer'],
                    "0",
                    cached_result.get('sources', []),
                    [],
                    [],
                    'API',
                    True
                )

                chat_session = llm_handler.get_chat_session(user_info['user_id'], "API")
                chat_session.add_message("user", request.query)
                chat_session.add_message("assistant", cached_result['answer'])

                yield f"data: {json.dumps({'chunk': cached_result['answer']})}\n\n"
                if 'sources' in cached_result:
                    yield f"data: {json.dumps({'sources': cached_result['sources']})}\n\n"
                return

            print(f"질문 처리 중 (사용자: {user_info['user_id']}): '{request.query[:50]}...'")
            start_time = time.time()

            # FR-10: 토큰 단위 스트리밍 — [server] stream_tokens 스위치.
            # 기본 false: 구 배포는 "완성 후 1프레임 + usage" 계약을 그대로 유지한다.
            # true 면 핸들러 generate_stream 을 타서 조각이 생기는 대로 chunk 프레임을 내보낸다.
            # usage 프레임은 유지되지만, 스트림 경로는 핸들러가 토큰을 집계하지 않으므로
            # 0 으로 온다(집계가 필요하면 stream_tokens=false 또는 후속 과제).
            if STREAM_TOKENS:
                handler = llm_handler.handlers.get(model)
                if handler is not None:
                    # 이전 complete 호출의 스테일 usage 가 프레임에 실리지 않게 초기화
                    from handler_base import UsageTracker
                    handler._usage_tracker = UsageTracker(model=getattr(handler, 'model_name', model))

                # process_question 은 구독 콘솔 계약이라 sub_id 필수(검증·로그에만 쓰임) —
                # API 경로에서는 인증 사용자 id 로 채운다. response_mode=streaming 으로
                # [options] response=complete 전역 기본을 이 엔드포인트에서만 덮는다.
                qa_result = await llm_handler.process_question(
                    {"question": request.query,
                     "sub_id": user_info.get("sub_id") or user_info.get("user_id", "api"),
                     "response_mode": "streaming"},
                    user_info['user_id'],
                    "API",
                    'API'
                )

                full_response = ""
                if "answer_stream" in qa_result:
                    async for chunk in qa_result["answer_stream"]:
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                else:
                    # 핸들러가 complete 형태를 돌려준 경우(오버라이드 미지원 커스텀 등) —
                    # 종전 계약(1프레임)으로 폴백
                    full_response = qa_result.get("answer", "")
                    yield f"data: {json.dumps({'chunk': full_response})}\n\n"

                sources = list(getattr(handler, 'last_sources', []) or []) if handler else []
                if not sources:
                    sources = qa_result.get("context_sources", []) or []
                if sources:
                    yield f"data: {json.dumps({'sources': sources})}\n\n"

                usage = handler._usage_tracker.to_dict() if handler is not None else {}
                yield f"data: {json.dumps({'usage': usage})}\n\n"

                end_time = time.time()
                logger = ChatLogger(model)
                logger.log(
                    user_info['user_id'],
                    request.query,
                    full_response,
                    f"{end_time - start_time:.2f}",
                    sources,
                    getattr(handler, 'last_concepts', []) if handler else [],
                    getattr(handler, 'last_related_concepts', []) if handler else [],
                    'API'
                )
                store_in_cache(request.query, model, {'answer': full_response, 'sources': sources})
                print(f"응답 완료(스트리밍): 소요시간 {time.time() - start_time:.2f}초")
                return

            try:
                complete_result = await llm_handler.get_complete_answer(
                    request.query,
                    user_info['user_id'],
                    "API",
                    'API'
                )

                full_response = complete_result.get("answer", "")
                sources = complete_result.get("sources", [])

                yield f"data: {json.dumps({'chunk': full_response})}\n\n"

                if sources:
                    yield f"data: {json.dumps({'sources': sources})}\n\n"

                usage = complete_result.get("usage", {})
                yield f"data: {json.dumps({'usage': usage})}\n\n"

                end_time = time.time()

                logger = ChatLogger(model)
                logger.log(
                    user_info['user_id'],
                    request.query,
                    full_response,
                    f"{end_time - start_time:.2f}",
                    sources,
                    [],
                    [],
                    'API'
                )

                cache_data = {
                    'answer': full_response,
                    'sources': sources
                }
                store_in_cache(request.query, model, cache_data)

            except AttributeError as e:
                print(f"get_complete_answer 메서드를 찾을 수 없음, process_question으로 대체: {str(e)}")

                qa_result = await llm_handler.process_question(
                    request.query,
                    user_info['user_id'],
                    "API",
                    'API'
                )

                if "answer_stream" in qa_result:
                    full_response = ""
                    async for chunk in qa_result["answer_stream"]:
                        full_response += chunk

                    yield f"data: {json.dumps({'chunk': full_response})}\n\n"

                    sources = qa_result.get("context_sources", [])
                    if sources:
                        yield f"data: {json.dumps({'sources': sources})}\n\n"

                    end_time = time.time()

                    handler = llm_handler.handlers.get(model) or llm_handler.handlers.get("llama")

                    logger = ChatLogger(model)
                    logger.log(
                        user_info['user_id'],
                        request.query,
                        full_response,
                        f"{end_time - start_time:.2f}",
                        sources,
                        getattr(handler, 'last_concepts', []),
                        getattr(handler, 'last_related_concepts', []),
                        'API'
                    )

                    cache_data = {
                        'answer': full_response,
                        'sources': sources
                    }
                    store_in_cache(request.query, model, cache_data)
                else:
                    raise ValueError("LLM 응답에 answer_stream이 없습니다.")

            print(f"응답 완료: 소요시간 {time.time() - start_time:.2f}초")

        except Exception as e:
            print(f"응답 생성 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _js_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2

def _js_offset_to_py_index(s: str, offset_js: int) -> int:
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        units = len(s[:mid].encode("utf-16-le")) // 2
        if units < offset_js:
            lo = mid + 1
        else:
            hi = mid
    return lo

def _js_slice(s: str, offset_js: int, count_js: int) -> tuple[str, int, int]:
    total_js = _js_len(s)
    if offset_js < 0:
        offset_js = 0
    if offset_js >= total_js:
        return "", offset_js, total_js

    start_py = _js_offset_to_py_index(s, offset_js)
    end_js = min(offset_js + max(0, int(count_js)), total_js)
    end_py = _js_offset_to_py_index(s, end_js)

    return s[start_py:end_py], end_js, total_js

response_cache = {}
sb_registry = StreamBufferRegistry(js_len_fn=_js_len, js_slice_fn=_js_slice)

def store_in_assistant_cache(key: str, model: str, data: dict):
    response_cache[(key, model)] = data

def get_assistant_cached_response(key: str, model: str):
    return response_cache.get((key, model))

def delete_asssistant_cached_response(key: str, model: str):
    response_cache[(key, model)]["delete"] = True

async def set_streambuffer_response(guid: str, qtype: str, query: str) -> None:
    await sb_registry.create(guid, qtype, query)

def get_streambuffer_response(guid: str) -> dict | None:
    sb = sb_registry.get(guid)
    return sb.meta() if sb else None

async def delete_streambuffer_response(guid: str) -> bool:
    sb = sb_registry.get(guid)
    if not sb:
        return False
    await sb_registry.remove(guid)
    return True

def clean_query_in_mixed_input(input_text: str) -> str:
    try:
        match = re.search(r'(\[\s*\{.*?\}\s*\])', input_text, re.DOTALL)
        is_array = True

        if not match:
            match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', input_text, re.DOTALL)
            is_array = False
            if match:
                print("📄 단일 JSON 객체 발견")

        if not match:
            return input_text

        json_part = match.group(1)
        other_part = input_text.replace(json_part, "")

        data = json.loads(json_part)

        items_to_process = [data] if not is_array else data

        for item in items_to_process:
            if isinstance(item, dict) and "query" in item and isinstance(item["query"], str):
                original_query = item["query"]
                query = original_query.lstrip()

                if query.startswith("|"):
                    query = query[1:].lstrip()

                query = query.replace('\\', '')
                query = query.replace('""', '"')

                item["query"] = query

        cleaned_json = json.dumps(data, ensure_ascii=False, indent=2)

        return f"{cleaned_json}\n{other_part.strip()}" if other_part.strip() else cleaned_json

    except json.JSONDecodeError as e:
        print(f"[오류] JSON 파싱 실패: {e}")
        return input_text
    except Exception as e:
        print(f"[오류] 예상치 못한 오류: {e}")
        return input_text

@api_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("📛 Validation failed for:", request.url)
    print("❌ Validation detail:", exc.errors())
    print("📦 Received body:", await request.body())

    return await request_validation_exception_handler(request, exc)

@api_app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    print("🚫 HTTPException 발생!")
    print(f"📍 URL: {request.url}")
    print(f"📡 Status code: {exc.status_code}")
    print(f"📄 Detail: {exc.detail}")

    print("📦 Headers:")
    for key, value in request.headers.items():
        print(f"  - {key}: {value}")

    try:
        body = await request.body()
        print(f"📝 Body: {body.decode('utf-8')}")
    except Exception as e:
        print(f"❗ 바디 읽기 실패: {e}")

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@api_app.post("/api/ai/playbooks")
async def api_playbook_generation(
    request: AssistantRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    user_guid = request.user_guid

    playbook_request = {
        "type": "playbook",
        "question": request.question,
        "history": request.messages or [],
        "response_format": request.response_format
    }

    answer_result = await llm_handler.get_complete_answer(
        playbook_request,
        user_info['user_id'],
        "API",
        user_guid
    )

    full_answer = answer_result.get("answer", "")
    intent = answer_result.get("intent", "playbook")
    sources = answer_result.get("sources", [])

    usage = answer_result.get("usage", {})

    return {
        "answer": full_answer,
        "intent": intent,
        "sources": sources,
        "playbook_type": "security_response",
        "usage": usage
    }

@api_app.post("/api/ai/chats/regression")
async def api_regression_response(
    request: AssistantRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    question_type = request.type
    user_guid = request.user_guid

    print("응답 형식: "+str(request.response_format))

    if request.response_format == None:
        response_format = True
    else:
        response_format = request.response_format

    answer_result = await llm_handler.get_complete_answer(
        {
            "type": question_type,
            "question": request.question,
            "history": request.messages,
            "response_format": response_format,
            "sub_id": user_info['id']
        },
        user_info['user_id'],
        "API",
        user_guid
    )

    full_answer = answer_result.get("answer", "")
    raw_answer = answer_result.get("raw_answer", "")
    intent = answer_result.get("intent", question_type)
    sources = answer_result.get("sources", [])
    response_time = answer_result.get("response_time", 0)
    question_type_result = answer_result.get("question_type", question_type)
    print(f"[REGRESSION DEBUG] keys={list(answer_result.keys())}, sources={sources}, intent={intent}, response_time={response_time}")

    usage = answer_result.get("usage", {})

    return {
        "answer": full_answer,
        "raw_answer": raw_answer,
        "intent": intent,
        "sources": sources,
        "response_time": response_time,
        "question_type": question_type_result,
        "usage": usage
    }

@api_app.post("/api/ai/queue", description="추가 질의", tags=["AI 어시스턴트"])
def additional_queries(
    request: QueueRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    try:
        intent = request.intent
        conv_guid = request.conversation_guid
        chat_guid = request.chat_guid
        updated_queue = request.post_action_response['updated_queue']
    except Exception as e:
        print(f"❌ 요청 데이터 파싱 실패: {e}")
        return {'result': 'failed'}

    try:
        validation_config = config.get_validation_config()
        api_key = validation_config['api_key']
        base_url = validation_config['base_url']
        url = base_url + validation_config["queue_path"].format(chat_guid=chat_guid)
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        session.verify = False
    except Exception as e:
        print(f"❌ API 세션 설정 실패: {e}")
        return {'result': 'failed'}

    for question in updated_queue:
        print(f"🔍 추가 질문{type(question)}: {question}")

        payload = {
            "intent": "action",
            "question": str(question)
        }
        print(f"Payload: {payload}")
        try:
            response = session.post(url, params=payload)
            response.raise_for_status()
            print(response.json())
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 실패: {e}")
            return {'result': 'failed'}
        finally:
            session.close()

    return {'result' : 'success'}

# ── OpenAI 호환 경로의 RAG 모드 스위치 ──────────────────────────
# 기본 OpenAI 호환 엔드포인트(/agent/chat/completions)는 원래 handler.agent_complete 로
# **직결**됐다 — messages 를 chat template 으로 변환해 LLM 에 그대로 넘기는 경로라
# RAG·카트리지 프롬프트·intent 분류가 전부 우회된다(2026-07-27 실측: prompt_len 136자).
# 즉 외부 서비스가 이 경로로 붙으면 "도메인 에이전트"가 아니라 생 LLM 을 쓰게 된다.
#
# agent_rag = true 면 이 엔드포인트가 /api/query/stream 과 **같은 파이프라인**
# (intent → PII → RAG 검색 → 카트리지 프롬프트)을 타도록 전환한다.
# 기본값 false — 기존 배포(구 콘솔 경로로 붙어 있는 소비자)의 동작을 바꾸지 않기 위함이다.
# 신규 설치는 config.ini.template 에서 true 로 온다.
# /agent/chat/completions2 는 이름(Direct)과 계약대로 **항상 직결**이며 이 스위치와 무관하다.
try:
    AGENT_RAG = config.config.getboolean('server', 'agent_rag', fallback=False)
except Exception:
    AGENT_RAG = False
print(f"🔍 [agent] OpenAI 호환 경로 RAG 모드: {'ON (카트리지 적용)' if AGENT_RAG else 'OFF (직결)'}")

# FR-10: /api/query/stream 토큰 단위 스트리밍 스위치 — agent_rag 와 같은 도입 방식
# (기본 false 로 구 배포 계약 보존, 신규 설치는 template 이 true).
try:
    STREAM_TOKENS = config.config.getboolean('server', 'stream_tokens', fallback=False)
except Exception:
    STREAM_TOKENS = False
print(f"🔍 [stream] /api/query/stream 토큰 스트리밍: {'ON' if STREAM_TOKENS else 'OFF (완성 후 1프레임)'}")

@api_app.post("/agent/chat/completions", description="AI Agent용 OpenAI 호환 API", tags=["AI Agent"])
async def agent_chat_completions(
    request: Dict[str, Any],
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    try:
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages are required")

        model = DEFAULT_MODEL
        llm_handler.set_model(model)
        print(f"🔍 [completions] 모델: {model} · RAG={'ON' if AGENT_RAG else 'OFF'}")

        response_text = None
        model_name = model
        usage = {}
        sources = []

        if AGENT_RAG:
            # 전체 파이프라인 경유 — /api/query/stream 과 동일한 진입점.
            # messages 의 마지막 user 발화를 질문으로, 그 앞을 history 로 넘긴다
            # (history 형식이 OpenAI messages 와 동일해 변환이 필요 없다).
            last_user = next((m for m in reversed(messages)
                              if (m or {}).get("role") == "user"), None)
            if last_user is None:
                raise HTTPException(status_code=400, detail="user 역할 메시지가 필요합니다")
            question = (last_user.get("content") or "").strip()
            if not question:
                raise HTTPException(status_code=400, detail="빈 질문입니다")
            history = [m for m in messages if m is not last_user]

            # sub_id 는 RAG 검색의 필수 인자다(없으면 handler 가 "sub_id가 필요합니다"로
            # 검색만 실패하고 프롬프트는 걸려, 카트리지 지식 없이 답하는 상태가 된다 —
            # 조용히 품질만 떨어지는 종류라 2026-07-27 실측에서야 드러났다).
            result = await llm_handler.get_complete_answer(
                {"question": question, "history": history, "sub_id": user_info['id']},
                user_info['user_id'], "API", 'API',
            ) or {}
            response_text = result.get("answer", "")
            model_name = (result.get("usage") or {}).get("model") or model
            usage = result.get("usage", {})
            sources = result.get("sources", []) or []

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }],
                "usage": usage,
                # OpenAI 표준 밖 필드 — 표준 클라이언트는 무시한다. RAG 근거 추적용.
                "sources": sources,
            }

        # 모델 무관 위임 — completions2 와 동일하게 handler.agent_complete 로 라우팅한다.
        # (per-model 하드코딩 대신 활성 핸들러가 프롬프트 조립·생성을 캡슐화 — 새 모델 자동 지원)
        handler = llm_handler.handlers.get(model)
        if handler is None or not getattr(handler, "available", False):
            raise HTTPException(status_code=503, detail=f"{model} 모델을 사용할 수 없습니다.")
        try:
            result = await handler.agent_complete(messages, {})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # 핸들러 미로드(local_model 미부착 등) → 500 아닌 503 으로 통일
            raise HTTPException(status_code=503, detail=f"{model} 모델을 사용할 수 없습니다: {e}")
        response_text = (result or {}).get("content", "")
        model_name = (result or {}).get("model", model)
        usage = (result or {}).get("usage", {})

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": usage
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요청 처리 실패: {str(e)}")

@api_app.post("/agent/chat/completions2", description="AI Agent용 OpenAI 호환 API (Direct)", tags=["AI Agent"])
async def agent_chat_completions2(
    body: Dict[str, Any],
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    """OpenAI 호환 직접 프롬프트 전달 API. 모델별 프롬프트 조립·생성은 핸들러의 agent_complete 가 캡슐화한다(새 모델 추가 시 이 엔드포인트 수정 불필요). model/length/reasoning_effort 는 request body > 구독 설정 > config 디폴트 순서로 결정."""
    try:
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages are required")

        # 3-tier 폴백: request body > 구독 설정 > config 디폴트 (빈 문자열은 None 처리)
        requested_model = _resolve_setting(
            body.get("model") or None,
            user_info.get("model"),
            DEFAULT_MODEL,
        ).strip()
        routing_target, openai_model_override = _resolve_routing(requested_model)
        print(f"🔍 [completions2] 요청 모델: {requested_model} → 핸들러: {routing_target}" +
              (f" (override={openai_model_override})" if openai_model_override else ""))

        handler = llm_handler.handlers.get(routing_target)
        if handler is None:
            raise HTTPException(status_code=503, detail=f"{requested_model} 모델을 사용할 수 없습니다. (핸들러 미로드)")

        # 모델별 프롬프트 조립·생성은 핸들러가 캡슐화한다 (build_agent_prompt / agent_complete).
        # → 새 모델을 추가해도 이 엔드포인트는 수정 불필요. 핸들러만 등록하면 된다.
        # 옵션(length/reasoning_effort)은 모델 무관하게 전달: 로컬 모델은 무시, OpenAI류만 사용.
        options = {
            "length": _resolve_setting(body.get("length"), user_info.get("length"), None),
            "reasoning_effort": _resolve_setting(body.get("reasoning_effort"), user_info.get("reasoning_effort"), None),
            "locale": _resolve_setting(body.get("locale"), user_info.get("locale"), None),
            "model_override": openai_model_override,
        }

        # -- stream:true 전용 additive 분기 -- 지원 핸들러(gemma)만. 비스트림(dict) 경로는 무변경.
        if body.get("stream", False) is True and hasattr(handler, "agent_stream"):
            def _sse_gen():
                cid = f"chatcmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": routing_target}
                first = dict(base, choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
                yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                try:
                    for piece in handler.agent_stream(messages, options):
                        if not piece:
                            continue
                        ch = dict(base, choices=[{"index": 0, "delta": {"content": piece}, "finish_reason": None}])
                        yield f"data: {json.dumps(ch, ensure_ascii=False)}\n\n"
                except Exception as _se:
                    _err = dict(base, choices=[{"index": 0, "delta": {"content": f"[stream error] {_se}"}, "finish_reason": "stop"}])
                    yield f"data: {json.dumps(_err, ensure_ascii=False)}\n\n"
                done = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_sse_gen(), media_type="text/event-stream")
        try:
            result = await handler.agent_complete(messages, options)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as rerr:
            detail = str(rerr)
            raise HTTPException(status_code=(503 if "미로드" in detail else 502), detail=detail)

        response_text = result["content"]
        model_name = result["model"]
        usage = result["usage"]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": usage
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요청 처리 실패: {str(e)}")

# ── 구 배포 호환 alias — config [server] agent_alias_prefix (예: 구 콘솔 경로 접두어) 설정 시
#    {prefix}/agent/chat/completions{,2} 경로를 동일 핸들러로 추가 등록한다. 카트리지/배포가 정의.
try:
    _agent_alias_prefix = config.config.get('server', 'agent_alias_prefix', fallback='').strip()
    if _agent_alias_prefix:
        api_app.add_api_route(f"{_agent_alias_prefix}/agent/chat/completions",
                              agent_chat_completions, methods=["POST"], tags=["AI Agent"])
        api_app.add_api_route(f"{_agent_alias_prefix}/agent/chat/completions2",
                              agent_chat_completions2, methods=["POST"], tags=["AI Agent"])
except Exception as _e:
    print(f"⚠️ agent alias 라우트 등록 실패: {_e}")

@api_app.get("/api/ai/hello", description="세션확인",tags=["AI 어시스턴트"])
def hello(
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    result = {
        "is_valid": True
    }
    return result

@api_app.post("/api/ai/summarize", description="제목 요약", tags=["AI 어시스턴트"])
async def generate_summarize(
    request: SummaryRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    stream_guid = str(uuid.uuid4())
    model = request.llm_model or DEFAULT_MODEL
    llm_handler.set_model(model)

    title_request = {
        "type": "TITLE",
        "question": request.question,
        "history": [],
        "response_format": False,
        "locale": request.locale,
    }

    usage = {}
    try:
        answer_result = await llm_handler.get_complete_answer(
            title_request, user_info['user_id'], "API", stream_guid
        )
        print(f"[TITLE] locale={request.locale}, answer={answer_result.get('answer', '')}")
        answer_text = answer_result.get("answer", request.question)
        usage = answer_result.get("usage", {})
        if not answer_text:
            answer_text = request.question
    except Exception as e:
        print(f"[TITLE] LLM 호출 실패: {e}, 원본 질문 사용")
        answer_text = request.question

    cache_data = {
        "answer": answer_text,
        "query": request.question
    }
    store_in_assistant_cache(stream_guid, DEFAULT_MODEL, cache_data)
    return JSONResponse(content={"guid": stream_guid, "usage": usage})

@api_app.post("/api/ai/validate", description="쿼리 검증 및 자동 수정", tags=["AI 어시스턴트"])
async def validate_query_api(
    query: str = Body(..., description="검증할 쿼리"),
    max_iterations: int = Body(3, description="최대 수정 반복 횟수"),
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    """
    쿼리를 검증하고 오류가 있을 경우 자동으로 수정합니다.

    Returns:
        {
            "success": bool,
            "original_query": str,
            "final_query": str,
            "is_valid": bool,
            "was_fixed": bool,
            "iterations": int,
            "errors": list,
            "validation_history": list
        }
    """
    # 시간 변수 처리
    if DEFAULT_MODEL in ["llama", "gpt-oss", "gpt", "gemma"]:
        now_dt = datetime.now()
        current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        query = re.sub(
            r"\$CUR_DATE\$\s*\$CUR_TIME\$|\$CUR_DATE\$|\$NOW|\${NOW}|\$\(now\)|<CURRENT_TIME>|now\(\)|\bnow\b",
            current_time,
            query
        )

        def replace_minus_nd(match):
            n = int(match.group(1))
            return (now_dt - timedelta(days=n)).strftime('%Y%m%d') + '000000'

        query = re.sub(
            r"\$CUR_DATE_MINUS_(\d+)D\$",
            replace_minus_nd,
            query
        )

    original_query = query
    current_query = query
    validation_history = []

    # 초기 검증
    validation = Query_validation()
    results = validation.validate_query(current_query)
    errors = []

    for item in results:
        if 'error' in item:
            command = item.get('command', item.get('query', ''))
            if command and '[' not in command and ']' not in command:
                errors.append({
                    "query": current_query,
                    "command": command,
                    "error": item.get('error', '알 수 없는 오류'),
                    "offset": item.get('offset', 0)
                })

    if not errors:
        return {
            "success": True,
            "original_query": original_query,
            "final_query": current_query,
            "is_valid": True,
            "was_fixed": False,
            "iterations": 0,
            "errors": [],
            "validation_history": ["✅ 쿼리에 오류가 없습니다."]
        }

    # 오류가 있으면 수정 시도
    for iteration in range(max_iterations):
        validation_history.append(f"반복 {iteration + 1}: {len(errors)}개 오류 발견")

        # LLM을 통한 수정 (ACTION JSON 형식으로 감싸서 전달)
        json_wrapped = json.dumps({"action": "query", "query": current_query}, ensure_ascii=False)
        correction_result = await get_corrected_query(
            json_wrapped,
            results,
            original_query,
            ""
        )

        if "corrected_response" not in correction_result:
            validation_history.append(f"반복 {iteration + 1}: LLM 수정 실패")
            break

        corrected_response = correction_result["corrected_response"]
        try:
            data = json.loads(corrected_response)
            current_query = data.get("query", current_query)
        except json.JSONDecodeError:
            pass

        # 수정된 쿼리 재검증
        results = validation.validate_query(current_query)
        errors = []

        for item in results:
            if 'error' in item:
                command = item.get('command', item.get('query', ''))
                if command and '[' not in command and ']' not in command:
                    errors.append({
                        "query": current_query,
                        "command": command,
                        "error": item.get('error', '알 수 없는 오류'),
                        "offset": item.get('offset', 0)
                    })

        if not errors:
            validation_history.append(f"✅ 반복 {iteration + 1}: 모든 오류 수정 완료")
            return {
                "success": True,
                "original_query": original_query,
                "final_query": current_query,
                "is_valid": True,
                "was_fixed": True,
                "iterations": iteration + 1,
                "errors": [],
                "validation_history": validation_history
            }

    # 최대 반복 후에도 오류가 있는 경우
    validation_history.append(f"⚠️ {max_iterations}회 수정 시도 후에도 {len(errors)}개 오류 남음")
    return {
        "success": False,
        "original_query": original_query,
        "final_query": current_query,
        "is_valid": False,
        "was_fixed": True,
        "iterations": max_iterations,
        "errors": errors,
        "validation_history": validation_history
    }

@api_app.post("/api/ai/chats", description="질문을 받아 답변 생성 및 저장", tags=["AI 어시스턴트"])
async def generate_response(
    request: AssistantRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    from uuid import uuid4
    user_guid = str(uuid4())
    question_type = request.type

    question = request.question
    if not question and request.messages:
        for msg in reversed(request.messages):
            if msg.get("role") == "user" and msg.get("content"):
                question = msg.get("content")
                break

    print(f"🔍 [API] 추출된 질문: {question}")
    print(f"🔍 [API] 질문 타입: {question_type}, locale: {request.locale}")

    # 3-tier model 폴백 (request body > ai_subscriptions.model > DEFAULT_MODEL) + 라우팅 결정
    requested_model = _resolve_setting(None, user_info.get("model"), DEFAULT_MODEL).strip()
    routing_target, model_override = _resolve_routing(requested_model)
    model = requested_model  # 캐시/로그 용
    print(f"🔍 [API/chats] sub_model={user_info.get('model')!r} → routing={routing_target} override={model_override}")

    global llm_handler
    if llm_handler is None:
        print("⚠️ llm_handler가 None입니다. 새로 생성합니다.")
        from aibot_llm_module import LLMHandler
        llm_handler = LLMHandler(
            prompt_type=DEFAULT_MODEL,
            use_rag=True,
            file_extension="yaml",
            use_kg=True,
            use_db=use_db_mode
        )

    apps = None
    if isinstance(request.installed_apps, list):
        apps = json.dumps(request.installed_apps, ensure_ascii=False)

    profiles = None
    if isinstance(request.connect_profiles, list):
        profiles = json.dumps(request.connect_profiles, ensure_ascii=False)

    llm_handler.set_model(routing_target)

    if question_type == "":
        answer_result = await llm_handler.get_streaming_answer(
            {"type":question_type, "question":question, "history":request.messages, "response_format": request.response_format if request.response_format is not None else True, "apps": apps, "sub_id": user_info['id'], "locale": request.locale, "model_override": model_override}, user_info['user_id'], "API", user_guid
        )
        stream_answer = answer_result.get("answer_stream", "")

        sb = await sb_registry.create(user_guid, question_type, request.question)

        def _is_async_iter(obj): return hasattr(obj, "__aiter__")
        def _is_sync_iter(obj):
            return hasattr(obj, "__iter__") and not _is_async_iter(obj) and not isinstance(obj, (str, bytes))

        async def _ensure_async_iter(obj):
            if _is_async_iter(obj):
                async for x in obj:
                    yield x
            elif _is_sync_iter(obj):
                for x in obj:
                    yield x
            else:
                if obj:
                    yield obj

        async def consume():
            micro_buf = []
            last_chunk = None

            async def _flush_micro():
                nonlocal micro_buf
                if micro_buf:
                    joined = "".join(micro_buf)
                    micro_buf = []
                    if joined and joined != last_chunk:
                        await sb.append(joined)

            try:
                async for chunk in _ensure_async_iter(stream_answer):
                    if not chunk:
                        continue

                    if chunk == last_chunk:
                        continue
                    last_chunk = chunk

                    if len(chunk) < 4:
                        micro_buf.append(chunk)
                        if sum(len(c) for c in micro_buf) >= 32:
                            await _flush_micro()
                    else:
                        await _flush_micro()
                        await sb.append(chunk)

            except asyncio.CancelledError:
                await sb.fail("cancelled")
                raise
            except Exception as e:
                await sb.fail(f"stream_error: {e}")
            finally:
                try:
                    await _flush_micro()
                finally:
                    if not sb.state.done:
                        await sb.finish()
                    await sb.to_cache(DEFAULT_MODEL)

        asyncio.create_task(consume())
        return {"guid": user_guid, "intent": question_type, "usage": {}}
    else:
        answer_result = await llm_handler.get_complete_answer(
            {"type":question_type, "question":question, "history":request.messages, "response_format": request.response_format if request.response_format is not None else True, "apps": apps, "sub_id": user_info['id'], "locale": request.locale, "model_override": model_override}, user_info['user_id'], "API", user_guid
        )
        full_answer = answer_result.get("answer", "")
        intent = answer_result.get("intent")
        sources = answer_result.get("sources", [])
        print(full_answer)
        if question_type == "action":
            verification_result = await verify_and_fix_query(
                full_answer, request.question, answer_result, user_info['user_id'], False
            )
            if verification_result['fixed']:
                answer_text = verification_result['corrected_response']
            else:
                answer_text = verification_result['original_response']
        elif question_type == "qna":
            verification_result = await verify_and_fix_query_for_qna(
                full_answer, request.question, answer_result, user_info['user_id']
            )
            if verification_result['fixed']:
                answer_text = verification_result['corrected_response']
                print(f"✅ QnA 쿼리 수정 완료")
            else:
                answer_text = verification_result['original_response']
        else:
            answer_text = full_answer

        if question_type == "action":
            answer_text = clean_query_in_mixed_input(answer_text)

            # JSON 파싱 검증 — 실패 시 LLM에게 JSON 재요청
            try:
                json_match = re.search(r'\{.*\}', answer_text, re.DOTALL)
                if json_match:
                    json.loads(json_match.group())
            except json.JSONDecodeError:
                print("⚠️ ACTION 응답 JSON 파싱 실패, LLM에게 JSON 재요청")
                try:
                    fix_system_prompt = (
                        "당신은 JSON 수정 전문가입니다. 입력된 텍스트에서 깨진 JSON을 올바른 형식으로 수정하세요.\n\n"
                        "규칙:\n"
                        "- 유효한 JSON만 출력하세요\n"
                        "- 코드블록(```), 설명, 부연 문구 등 JSON 외의 텍스트는 절대 출력하지 마세요\n"
                        "- JSON 키와 값의 내용은 변경하지 마세요\n"
                        "- 구문 오류(쉼표, 따옴표, 괄호 등)만 수정하세요\n"
                        "- 출력 형식: {\"action\": \"query\", \"query\": \"...\"}"
                    )
                    if DEFAULT_MODEL == "gpt":
                        fix_system_prompt += (
                            "\n\n중요: 반드시 JSON만 출력하세요. "
                            "```json 코드블록, 마크다운, 설명 텍스트 등을 절대 포함하지 마세요. "
                            "첫 번째 문자가 { 이고 마지막 문자가 } 인 순수 JSON만 출력하세요."
                        )
                    fix_prompt = {
                        "system": fix_system_prompt,
                        "user": f"{answer_text}"
                    }
                    fix_result = await llm_handler.process_question_error(
                        fix_prompt, user_info['user_id'], None, "INTERNAL", None
                    )
                    if fix_result and isinstance(fix_result, dict):
                        fixed_text = ""
                        if "answer_stream" in fix_result:
                            async for chunk in fix_result["answer_stream"]:
                                fixed_text += chunk
                        elif "answer" in fix_result:
                            fixed_text = fix_result["answer"]

                        if fixed_text:
                            try:
                                fix_json_match = re.search(r'\{.*\}', fixed_text, re.DOTALL)
                                if fix_json_match:
                                    json.loads(fix_json_match.group())
                                    answer_text = fixed_text
                                    print("✅ JSON 재요청으로 파싱 오류 복구 완료")
                            except json.JSONDecodeError:
                                print("⚠️ JSON 재요청 후에도 파싱 실패, 원본 유지")
                except Exception as e:
                    print(f"⚠️ JSON 재요청 중 오류: {e}")

        print(answer_text)
        usage = answer_result.get("usage", {})
        cache_data = {
            "answer": answer_text,
            "query": request.question,
            "type" : question_type,
            "delete": False,
        }

        # 캐시 키는 DEFAULT_MODEL 로 고정 — GET/DELETE /api/ai/stream/{guid} 가 항상
        # DEFAULT_MODEL 로 조회하므로, 라우팅 모델(requested_model)로 저장하면
        # 비-default 모델(furiosa gpt 등) 응답이 404 로 안 잡힌다. guid 가 uuid4 로 고유해
        # model 네임스페이스는 불필요. (기존 multi-model 라우팅 도입 시 발생한 키 불일치 fix)
        store_in_assistant_cache(user_guid, DEFAULT_MODEL, cache_data)

        return {"guid": user_guid, "intent": intent, "usage": usage}

@api_app.get("/api/ai/stream/{guid}", description="오프셋 기반 응답 조각 반환", tags=["AI 어시스턴트"])
async def get_cached_chunk(
        guid: str,
        offset: int = Query(0),
        user_info: Dict = Depends(get_bearer_api_key_user)
    ):
    sb = sb_registry.get(guid)
    if sb:
        qtype = sb.state.type
        count_js = 100
        return await sb.get_chunk(offset, count_js)

    cache_data = get_assistant_cached_response(guid, DEFAULT_MODEL)

    if not cache_data:
        raise HTTPException(status_code=404, detail="해당 응답이 존재하지 않습니다.")

    question_type = cache_data.get("type", "")
    answer = cache_data.get("answer", "")

    count_js = 100 if question_type == "qna" else len(answer)

    chunk, next_offset_js, total_js = _js_slice(answer, offset_js=offset, count_js=count_js)
    return {
        "seq": next_offset_js,
        "value": chunk,
        "is_finished": next_offset_js >= total_js,
    }


@api_app.delete("/api/ai/stream/{guid}", description="응답 캐시 삭제", tags=["AI 어시스턴트"])
async def delete_cached_response(
    guid: str,
    user_info: Dict = Depends(get_bearer_api_key_user),
):
    model = DEFAULT_MODEL
    sb = sb_registry.get(guid)
    if sb:
        sb.delete_cache(guid, model)
        await sb_registry.remove(guid)
        return {
            "message": f"{guid} 삭제 완료",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "embedding_tokens": 0
        }

    key = (guid, model)

    cache_data = response_cache.get(key)
    if not cache_data:
        raise HTTPException(status_code=404, detail="삭제할 응답 캐시가 없습니다.")

    delete_asssistant_cached_response(guid, model)

    return {
        "message": f"{guid} 삭제 완료",
        "prompt_tokens": cache_data.get("prompt_tokens", 0),
        "completion_tokens": cache_data.get("completion_tokens", 0),
        "embedding_tokens": cache_data.get("embedding_tokens", 0)
    }


@api_app.get("/api/ai/prompts", description="", tags=["AI 어시스턴트"])
async def get_promptslist(
    req: Request,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    return {"prompts": []}


# ============================================================
# 싱크 엔드포인트 (메인 서버 역할 시 사용)
# — Qdrant I/O는 동기 블로킹이므로 run_in_executor로 스레드풀 위임
#   → LLM 요청 처리하는 이벤트 루프를 막지 않음
# ============================================================
def _sync_manifest_blocking():
    """(동기) sub_id=1 경량 manifest 수집"""
    from aibot_prompts_functions import _get_qdrant_client
    from qdrant_client import models as qd_models

    client = _get_qdrant_client()
    items = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=qdrant_collection(),
            scroll_filter=qd_models.Filter(
                must=[
                    qd_models.FieldCondition(key="sub_id", match=qd_models.MatchValue(value=1)),
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        points, next_offset = result
        for pt in points:
            p = pt.payload or {}
            guid = p.get("guid")
            if not guid:
                continue
            items.append({
                "guid": guid,
                "rev": p.get("rev", 0),
                "content_hash": p.get("content_hash", ""),
                "enabled": p.get("enabled", True),
                "doc_type": p.get("doc_type", "qna"),
            })
        if next_offset is None:
            break
        offset = next_offset

    return {"items": items, "total": len(items)}


def _sync_points_blocking(guids: list):
    """(동기) guid 목록의 포인트를 벡터 포함 수집"""
    from aibot_prompts_functions import _get_qdrant_client
    from qdrant_client import models as qd_models

    client = _get_qdrant_client()
    points_out = []

    for guid in guids:
        result = client.scroll(
            collection_name=qdrant_collection(),
            scroll_filter=qd_models.Filter(
                must=[
                    qd_models.FieldCondition(key="sub_id", match=qd_models.MatchValue(value=1)),
                    qd_models.FieldCondition(key="guid", match=qd_models.MatchValue(value=guid)),
                ]
            ),
            with_payload=True,
            with_vectors=True,
            limit=1,
        )
        pts, _ = result
        if not pts:
            continue

        pt = pts[0]
        vectors_out = {}

        # dense
        if hasattr(pt.vector, 'get'):
            raw_dense = pt.vector.get("dense")
        elif isinstance(pt.vector, dict):
            raw_dense = pt.vector.get("dense")
        else:
            raw_dense = None
        if raw_dense is not None:
            vectors_out["dense"] = list(raw_dense) if not isinstance(raw_dense, list) else raw_dense

        # colbert
        raw_colbert = pt.vector.get("colbert") if isinstance(pt.vector, dict) else None
        if raw_colbert is not None:
            vectors_out["colbert"] = raw_colbert

        # sparse
        raw_sparse = pt.vector.get("sparse") if isinstance(pt.vector, dict) else None
        if raw_sparse is not None:
            if hasattr(raw_sparse, 'indices'):
                vectors_out["sparse"] = {
                    "indices": list(raw_sparse.indices),
                    "values": list(raw_sparse.values),
                }
            elif isinstance(raw_sparse, dict):
                vectors_out["sparse"] = raw_sparse

        points_out.append({
            "id": pt.id,
            "vectors": vectors_out,
            "payload": pt.payload,
        })

    return {"points": points_out, "total": len(points_out)}


@api_app.get("/api/ai/sync/manifest", description="sub_id=1 프롬프트 경량 목록 (싱크용)", tags=["싱크"])
async def sync_manifest(req: Request, user_info: Dict = Depends(get_bearer_api_key_user)):
    """벡터 없이 guid, rev, content_hash, enabled, doc_type 만 반환"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_manifest_blocking)


@api_app.post("/api/ai/sync/points", description="guid 목록에 해당하는 벡터+payload 전체 반환 (싱크용)", tags=["싱크"])
async def sync_points(req: Request, user_info: Dict = Depends(get_bearer_api_key_user)):
    """요청된 guid 목록의 포인트를 벡터 포함하여 반환"""
    body = await req.json()
    guids = body.get("guids", [])
    if not guids:
        return {"points": []}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_points_blocking, guids)


@api_app.post("/api/ai/prompts", description="새로운 프롬프트를 생성", tags=["AI 어시스턴트"])
async def edit_prompt(
    req: Request,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    if config.get_server_config().get('saas_mode'):
        raise HTTPException(status_code=403, detail="SaaS 모드에서는 AI 프롬프트 기능을 지원하지 않습니다.")

    body = await req.json()
    name = body['name']
    prompt_type = body['type']
    description = body['description']
    prompt_text = body['prompt']

    # 파일 경로 기반 결정적 GUID 생성 (같은 type/name → 항상 같은 GUID)
    from aibot_prompts_functions import generate_file_guid
    file_key = f"{prompt_type.lower()}/{name}.yaml"
    guid = body.get('guid', generate_file_guid(file_key))

    # sub_id override
    if 'sub_id' in body:
        user_info = {**user_info, 'id': int(body['sub_id'])}

    prompt = parse_prompt(prompt_type, name, description, prompt_text, guid)
    try:
        data = create_prompt(user_info, guid, prompt, DEFAULT_MODEL, llm_handler)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    token_count = [v.token_count for v in data.values()]

    return {
        'guid': guid,
        'token_count': token_count[0]
    }

@api_app.post("/api/ai/prompts/bulk", description="YAML 파일 대량 업로드 (임베딩 + Qdrant 저장)", tags=["AI 어시스턴트"])
async def bulk_upload_prompts(
    files: List[UploadFile] = File(..., description="YAML 파일들"),
    sub_id: int = Form(default=1, description="구독 ID"),
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    if config.get_server_config().get('saas_mode'):
        raise HTTPException(status_code=403, detail="SaaS 모드에서는 AI 프롬프트 기능을 지원하지 않습니다.")

    from aibot_prompts_functions import generate_file_guid, parse_prompt, create_prompt

    # sub_id override
    user_info = {**user_info, 'id': sub_id}

    results = []
    created = 0
    updated = 0
    failed = 0

    for file in files:
        filename = file.filename or ""
        try:
            content = await file.read()
            prompt_text = content.decode('utf-8')

            # 파일명에서 type/name 추출 (예: "qna/my-topic.yaml" 또는 "my-topic.yaml")
            parts = filename.replace("\\", "/").split("/")
            if len(parts) >= 2:
                prompt_type = parts[-2]  # qna, action, plan
                name = parts[-1].replace(".yaml", "").replace(".yml", "")
            else:
                # 파일명만 있으면 YAML 내부에서 type 추출 시도
                import yaml as _yaml
                try:
                    parsed = _yaml.safe_load(prompt_text)
                    prompt_type = parsed.get('type', 'qna') if isinstance(parsed, dict) else 'qna'
                except:
                    prompt_type = 'qna'
                name = parts[-1].replace(".yaml", "").replace(".yml", "")

            file_key = f"{prompt_type.lower()}/{name}.yaml"
            guid = generate_file_guid(file_key)

            # YAML에서 description 추출
            description = ""
            try:
                import yaml as _yaml
                parsed = _yaml.safe_load(prompt_text)
                if isinstance(parsed, dict):
                    description = parsed.get('description', '')
            except:
                pass

            prompt = parse_prompt(prompt_type, name, description, prompt_text, guid)
            data = create_prompt(user_info, guid, prompt, DEFAULT_MODEL, llm_handler, skip_reload=True)
            token_count = [v.token_count for v in data.values()][0]

            # Qdrant에서 기존 문서 존재 여부로 created/updated 판별
            results.append({"file": file_key, "guid": guid, "status": "ok", "token_count": token_count})
            created += 1

        except Exception as e:
            results.append({"file": filename, "status": "failed", "error": str(e)})
            failed += 1
            print(f"❌ [Bulk] {filename} 처리 실패: {e}")

    # 전체 완료 후 RAG 리로드 (1회)
    try:
        llm_handler.reload_embedding(sub_id=sub_id)
        print(f"✅ [Bulk] RAG 리로드 완료 (sub_id={sub_id})")
    except Exception as e:
        print(f"⚠️ [Bulk] RAG 리로드 실패: {e}")

    return {
        "total": len(files),
        "created": created,
        "failed": failed,
        "details": results
    }


@api_app.delete("/api/ai/prompts", description="", tags=["AI 어시스턴트"])
async def delete_prompt(
    req: Request,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    if config.get_server_config().get('saas_mode'):
        raise HTTPException(status_code=403, detail="SaaS 모드에서는 AI 프롬프트 기능을 지원하지 않습니다.")

    import re
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)

    body = await req.json()
    guids = body['guids']
    guid_list = [g.strip() for g in guids.split(',') if UUID_PATTERN.match(g.strip())]

    if not guid_list:
        return {"message": "No valid GUIDs provided"}

    # sub_id override
    if 'sub_id' in body:
        user_info = {**user_info, 'id': int(body['sub_id'])}

    sub_id = user_info['id']
    # DB는 대화 로깅 전용 — use_db_mode=False(경량/우회)면 건너뛴다. 업로드(bulk)는 DB를
    # 안 쓰는데 삭제만 DB를 강제하던 비대칭 버그로 DB-off 콘솔에서 500 발생 (cartridge unmount).
    if config.get_db_config().get('use_db_mode', 'False').lower() == 'true':
        db_manager.remove_prompt(sub_id, guid_list)
    remove_prompt(sub_id, DEFAULT_MODEL, llm_handler, guid_list=guid_list)
    return

@api_app.get("/api/ai/prompts/{guid}", description="", tags=["AI 어시스턴트"])
async def get_prompt(
    guid: str,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    return {"prompts": []}

@api_app.post("/api/ai/prompts/{guid}/{toggle}", description="프롬프트 토글", tags=["AI 어시스턴트"])
async def set_prompt(
    guid: str,
    toggle: str,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    if config.get_server_config().get('saas_mode'):
        raise HTTPException(status_code=403, detail="SaaS 모드에서는 AI 프롬프트 기능을 지원하지 않습니다.")

    sub_id = user_info['id']
    enabled = toggle == 'enable'

    db_manager.toggle_prompt(sub_id, guid, enabled)

    # BGE 모드일 때 Qdrant payload도 업데이트
    if _is_bge_mode():
        toggle_prompt_bge(sub_id, guid, enabled)

    return

@api_app.put("/api/ai/prompts/{guid}", description="프롬프트를 정보를 수정", tags=["AI 어시스턴트"])
async def edit_prompt(
    guid: str,
    req: Request,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    if config.get_server_config().get('saas_mode'):
        raise HTTPException(status_code=403, detail="SaaS 모드에서는 AI 프롬프트 기능을 지원하지 않습니다.")

    body = await req.json()
    name = body['name']
    description = body['description']
    prompt_type = body['type']
    prompt_text = body['prompt']

    prompt = parse_prompt(prompt_type, name, description, prompt_text, guid)

    # BGE 모드일 때 업데이트 함수 사용
    try:
        if _is_bge_mode():
            data = update_prompt_bge(user_info, guid, prompt, llm_handler)
        else:
            data = create_prompt(user_info, guid, prompt, DEFAULT_MODEL, llm_handler)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    token_count = [v.token_count for v in data.values()]

    return {
        'token_count': token_count[0] if token_count else 0,
        'rev': 1
    }

# 티켓 메모 생성 API
@api_app.post("/api/ai/ticket/memo", description="티켓 메모 생성", tags=["AI 어시스턴트"])
async def create_ticket_memo(
    req: TicketRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    """ req 입력 예시
    reference_id=[
        {
            'guid': '8d086dfb-28d5-44ad-8f2c-18ef0d0e8b83', 
            'id': 4, 
            'title': 'qwe', 
            'priority': 'LOW', 
            'status': 'NEW', 
            'repo_name': '위협분석', 
            'count': 1, 
            'content': 'qwe', 
            'format': 'MARKDOWN', 
            'created': '2025-12-17 15:17:03+0900', 
            'updated': '2025-12-17 15:17:03+0900', 
            'raw_data': [], 티켓 관련 로우 데이터
            'reference_tickets': [], 정/오탐 사례 1건씩
        }]
    """
    """
    class TicketRequest(BaseModel):
        user_guid: str
        type: str
        llm_model: Optional[str] = DEFAULT_MODEL
        locale: str
        reference_id: List[Dict]
    """
    ticket_info = req.reference_id[0] if req.reference_id else {}

    # raw_data와 reference_tickets 포맷팅 함수
    def format_raw_data(raw_data):
        if not raw_data:
            return "로그 데이터 없음"

        if isinstance(raw_data, list):
            return "\n".join([str(item) for item in raw_data])
        elif isinstance(raw_data, str):
            return raw_data
        else:
            return str(raw_data)

    def format_reference_tickets(reference_tickets):
        if not reference_tickets:
            return "참조 티켓 없음"

        if isinstance(reference_tickets, list):
            formatted_refs = []
            for i, ref in enumerate(reference_tickets, 1):
                if isinstance(ref, dict):
                    formatted_refs.append(f"참조문서({i}): {ref.get('guid', 'Unknown')}")
                    formatted_refs.append(str(ref))
                else:
                    formatted_refs.append(f"참조문서({i}): {str(ref)}")
            return "\n\n".join(formatted_refs)
        else:
            return str(reference_tickets)

    config_manager = ConfigManager()
    ticket_memo_prompt_path = config_manager.get_prompt_config().get('ticket_memo')

    try:
        import yaml
        with open(ticket_memo_prompt_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
            system_prompt = prompt_config.get('prompt', '')
    except Exception as e:
        print(f"프롬프트 파일 로드 실패: {e}")
        system_prompt = "보안 티켓을 분석하고 메모를 작성해주세요."

    # 티켓 데이터 구성
    ticket_csv_data = (
        f'"{ticket_info.get("id", "")}",'
        f'"{ticket_info.get("priority", "LOW")}",'
        f'"{ticket_info.get("guid", "")}",'
        f'"{ticket_info.get("title", "")}",'
        f'"{ticket_info.get("status", "NEW")}",'
        f'"{ticket_info.get("attack", "false")}",'
        f'"{ticket_info.get("incident", "")}",'
        f'"{ticket_info.get("count", "1")}",'
        f'"{ticket_info.get("repo_name", "위협분석")}",'
        f'"{ticket_info.get("format", "MARKDOWN")}",'
        f'"{ticket_info.get("assignees", "[]")}",'
        f'"{ticket_info.get("approvers", "[]")}",'
        f'"{ticket_info.get("tags", "")}",'
        f'"{ticket_info.get("created", "")}",'
        f'"{ticket_info.get("updated", "")}",'
        f'"{ticket_info.get("closed", "")}",'
        f'"{ticket_info.get("repo_guid", "")}"'
    )

    # 로그 데이터 구성
    raw_data_formatted = format_raw_data(ticket_info.get('raw_data', []))

    # 참조 티켓 구성
    reference_tickets_formatted = format_reference_tickets(ticket_info.get('reference_tickets', []))

    # CSV 헤더
    csv_header = "id,priority,guid,title,status,attack,incident,count,repo_name,format,assignees,approvers,tags,created,updated,closed,repo_guid"

    # 전체 데이터 섹션 구성
    ticket_data_section = (
        f"### [티켓 데이터]\n"
        f"{csv_header}\n"
        f"{ticket_csv_data}\n"
        f"\n"
        f"### [근거 로그 데이터]\n"
        f"{raw_data_formatted}\n"
        f"\n"
        f"{reference_tickets_formatted}"
    )

    prompt = f"{system_prompt}\n\n{ticket_data_section}"
    try:
        # 활성 로컬 모델 선택 (DEFAULT_MODEL=gemma 면 gemma, 아니면 gpt-oss) — 둘 다 generate_complete 인터페이스 동일
        # 활성 모델(DEFAULT_MODEL)의 local_model 사용 — 없으면 로드된 아무 로컬 핸들러로 폴백.
        # 모든 로컬 모델(llama/gpt-oss/gemma/qwen)이 generate_complete 인터페이스를 공유한다.
        active = llm_handler.handlers.get(DEFAULT_MODEL)
        if not (active and getattr(active, "local_model", None)):
            active = next((h for h in llm_handler.handlers.values()
                           if getattr(h, "local_model", None)), None)
        if active is None:
            raise RuntimeError("로컬 모델 핸들러를 찾을 수 없습니다 (ticket_memo)")
        local_model = active.local_model
        raw_response = local_model.generate_complete(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.1
        )

        # <|message|> 태그에서 메모 내용 파싱
        if "<|message|>" in raw_response:
            message_start = raw_response.find("<|message|>") + len("<|message|>")
            memo_text = raw_response[message_start:].strip()
            print(f"✅ message 태그 발견! 메모 추출: {memo_text}")
        else:
            memo_text = raw_response.strip()
            print(f"⚠️ message 태그 없음 - 에러: {memo_text}")
            return {"type": "MARKDOWN", "memo": f"", "_error": f"티켓 메모 생성에 실패하였습니다. (응답 형식 오류)"}

        print(f"최종 메모: {memo_text}")
        return {"type": "MARKDOWN", "memo": memo_text}
    except Exception as e:
        print(f"메모 생성 중 오류: {e}")
        return {"type": "MARKDOWN", "memo": f"", "_error": f"티켓 메모 생성 중 오류가 발생했습니다: {str(e)}"}

# API 키 생성 및 관리 라우트들
@api_app.post("/api/generate", description="새 API 키 등록 (관리자 권한 필요)", tags=["인증"])
async def generate_api_key(request: Request, generate_request: GenerateApiKeyRequest, background_tasks: BackgroundTasks):
    """새로운 API 키를 생성하고 기본 구독 데이터를 복사한 후 메모리에 즉시 로드"""
    # 클라이언트 IP 가져오기 (필요시 구현)
    client_ip = getattr(request.client, 'host', 'unknown')

    # 관리자 인증 (security-review.md S-3) — 전역 인증 미들웨어가 없으므로 여기서 막는다.
    # 이게 없으면 누구나 유효기간 1년짜리 키를 스스로 발급받을 수 있었다.
    if not verify_admin_key(generate_request.admin_key):
        raise HTTPException(status_code=401, detail="관리자 인증이 필요합니다.")

    name = generate_request.name
    account = generate_request.account
    description = generate_request.description
    acl = generate_request.acl
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.000")
    expires_at = (now + timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S.000")

    guid = str(uuid.uuid4())
    api_key = str(uuid.uuid4())

    # DB 매니저 가져오기
    from aibot_db_manager import AibotDBManager
    from aibot_db_command import SQL_QUERIES

    db_manager = AibotDBManager(
        config=config,
        query_properties=SQL_QUERIES
    )

    insert_sql = """
        INSERT INTO ai_subscriptions (guid, name, account, description, api_key, acl,
                                       model, length, reasoning_effort,
                                       created_at, updated_at, expires_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(insert_sql, (
                    guid, name, account, description, api_key, acl,
                    generate_request.model, generate_request.length, generate_request.reasoning_effort,
                    formatted_now, formatted_now, expires_at
                ))
                sub_id = cursor.lastrowid
            conn.commit()

        # 기본 구독에서 FAISS와 지식그래프 복사 + 즉시 메모리 로드
        background_tasks.add_task(copy_base_embedding_to_subscription, sub_id)

        print(f'✅ API 키 생성 완료: sub_id={sub_id}, name={name}')

        return {
            "message": "Subscription created successfully",
            "guid": guid,
            "api_key": api_key,
            "sub_id": sub_id,
            "created_at": formatted_now,
            "expires_at": expires_at,
            "embedding_status": "processing",
            "note": "임베딩 데이터가 복사되고 메모리에 자동으로 로드됩니다."
        }
    except Exception as e:
        print(f'❌ API 키 생성 실패: {e}')
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")

# ─────────────────────────────────────────────────────────────
# 카트리지 런타임 반영 (Phase 4) — 장착 자체는 CLI/위저드가 하고, 여기서는 "반영"만 한다.
# 이게 없으면 config.ini 를 고쳐도 돌고 있는 핸들러가 옛 프롬프트를 계속 써서 재시작이 필요했다.
# ─────────────────────────────────────────────────────────────
class CartridgeAdminRequest(BaseModel):
    admin_key: str = Field(..., min_length=32, max_length=64, description="관리자 키 (api_keys/admin.key)")


@api_app.post("/api/cartridge/reload", description="config.ini [prompts] 배선을 런타임 재적용 (관리자 권한 필요)", tags=["카트리지"])
async def cartridge_reload_api(req: CartridgeAdminRequest):
    """장착/해제를 끝낸 쪽이 호출하는 '반영' 훅 — 장착 자체는 하지 않는다.

    장착 경로는 두 개다: CLI(`aibotctl cartridge mount`, 로컬 실행)와
    위저드(`/api/wizard/cartridge-mount`). 둘 다 콘솔이 죽어 있어도 장착은 되어야 하므로
    배선·적재는 각자 하고, 콘솔이 떠 있을 때만 이 훅으로 런타임에 반영한다.
    이 호출이 성공하면 `restart_required` 는 False 다.
    """
    if not verify_admin_key(req.admin_key):
        raise HTTPException(status_code=401, detail="관리자 인증이 필요합니다.")
    return reload_cartridge_prompts()


@api_app.post("/api/reload", description="구독의 임베딩 데이터를 메모리에 리로드", tags=["관리"])
async def reload_subscription_embedding_api(request: Request, reload_request: dict):
    """특정 구독의 임베딩 데이터를 메모리에 리로드"""
    try:
        api_key = reload_request.get("api_key")
        if not api_key:
            raise HTTPException(status_code=400, detail="API 키가 필요합니다.")

        # DB 매니저 가져오기
        from aibot_db_manager import AibotDBManager
        from aibot_db_command import SQL_QUERIES

        db_manager = AibotDBManager(
            config=config,
            query_properties=SQL_QUERIES
        )

        # API 키로 구독 ID 찾기
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, name FROM ai_subscriptions WHERE api_key = %s", (api_key,))
                row = cursor.fetchone()

                if not row:
                    raise HTTPException(status_code=404, detail="해당 API 키가 존재하지 않습니다.")

                sub_id = row["id"]
                sub_name = row["name"]

        # 임베딩 리로드 실행
        success = reload_subscription_embedding(sub_id)
        if not success:
            raise HTTPException(status_code=500, detail="임베딩 리로드에 실패했습니다. AI 시스템이 초기화되지 않았을 수 있습니다.")

        return {
            "message": f"구독 '{sub_name}' (ID: {sub_id})의 임베딩 데이터가 성공적으로 리로드되었습니다.",
            "sub_id": sub_id,
            "sub_name": sub_name
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f'❌ 리로드 API 오류: {e}')
        raise HTTPException(status_code=500, detail="리로드 중 서버 오류가 발생했습니다.")

@api_app.get("/init", description="프로그램 구동 최초 설정", tags=["시스템"])
async def init(user_info: Dict = Depends(get_bearer_api_key_user)):
    if db_manager.initialize_tables():
        from datetime import datetime, timedelta

        guid = str(uuid.uuid4())
        api_key = str(uuid.uuid4())
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.000")
        expires_at = (now + timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S.000")

        insert_sql = """
            INSERT INTO ai_subscriptions (guid, name, account, description, api_key, acl, created_at, updated_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, (
                        guid, "default", "default", "default", api_key, None,
                        formatted_now, formatted_now, expires_at
                    ))
                    sub_id = cursor.lastrowid
                conn.commit()
            return {"result": "success",
                    "sub_id": sub_id
                    }
        except Exception as e:
            return {"result": "fail",
                    "sub_id": None
                    }

    return {"result": "fail",
            "sub_id": None
            }

@api_app.get("/", description="API 상태 확인", tags=["시스템"])
async def root():

    models_info = {key: False for key in HANDLER_CLASSES.keys() if key != "qwen"}  # qwen 은 root 응답에서 historically 미노출

    if llm_handler:
        for key in models_info:
            h = llm_handler.handlers.get(key)
            if h and hasattr(h, 'available'):
                models_info[key] = h.available

    search_info = {
        "rag_enabled": False,
        "knowledge_graph_enabled": False,
        "enhanced_search_available": False,
        "search_strategy": "unknown",
        "cache_enabled": False
    }

    if llm_handler and hasattr(llm_handler, 'rag_system') and llm_handler.rag_system:
        search_info["rag_enabled"] = True

    try:
        search_status = None

        if llm_handler:
            # HANDLER_CLASSES 우선순위 (gpt → gpt-oss → qwen → gemma → llama) 첫 가용 핸들러
            for key in HANDLER_CLASSES.keys():
                h = llm_handler.handlers.get(key)
                if h is not None and hasattr(h, 'get_search_status'):
                    search_status = h.get_search_status()
                    break

        if search_status:
            search_info.update({
                "knowledge_graph_enabled": search_status.get('knowledge_graph_enabled', False),
                "enhanced_search_available": search_status.get('enhanced_search_available', False),
                "search_strategy": search_status.get('search_config', {}).get('search_strategy', 'unknown'),
                "cache_enabled": search_status.get('search_config', {}).get('cache_enabled', False)
            })

    except Exception as e:
        print(f"검색 정보 수집 실패: {e}")

    cache_stats = {
        "response_cache_size": 0,
        "chat_sessions": 0
    }

    try:
        if 'response_cache' in globals():
            cache_stats["response_cache_size"] = len(response_cache)

        if (llm_handler and
            hasattr(llm_handler, 'chat_sessions') and
            llm_handler.chat_sessions):
            cache_stats["chat_sessions"] = len(llm_handler.chat_sessions)
    except Exception as e:
        print(f"캐시 통계 수집 실패: {e}")

    default_model = "unknown"
    try:
        if 'DEFAULT_MODEL' in globals():
            default_model = DEFAULT_MODEL
    except:
        pass

    return {
        "status": "online",
        "service": "RAG & 지식 그래프 기반 AI 질의응답 API",
        "version": "1.0",
        "default_model": default_model,
        "models": models_info,
        "features": {
            "rag": search_info["rag_enabled"],
            "knowledge_graph": search_info["knowledge_graph_enabled"],
            "enhanced_search": search_info["enhanced_search_available"],
            "caching": True
        },
        "search_system": search_info,
        "cache_stats": cache_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# ============================================================
# RAG 검색 전용 API (AGENT용)
# ============================================================
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="검색 쿼리")
    search_type: str = Field(default="QNA", description="검색 타입: QNA, ACTION, PLAN")
    top_k: int = Field(default=5, ge=1, le=20, description="반환할 결과 수")


@api_app.post("/api/search", description="RAG 검색 (참조문서 검색)", tags=["검색"])
async def search_documents(
    request: SearchRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    """
    RAG 시스템을 통한 참조문서 검색
    적재된 지식(쿼리 문법·예제·가이드 등)을 검색합니다.
    """
    try:
        # 검색용 핸들러: 활성 모델(DEFAULT_MODEL) 우선, 없으면 gpt-oss/OpenAI 폴백
        handler = (llm_handler.handlers.get(DEFAULT_MODEL)
                   or llm_handler.handlers.get("gpt-oss")
                   or llm_handler.handlers.get("gpt"))

        if not handler or not handler.rag_system:
            raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")

        # sub_id 고정 (내부 시스템용)
        sub_id = 2

        # RAG 검색 수행 - handler의 get_related_documents 사용
        rag_request = {
            "question": request.query,
            "type": request.search_type.upper(),
            "sub_id": sub_id
        }

        result = await handler.get_related_documents(rag_request)

        # 결과 파싱
        context = ""
        source_files = []
        related_concepts = []
        concepts = []

        if len(result) >= 1:
            context = result[0] or ""
        if len(result) >= 2:
            source_files = result[1] or []
        if len(result) >= 3:
            related_concepts = result[2] or []
        if len(result) >= 4:
            concepts = result[3] or []

        return {
            "success": True,
            "query": request.query,
            "search_type": request.search_type,
            "context": context,
            "sources": source_files[:request.top_k] if source_files else [],
            "related_concepts": related_concepts,
            "concepts": concepts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "query": request.query,
            "context": "",
            "sources": []
        }


@api_app.get("/api/search/status", description="검색 시스템 상태 확인", tags=["검색"])
async def get_search_status(user_info: Dict = Depends(get_bearer_api_key_user)):
    try:
        if "admin" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

        status_result = {}
        # API 응답 키는 legacy 속성명 유지 (openai_handler / gptoss_handler / ...)
        for legacy_attr, key in LEGACY_ATTR_TO_KEY.items():
            h = llm_handler.handlers.get(key)
            if h:
                status_result[legacy_attr] = h.get_search_status()

        rag_stats = {}
        if llm_handler.rag_system:
            rag_stats = llm_handler.rag_system.get_search_stats()

        return {
            "status": "success",
            "search_system": {
                **status_result,
                "rag_system": rag_stats
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 상태 확인 실패: {str(e)}")


@api_app.post("/api/search/config", description="검색 설정 변경", tags=["검색"])
async def update_search_config(
    config_update: Dict[str, Any],
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    try:
        if "admin" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

        allowed_keys = ["search_strategy", "cache_enabled"]

        updated_configs = {}
        # search_config 업데이트는 search 가능한 핸들러 모두 (gpt/gpt-oss/llama 가 historically 대상)
        for key, value in config_update.items():
            if key in allowed_keys:
                for model_key in ("gpt", "gpt-oss", "llama"):
                    h = llm_handler.handlers.get(model_key)
                    if h is not None and hasattr(h, 'search_config'):
                        h.search_config[key] = value
                        updated_configs[key] = value

        if updated_configs:
            return {
                "status": "success",
                "message": "검색 설정이 업데이트되었습니다.",
                "updated_configs": updated_configs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            return {
                "status": "warning",
                "message": "업데이트된 설정이 없습니다.",
                "allowed_keys": allowed_keys
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 설정 업데이트 실패: {str(e)}")

async def clean_search_cache():
    while True:
        try:
            if hasattr(llm_handler, 'rag_system') and llm_handler.rag_system:
                if hasattr(llm_handler.rag_system, 'clear_search_cache'):
                    await asyncio.sleep(21600)

                    try:
                        stats = llm_handler.rag_system.get_search_stats()
                        cache_size = stats.get('cache_size', 0)

                        if cache_size > 100:
                            llm_handler.rag_system.clear_search_cache()
                            print(f"🧹 검색 캐시 정리 완료 (항목 수: {cache_size})")

                    except Exception as cache_error:
                        print(f"검색 캐시 정리 중 오류: {cache_error}")
                else:
                    await asyncio.sleep(86400)
            else:
                break

        except Exception as e:
            print(f"검색 캐시 정리 태스크 오류: {str(e)}")
            await asyncio.sleep(3600)

def run_api_server(port=5002):
    print(f"🚀 API 서버 시작 (포트: {port})...")
    uvicorn.run(api_app,
                host="0.0.0.0",
                port=port,
                reload=False,
                ssl_certfile="./ssl/selfsigned.crt",
                ssl_keyfile="./ssl/selfsigned.key",
                )

def main():
    print("⚡️ 설정을 로드하는 중...")
    config = ConfigManager()
    DEFAULT_MODEL = config.get_model_config().get('model')
    print(f"✅ DEFAULT_MODEL: {DEFAULT_MODEL} ")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    init_generator(DEFAULT_MODEL)

    print(f"\n=== RAG & 지식 그래프 통합 AI 답변 봇 서비스 시작 (기본 모델: {DEFAULT_MODEL.upper()}) ===")

    use_db_mode = config.get_db_config().get('use_db_mode', 'False').lower() == 'true'
    if use_db_mode:
        config.check_and_setup_db_credentials()

    if use_db_mode:
        print(f"📊 DB 모드로 초기화 (다중 구독 지원)")
    else:
        print("📁 파일 모드로 초기화")

    print("🤖 AI 시스템을 초기화하는 중...")
    start_time = time.time()

    global llm_handler
    llm_handler = LLMHandler(
        prompt_type=DEFAULT_MODEL,
        use_rag=True,
        file_extension="yaml",
        use_kg=True,
        use_db=use_db_mode  
    )
    print(f"ID : {id(llm_handler.rag_system)}")
    print(f"✅ AI 시스템 초기화 완료! (소요시간: {time.time() - start_time:.2f}초)")

    if not use_db_mode and hasattr(llm_handler.rag_system, 'initialize_ann_index'):
        if not hasattr(llm_handler.rag_system, 'ann_initialized') or not llm_handler.rag_system.ann_initialized:
            print("🔍 ANN 인덱스 초기화 중...")
            llm_handler.rag_system.initialize_ann_index()
            print("✅ ANN 인덱스 초기화 완료")

    print("\n📊 검색 시스템 상태:")
    try:
        # DEFAULT_MODEL 핸들러 우선, 없으면 llama 폴백 (원본 else 분기 보존)
        primary = llm_handler.handlers.get(DEFAULT_MODEL) or llm_handler.handlers["llama"]
        search_status = primary.get_search_status()

        print(f"   - RAG 시스템: {'✅ 활성화' if search_status['rag_system_available'] else '❌ 비활성화'}")
        print(f"   - 지식 그래프: {'✅ 활성화' if search_status['knowledge_graph_enabled'] else '❌ 비활성화'}")
        if 'cache_size' in search_status:
            print(f"   - 검색 캐시: {search_status['cache_size']}개 항목")
    except Exception as e:
        print(f"   ⚠️ 검색 상태 확인 실패: {e}")

    api_keys_dir = config.get_paths_config().get('api_keys_dir', 'api_keys')
    if not os.path.exists(api_keys_dir):
        os.makedirs(api_keys_dir)
        print(f"✅ API 키 디렉토리 생성: {api_keys_dir}")

    asyncio.run_coroutine_threadsafe(clean_cache(), loop)

    asyncio.run_coroutine_threadsafe(clean_chat_sessions(), loop)

    asyncio.run_coroutine_threadsafe(clean_search_cache(), loop)

    # === 프롬프트 싱크 시작 ===
    try:
        from aibot_sync import init_sync, periodic_sync
        from aibot_prompts_functions import _get_qdrant_client

        sync_client = init_sync(config, _get_qdrant_client(), llm_handler)
        if sync_client and sync_client.enabled:
            asyncio.run_coroutine_threadsafe(periodic_sync(), loop)
            print("✅ 프롬프트 싱크 활성화 (부팅 싱크 + 주기적 싱크)")
        else:
            print("ℹ️ 프롬프트 싱크 비활성화")
    except Exception as e:
        print(f"⚠️ 프롬프트 싱크 초기화 실패 (API 서버는 정상 운영): {e}")
    # === 프롬프트 싱크 끝 ===

    api_thread = threading.Thread(target=run_api_server,
                                    kwargs={"port": config.get_server_config().get('port', 5002)})

    api_thread.start()

    # === Slack 봇 시작 (선택적) ===
    slack_config = config.get_slack_config()
    if SLACK_AVAILABLE and slack_config.get('enabled', 'false').lower() == 'true':
        try:
            slack_app_instance = init_slack_bot(llm_handler, loop)
            register_slack_handlers(slack_app_instance)

            slack_thread = threading.Thread(
                target=run_slack_bot,
                args=(slack_app_instance, slack_config['app_token']),
                daemon=True
            )
            slack_thread.start()
            print("✅ Slack 봇이 성공적으로 시작되었습니다.")
        except Exception as e:
            print(f"⚠️ Slack 봇 시작 실패 (API 서버는 정상 운영): {e}")
    elif SLACK_AVAILABLE:
        print("ℹ️ Slack 봇 비활성화 (config.ini [slack] enabled = False)")
    # === Slack 봇 끝 ===

    print("✅ API 서버가 성공적으로 시작되었습니다.")

    def _shutdown_handler(signum, frame):
        print(f"시그널 {signum} 수신, 서비스를 종료합니다...")
        loop.call_soon_threadsafe(loop.stop)

    import signal as _sig
    _sig.signal(_sig.SIGTERM, _shutdown_handler)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("서비스를 종료합니다...")
    finally:
        from handler_gpt_oss import get_server_manager, get_translation_server_manager
        mgr = get_server_manager()
        if mgr:
            mgr.stop()
        trans_mgr = get_translation_server_manager()
        if trans_mgr:
            trans_mgr.stop()
        # gemma llama-server도 대칭 정리 — atexit은 SIGKILL 시 실행되지 않아 이 경로가 유일
        from handler_gemma import get_server_manager as _get_gemma_manager
        gemma_mgr = _get_gemma_manager()
        if gemma_mgr:
            gemma_mgr.stop()

        loop.close()
        api_thread.join()


from pydantic import BaseModel
from typing import Dict, List, Optional, Any

class IncrementalTestRequest(BaseModel):
    files_data: Dict[str, str]
    test_question: str
    intent_type: Optional[str] = "qna"
    top_k: Optional[int] = 5
    include_existing: Optional[bool] = True
    response_format: Optional[bool] = False

class IncrementalTestResponse(BaseModel):
    success: bool
    message: str
    test_results: Optional[Dict] = None
    timing_info: Optional[Dict] = None
    debug_info: Optional[Dict] = None

@api_app.post("/api/test/complete-incremental",
              description="완전한 증분 업데이트 테스트 (파일명 + 임베딩 + LLM)",
              tags=["테스트"],
              response_model=IncrementalTestResponse)
async def test_complete_incremental_update(
    request: IncrementalTestRequest,
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    start_time = time.time()
    test_id = f"test_{uuid.uuid4().hex[:8]}"

    try:
        print(f"🧪 완전한 증분 업데이트 테스트 시작 (ID: {test_id})")
        print(f"   - 파일 수: {len(request.files_data)}개")
        print(f"   - 테스트 질문: '{request.test_question}'")
        print(f"   - 의도: {request.intent_type}")

        timing = {}

        embedding_start = time.time()
        temp_embeddings = await generate_temp_embeddings_with_metadata(
            request.files_data,
            test_id
        )
        timing['embedding_generation'] = time.time() - embedding_start

        if not temp_embeddings:
            return IncrementalTestResponse(
                success=False,
                message="임베딩 생성 실패"
            )

        print(f"✅ 임베딩 생성 완료: {len(temp_embeddings)}개 ({timing['embedding_generation']:.2f}초)")

        llm_request = {
            "question": request.test_question,
            "type": request.intent_type,
            "temp_prompt": temp_embeddings,
            "response_format": request.response_format or False,
            "sub_id": 2  # 테스트 환경용 고정 구독 ID
        }

        llm_start = time.time()

        try:
            answer_result = await llm_handler.get_complete_answer(
                llm_request,
                f"test_user_{test_id}",
                "API_TEST",
                f"test_guid_{test_id}"
            )

            full_response = answer_result.get("answer", "응답 생성 실패")
            context_sources = answer_result.get("sources", [])
            detected_intent = answer_result.get("intent", request.intent_type)

            timing['llm_processing'] = time.time() - llm_start
            timing['total'] = time.time() - start_time

            print(f"✅ LLM 처리 완료: {len(full_response)}자 ({timing['llm_processing']:.2f}초)")

        except Exception as e:
            print(f"❌ LLM 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            full_response = f"LLM 처리 중 오류 발생: {str(e)}"
            context_sources = []
            detected_intent = request.intent_type
            timing['llm_processing'] = time.time() - llm_start
            timing['total'] = time.time() - start_time

        analysis_result = analyze_incremental_test_results(
            temp_embeddings=temp_embeddings,
            question=request.test_question,
            llm_response=full_response,
            context_sources=context_sources
        )

        return IncrementalTestResponse(
            success=True,
            message=f"증분 업데이트 테스트 완료 (ID: {test_id})",
            test_results={
                "question": request.test_question,
                "intent": detected_intent,
                "llm_response": full_response,
                "response_length": len(full_response),
                "context_sources": context_sources,
                "temp_files": list(request.files_data.keys()),
                "analysis": analysis_result
            },
            timing_info=timing,
            debug_info={
                "test_id": test_id,
                "temp_embeddings_count": len(temp_embeddings),
                "embedding_keys": list(temp_embeddings.keys())
            }
        )

    except Exception as e:
        print(f"❌ 증분 업데이트 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

        return IncrementalTestResponse(
            success=False,
            message=f"테스트 실패: {str(e)}",
            timing_info={"total": time.time() - start_time}
        )

async def generate_temp_embeddings_with_metadata(files_data: Dict[str, str], test_id: str) -> Dict[str, Dict]:
    """임시 임베딩 생성 (BGE 모드 / FAISS 모드 자동 분기)"""
    print(f"🔄 임베딩 생성 중...")

    if _is_bge_mode():
        return await _generate_temp_embeddings_bge(files_data, test_id)
    else:
        return await _generate_temp_embeddings_faiss(files_data, test_id)


async def _generate_temp_embeddings_bge(files_data: Dict[str, str], test_id: str) -> Dict[str, Dict]:
    """BGE-M3 모드: 임베딩은 RAGSystemBGE.find_similar_docs_test()에서 생성하므로 메타데이터만 구성"""
    print(f"   🔧 BGE-M3 모드: 메타데이터 준비 (임베딩은 RAGSystemBGE에서 생성)")
    temp_embeddings = {}

    for file_path, content in files_data.items():
        try:
            print(f"   - 준비 중: {file_path}")
            _, yaml_data = parse_yaml_for_embedding(content)

            temp_embeddings[file_path] = {
                "content": content,
                "filename": file_path.split("/")[-1],
                "file_path": file_path,
                "file_extension": file_path.split(".")[-1] if "." in file_path else "",

                "intent_folder": file_path.split("/")[0] if "/" in file_path else "general",
                "prompt_type": file_path.split("/")[0] if "/" in file_path else "general",

                "yaml_structure": yaml_data if yaml_data else {},
                "question": yaml_data.get("question", "") if yaml_data else "",
                "answer": yaml_data.get("answer", "") if yaml_data else "",
                "aliases": yaml_data.get("aliases", []) if yaml_data else [],
                "keywords": extract_keywords_from_yaml(yaml_data) if yaml_data else [],

                "test_id": test_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "temp_data": True,
            }

            print(f"     ✅ 준비 완료: {file_path}")

        except Exception as file_error:
            print(f"     ❌ 파일 처리 실패: {file_error}")
            continue

    print(f"✅ BGE 메타데이터 준비 완료: {len(temp_embeddings)}개")
    return temp_embeddings


async def _generate_temp_embeddings_faiss(files_data: Dict[str, str], test_id: str) -> Dict[str, Dict]:
    """FAISS 모드: 기존 단일 dense 벡터 생성"""
    temp_generator = None
    created_new_generator = False

    try:
        global llm_handler
        if hasattr(llm_handler, 'rag_system') and hasattr(llm_handler.rag_system, 'embedding_generator'):
            temp_generator = llm_handler.rag_system.embedding_generator
            print(f"   ✅ 기존 임베딩 생성기 재사용")
        else:
            print(f"   ⚠️ 기존 임베딩 생성기 없음, 새로 생성")
            from aibot_embedding import EmbeddingGenerator
            temp_generator = EmbeddingGenerator(
                use_local_model=True,
                local_model_path="../models/bge-m3",
                save_db=False
            )
            created_new_generator = True

        temp_embeddings = {}

        for file_path, content in files_data.items():
            try:
                print(f"   - 처리 중: {file_path}")

                embedding_text, yaml_data = parse_yaml_for_embedding(content)

                embedding_result = temp_generator.embedding_adapter.generate_embedding(embedding_text)

                if embedding_result and 'data' in embedding_result:
                    vector = embedding_result['data'][0]['embedding']
                    token_count = embedding_result['usage']['total_tokens']

                    temp_embeddings[file_path] = {
                        "vector": vector,
                        "content": content,
                        "token_count": token_count,
                        "embedding_text": embedding_text,

                        "filename": file_path.split("/")[-1],
                        "file_path": file_path,
                        "file_extension": file_path.split(".")[-1] if "." in file_path else "",

                        "intent_folder": file_path.split("/")[0] if "/" in file_path else "general",
                        "prompt_type": file_path.split("/")[0] if "/" in file_path else "general",

                        "yaml_structure": yaml_data if yaml_data else {},
                        "question": yaml_data.get("question", "") if yaml_data else "",
                        "answer": yaml_data.get("answer", "") if yaml_data else "",
                        "aliases": yaml_data.get("aliases", []) if yaml_data else [],
                        "keywords": extract_keywords_from_yaml(yaml_data) if yaml_data else [],

                        "test_id": test_id,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "temp_data": True
                    }

                    print(f"     ✅ 완료: 벡터 차원 {len(vector)}, 토큰 {token_count}")

                else:
                    print(f"     ❌ 임베딩 생성 실패")

            except Exception as file_error:
                print(f"     ❌ 파일 처리 실패: {file_error}")
                continue

        print(f"✅ 임베딩 생성 완료: {len(temp_embeddings)}개")
        return temp_embeddings

    except Exception as e:
        print(f"❌ 임베딩 생성 전체 실패: {e}")
        return {}

    finally:
        if created_new_generator and temp_generator:
            try:
                if hasattr(temp_generator, 'model'):
                    del temp_generator.model

                if hasattr(temp_generator, 'embedding_adapter'):
                    if hasattr(temp_generator.embedding_adapter, 'model'):
                        del temp_generator.embedding_adapter.model
                    del temp_generator.embedding_adapter

                del temp_generator

                gc.collect()

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

                print(f"   🧹 새로 생성한 임베딩 생성기 메모리 정리 완료")

            except Exception as cleanup_error:
                print(f"   ⚠️ 임베딩 생성기 정리 중 오류: {cleanup_error}")

def parse_yaml_for_embedding(content: str) -> tuple[str, dict]:
    try:
        yaml_data = yaml.safe_load(content)
        if not isinstance(yaml_data, dict):
            return content, {}

        embedding_parts = []

        for field, prefix in [
            ('question', '질문'),
            ('cot', '추론과정'),
            ('answer', '답변'),
            ('aliases', '관련키워드')
        ]:
            value = yaml_data.get(field, '')
            if value:
                if isinstance(value, list):
                    value = ' '.join(str(item) for item in value)
                embedding_parts.append(f"{prefix}: {str(value).strip()}")

        for field in ['keywords', 'category', 'tags', 'description']:
            value = yaml_data.get(field, '')
            if value:
                if isinstance(value, list):
                    value = ' '.join(str(item) for item in value)
                embedding_parts.append(f"{field}: {str(value)}")

        embedding_text = ' '.join(embedding_parts) if embedding_parts else content

        if len(embedding_text) > 6000:
            embedding_text = embedding_text[:6000]

        return embedding_text, yaml_data

    except yaml.YAMLError:
        return content, {}

def extract_keywords_from_yaml(yaml_data: dict) -> List[str]:
    keywords = []

    for field in ['keywords', 'aliases', 'tags']:
        value = yaml_data.get(field, [])
        if isinstance(value, list):
            keywords.extend([str(v) for v in value])
        elif isinstance(value, str):
            keywords.extend(value.split(','))

    for field in ['question', 'answer']:
        text = yaml_data.get(field, '')
        if text:
            import re
            words = re.findall(r'\b[가-힣a-zA-Z]{2,}\b', str(text))
            keywords.extend(words[:5])

    unique_keywords = list(set([k.strip() for k in keywords if k.strip()]))
    return unique_keywords[:10]

def analyze_incremental_test_results(temp_embeddings: Dict, question: str, llm_response: str, context_sources: List) -> Dict:

    analysis = {
        "temp_data_usage": {
            "total_temp_files": len(temp_embeddings),
            "temp_files_in_context": 0,
            "temp_file_matches": []
        },
        "response_quality": {
            "response_length": len(llm_response),
            "contains_temp_info": False,
            "relevance_indicators": []
        },
        "search_effectiveness": {
            "context_sources_count": len(context_sources),
            "temp_vs_existing_ratio": "unknown"
        }
    }

    temp_file_names = list(temp_embeddings.keys())
    for source in context_sources:
        source_str = str(source)
        for temp_file in temp_file_names:
            if temp_file in source_str or temp_file.split("/")[-1] in source_str:
                analysis["temp_data_usage"]["temp_files_in_context"] += 1
                analysis["temp_data_usage"]["temp_file_matches"].append({
                    "temp_file": temp_file,
                    "found_in_source": source_str
                })

    for file_path, embedding_data in temp_embeddings.items():
        yaml_data = embedding_data.get("yaml_structure", {})
        if yaml_data:
            for field in ['question', 'answer']:
                field_content = yaml_data.get(field, '')
                if field_content and len(str(field_content)) > 5:
                    import re
                    keywords = re.findall(r'\b[가-힣a-zA-Z]{3,}\b', str(field_content))
                    for keyword in keywords[:3]:
                        if keyword.lower() in llm_response.lower():
                            analysis["response_quality"]["contains_temp_info"] = True
                            analysis["response_quality"]["relevance_indicators"].append({
                                "keyword": keyword,
                                "from_file": file_path,
                                "field": field
                            })

    return analysis

# Embedding Management APIs
@api_app.get("/api/embeddings/list", description="임베딩 목록 조회", tags=["임베딩 관리"])
async def get_embeddings_list(user_info: Dict = Depends(get_bearer_api_key_user)):
    """DB 또는 Qdrant에서 임베딩 데이터 목록을 조회합니다."""
    try:
        # BGE-M3 모드: Qdrant에서 조회
        if _is_bge_mode():
            from aibot_prompts_functions import _get_bge_db_manager
            from qdrant_client import models

            bge_db = _get_bge_db_manager()
            client = bge_db.qdrant_client
            sub_id = user_info.get('id', 1)

            # scroll로 전체 문서 조회 (sub_id 필터)
            embeddings = []
            offset = None
            while True:
                points, next_offset = client.scroll(
                    collection_name=qdrant_collection(),
                    with_payload=True,
                    with_vectors=False,
                    limit=100,
                    offset=offset
                )

                # guid가 없는 포인트에 자동 생성
                points_to_patch = []
                for point in points:
                    p = point.payload or {}
                    guid = p.get('guid')
                    if not guid:
                        import uuid
                        guid = str(uuid.uuid4())
                        points_to_patch.append((point.id, guid))

                    embeddings.append({
                        'id': point.id,
                        'sub_id': p.get('sub_id', 0),
                        'prompt': p.get('file_key') or p.get('name', ''),
                        'answer': p.get('content') or p.get('text', ''),
                        'dimension': 1024,
                        'created_at': None,
                        'updated_at': None,
                        'rev': p.get('rev', 0),
                        'type': p.get('doc_type') or p.get('type', 'general'),
                        'guid': guid,
                        'enabled': p.get('enabled', True),
                        'tags': p.get('tags'),
                        'capec_ids': p.get('capec_ids', []),
                        'cwe_ids': p.get('cwe_ids', []),
                        'cve_ids': p.get('cve_ids', []),
                        'attack_technique_ids': p.get('attack_technique_ids', []),
                        'attack_group_ids': p.get('attack_group_ids', []),
                        'attack_software_ids': p.get('attack_software_ids', []),
                        'norm_ids': p.get('norm_ids', []),
                    })

                # guid가 없던 포인트들에 guid 패치
                if points_to_patch:
                    for pid, new_guid in points_to_patch:
                        client.set_payload(
                            collection_name=qdrant_collection(),
                            payload={"guid": new_guid},
                            points=[pid],
                        )

                if next_offset is None:
                    break
                offset = next_offset

            embeddings.sort(key=lambda x: x['id'])
            return JSONResponse(content={
                "success": True,
                "embeddings": embeddings,
                "total": len(embeddings)
            })

        # 레거시 모드: MariaDB에서 조회
        db_manager = AibotDBManager(
            config=config,
            query_properties=SQL_QUERIES
        )

        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT
                        id,
                        name as prompt,
                        prompt as answer,
                        LENGTH(vector) as vector_dimension,
                        created_at,
                        updated_at,
                        rev,
                        type
                    FROM openai_prompts
                    WHERE subscription_id = 1
                    ORDER BY id ASC
                ''')

                results = cursor.fetchall()

                embeddings = []
                for row in results:
                    embedding_data = {
                        'id': row['id'],
                        'prompt': row['prompt'] or '',
                        'answer': row['answer'] or '',
                        'dimension': row['vector_dimension'] if row['vector_dimension'] else 0,
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
                        'rev': row['rev'] or 0,
                        'type': row['type'] or 'general'
                    }
                    embeddings.append(embedding_data)

                return JSONResponse(content={
                    "success": True,
                    "embeddings": embeddings,
                    "total": len(embeddings)
                })

    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_app.put("/api/embeddings/{embedding_id}", description="임베딩 데이터 수정 및 재임베딩", tags=["임베딩 관리"])
async def update_embedding(
    embedding_id: int,
    prompt: str = Body(...),
    answer: str = Body(...),
    reembed: bool = Body(True),
    user_info: Dict = Depends(get_bearer_api_key_user)
):
    """프롬프트 데이터를 수정하고 재임베딩을 수행합니다."""
    try:
        # BGE-M3 모드: Qdrant에서 직접 업데이트
        if _is_bge_mode():
            from aibot_prompts_functions import _get_bge_adapter, _get_bge_db_manager, _extract_embedding_text
            from qdrant_client import models

            sub_id = user_info.get('id', 1)
            point_id = embedding_id  # Qdrant point ID
            bge_adapter = _get_bge_adapter()
            bge_db = _get_bge_db_manager()
            client = bge_db.qdrant_client

            # 기존 point 확인
            try:
                points = client.retrieve(collection_name=qdrant_collection(), ids=[point_id], with_payload=True, with_vectors=False)
                if not points:
                    raise HTTPException(status_code=404, detail="임베딩을 찾을 수 없습니다.")
                existing_payload = points[0].payload or {}
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"임베딩을 찾을 수 없습니다: {e}")

            file_key = prompt  # prompt 필드 = 파일명
            yaml_content = answer  # answer 필드 = YAML 내용
            guid = existing_payload.get('guid')

            if reembed:
                # 임베딩 텍스트 추출 및 재임베딩
                embedding_text = _extract_embedding_text(yaml_content)
                embedding_result = bge_adapter.generate_embedding(embedding_text)
                packed_vector = embedding_result['data'][0]['embedding']

                # update_document 호출 (API update_prompt_bge와 동일 경로)
                doc = {
                    'text': yaml_content,
                    'file_key': file_key,
                    'doc_type': existing_payload.get('doc_type', 'qna'),
                    'guid': guid,
                    'enabled': existing_payload.get('enabled', True),
                }
                success = bge_db.update_document(sub_id, point_id, packed_vector, doc)

                if not success:
                    raise HTTPException(status_code=500, detail="Qdrant 업데이트 실패")

                print(f"✅ BGE 재임베딩 완료: point_id={point_id}")
            else:
                # 텍스트만 업데이트 (payload만 변경)
                client.set_payload(
                    collection_name=qdrant_collection(),
                    payload={"content": yaml_content, "text": yaml_content, "file_key": file_key, "name": file_key},
                    points=[point_id]
                )

            # RAG 리로드
            try:
                llm_handler.reload_embedding(sub_id=sub_id)
            except Exception as e:
                print(f"⚠️ RAG 리로드 실패: {e}")

            return JSONResponse(content={
                "success": True,
                "message": f"임베딩 ID {point_id} {'재임베딩 및 ' if reembed else ''}업데이트 완료",
                "data": {
                    'id': point_id,
                    'prompt': file_key,
                    'answer': yaml_content,
                    'dimension': 1024,
                    'reembedded': reembed
                }
            })

        # 레거시 모드: 기존 로직
        db_manager = AibotDBManager(
            config=config,
            query_properties=SQL_QUERIES
        )

        sub_id = user_info.get('id', 1)

        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT subscription_id, source, guid, vector
                    FROM openai_prompts
                    WHERE id = %s
                ''', (embedding_id,))
                existing_data = cursor.fetchone()

                if not existing_data:
                    raise HTTPException(status_code=404, detail="임베딩을 찾을 수 없습니다.")

                sub_id = existing_data['subscription_id'] or sub_id
                source = existing_data['source'] or f"manual_edit_{embedding_id}"
                guid = existing_data['guid']

                existing_vector_dim = None
                if existing_data['vector']:
                    try:
                        import json
                        existing_vector = json.loads(existing_data['vector'])
                        existing_vector_dim = len(existing_vector)
                        print(f"🔍 기존 벡터 차원: {existing_vector_dim}")
                    except:
                        print(f"⚠️ 기존 벡터 파싱 실패")

        # EmbeddingGenerator를 통한 임베딩 처리
        if reembed:
            try:
                from aibot_prompts_functions import _generator_cache, init_generator, parse_prompt

                # 모델 타입 확인 (DEFAULT_MODEL 사용)
                model = DEFAULT_MODEL

                # Generator가 없으면 초기화
                if model not in _generator_cache:
                    init_generator(model)

                generator = _generator_cache.get(model)
                if generator:
                    # 사용자가 입력한 데이터 매핑:
                    # prompt -> 파일명, answer -> YAML 데이터
                    filename = prompt  # 파일명
                    yaml_content = answer  # YAML 데이터

                    # 파일명에서 확장자 제거하여 name 생성
                    name = filename.replace('.yaml', '') if filename.endswith('.yaml') else filename

                    # parse_prompt를 사용하여 올바른 구조 생성
                    try:
                        # YAML에서 type 값 추출
                        import yaml
                        parsed_yaml = yaml.safe_load(yaml_content)
                        prompt_type = parsed_yaml.get('type', 'prompt')  # 기본값은 'prompt'

                        prompt_data = parse_prompt(
                            prompt_type=prompt_type,  # YAML의 실제 type 값 사용
                            name=name,
                            description=f"Updated via web interface - {filename}",
                            prompt=yaml_content,  # YAML 데이터가 prompt 파라미터로 전달됨
                            guid=guid
                        )
                    except Exception as parse_error:
                        print(f"❌ parse_prompt 실패: {parse_error}")
                        # 파싱 실패시 간단한 형태로 fallback
                        prompt_data = {
                            f"prompt/{name}.yaml": (yaml_content, yaml_content)
                        }

                    # 임베딩 업데이트 처리
                    print(f"🔍 임베딩 생성 시작:")
                    print(f"   - 모델: {model}")
                    print(f"   - 구독키: {sub_id}")
                    print(f"   - 프롬프트 데이터 키: {list(prompt_data.keys())}")

                    ctx = generator.embed_changes_only(sub_id, prompt_data, tags=None)
                    print(f"🔍 임베딩 생성 결과:")
                    print(f"   - 새 임베딩 키들: {list(ctx.get('new_embedding', {}).keys())}")
                    if ctx.get('new_embedding'):
                        for key, emb_obj in ctx['new_embedding'].items():
                            print(f"   - {key}: 타입={type(emb_obj)}")
                            if hasattr(emb_obj, 'vector'):
                                print(f"   - {key}: 벡터 차원={len(emb_obj.vector) if emb_obj.vector else 'None'}")
                            if hasattr(emb_obj, '__dict__'):
                                print(f"   - {key}: 속성={list(emb_obj.__dict__.keys())}")

                    # 새 벡터 추출
                    new_vector = None
                    if ctx.get('new_embedding') and prompt_data:
                        file_key = list(prompt_data.keys())[0]
                        if file_key in ctx['new_embedding']:
                            embedding_obj = ctx['new_embedding'][file_key]

                            # Embedding 객체에서 벡터 추출 시도
                            raw_vector = None
                            if hasattr(embedding_obj, 'vector'):
                                raw_vector = embedding_obj.vector
                            elif hasattr(embedding_obj, 'embedding'):
                                raw_vector = embedding_obj.embedding
                            elif hasattr(embedding_obj, 'data'):
                                raw_vector = embedding_obj.data

                            if raw_vector is not None:
                                # 벡터 데이터 정규화
                                if isinstance(raw_vector, bytes):
                                    # 바이트 문자열인 경우 JSON으로 파싱
                                    import json
                                    try:
                                        new_vector = json.loads(raw_vector.decode('utf-8'))
                                    except:
                                        new_vector = None
                                elif isinstance(raw_vector, str):
                                    # 문자열인 경우 JSON으로 파싱
                                    import json
                                    try:
                                        new_vector = json.loads(raw_vector)
                                    except:
                                        new_vector = None
                                elif hasattr(raw_vector, 'tolist'):
                                    # numpy 배열인 경우 리스트로 변환
                                    new_vector = raw_vector.tolist()
                                else:
                                    # 이미 리스트인 경우
                                    new_vector = raw_vector

                                if new_vector is not None and len(new_vector) > 0:
                                    new_vector_dim = len(new_vector)
                                    print(f"✅ 재임베딩 성공: ID {embedding_id}, 새 벡터 차원 {new_vector_dim}")

                                    # 벡터 차원 비교
                                    if existing_vector_dim and existing_vector_dim != new_vector_dim:
                                        print(f"⚠️ 벡터 차원 불일치 감지!")
                                        print(f"   - 기존 차원: {existing_vector_dim}")
                                        print(f"   - 새 차원: {new_vector_dim}")
                                        print(f"   - 모델: {model}")
                                        print(f"⚠️ 차원이 다른 벡터는 FAISS 인덱스 호환성 문제를 일으킬 수 있습니다.")
                                        # 차원이 다르면 벡터 저장 안함
                                        new_vector = None
                                        print(f"⚠️ 벡터 차원 불일치로 텍스트만 업데이트합니다.")
                                else:
                                    new_vector = None

                    if new_vector is None:
                        print("⚠️ 임베딩 어댑터를 사용할 수 없어 텍스트만 업데이트합니다.")

                    # 모든 활성 구독키에 대해 FAISS/지식그래프 업데이트 및 리로드
                    try:
                        # 활성 구독키 목록 조회
                        active_subs = db_manager.get_active_subscription_ids()
                        if not active_subs:
                            active_subs = [1]  # 기본값

                        # 각 구독키별로 FAISS/지식그래프 업데이트 및 리로드
                        for active_sub_id in active_subs:
                            try:
                                # 1번 구독키 데이터가 업데이트되었으므로 모든 구독키에 대해 postprocess 실행
                                generator.postprocess_with_faiss_and_kg(active_sub_id, ctx)

                                # 구독키별 리로드
                                llm_handler.set_model(model)
                                llm_handler.reload_embedding(sub_id=int(active_sub_id))
                            except Exception as e:
                                print(f"⚠️ 구독 {active_sub_id} 업데이트/리로드 실패: {e}")

                        print(f"✅ {len(active_subs)}개 구독키 FAISS/지식그래프/임베딩 업데이트 완료")
                    except Exception as e:
                        print(f"⚠️ 활성 구독키 업데이트 중 오류: {e}")
                        # 기본적으로 1번 구독키는 업데이트
                        try:
                            generator.postprocess_with_faiss_and_kg(1, ctx)
                            llm_handler.set_model(model)
                            llm_handler.reload_embedding(sub_id=1)
                            print("✅ 기본 구독키(1번) 업데이트 완료")
                        except Exception as fallback_e:
                            print(f"⚠️ 기본 구독키 업데이트도 실패: {fallback_e}")

                else:
                    print("⚠️ 임베딩 제너레이터를 찾을 수 없어 텍스트만 업데이트합니다.")
                    new_vector = None

            except Exception as e:
                print(f"⚠️ 임베딩 처리 중 오류 (텍스트만 업데이트): {e}")
                import traceback
                traceback.print_exc()
                new_vector = None
        else:
            new_vector = None

        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                if new_vector:
                    # 기존 시스템과 동일한 방식으로 벡터 저장 (바이트 형태)
                    try:
                        import numpy as np
                        # 벡터를 numpy 배열로 변환 후 바이트로 저장 (기존 방식과 동일)
                        vector_array = np.array(new_vector, dtype=np.float32)
                        vector_bytes = vector_array.tobytes()

                        print(f"🔍 벡터 저장 준비:")
                        print(f"   - 벡터 타입: {type(new_vector)}")
                        print(f"   - 벡터 차원: {len(new_vector)}")
                        print(f"   - numpy 배열 형태: {vector_array.shape}")
                        print(f"   - 바이트 길이: {len(vector_bytes)}")

                        cursor.execute('''
                            UPDATE openai_prompts
                            SET name = %s,
                                prompt = %s,
                                vector = %s,
                                updated_at = NOW()
                            WHERE id = %s
                        ''', (prompt, answer, vector_bytes, embedding_id))

                        print(f"✅ 벡터 저장 완료: ID {embedding_id} (기존 바이트 형식)")
                    except Exception as vec_error:
                        print(f"⚠️ 벡터 바이트 변환 실패: {vec_error} - 텍스트만 업데이트")
                        new_vector = None
                else:
                    # 텍스트만 업데이트
                    cursor.execute('''
                        UPDATE openai_prompts
                        SET name = %s,
                            prompt = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    ''', (prompt, answer, embedding_id))

                conn.commit()


                # 업데이트된 데이터 조회
                cursor.execute('''
                    SELECT
                        id,
                        name as prompt,
                        prompt as answer,
                        LENGTH(vector) as vector_dimension,
                        created_at,
                        updated_at
                    FROM openai_prompts
                    WHERE id = %s
                ''', (embedding_id,))

                updated_row = cursor.fetchone()

                if updated_row:
                    updated_data = {
                        'id': updated_row['id'],
                        'prompt': updated_row['prompt'] or '',
                        'answer': updated_row['answer'] or '',
                        'dimension': updated_row['vector_dimension'] if updated_row['vector_dimension'] else 0,
                        'created_at': updated_row['created_at'].isoformat() if updated_row['created_at'] else None,
                        'updated_at': updated_row['updated_at'].isoformat() if updated_row['updated_at'] else None,
                        'reembedded': bool(new_vector)
                    }

                    return JSONResponse(content={
                        "success": True,
                        "message": f"임베딩 ID {embedding_id} {'재임베딩 및 ' if new_vector else ''}업데이트 완료",
                        "data": updated_data
                    })
                else:
                    raise HTTPException(status_code=404, detail="임베딩을 찾을 수 없습니다.")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    query_validation = Query_validation()



    main()

from aibot_prompts_class import Record, LlmPrompt, OpenAiPrompt
from typing import List, Optional, Set, Dict, Any
from uuid import UUID
from aibot_embedding import EmbeddingGenerator, Embedding
from config_utils import ConfigManager, qdrant_collection
import pymysql
import yaml, os, time, json, uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor


def generate_file_guid(file_key: str) -> str:
    """파일 경로 기반 결정적 GUID 생성 (같은 파일 → 항상 같은 GUID)

    Args:
        file_key: 파일 경로 (예: "qna/my-topic.yaml", "action/restart-server.yaml")

    Returns:
        str: UUID5 문자열
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, file_key))

# ============================================================
# BGE 모드 설정 및 헬퍼 함수들
# ============================================================
_config = ConfigManager()
_bge_adapter = None
_bge_db_manager = None
_qdrant_client = None

def _is_bge_mode() -> bool:
    """BGE 모드 여부 확인"""
    return _config.config.get('embedding', 'use_bge_mode', fallback='False').lower() == 'true'

def _get_bge_adapter(llm_handler=None):
    """BGE 임베딩 어댑터 싱글톤 (기존 모델 재사용)"""
    global _bge_adapter
    if _bge_adapter is None:
        from aibot_embedding_BGE import BGEEmbeddingAdapter
        bge_model_path = _config.config.get('embedding', 'bge_model_path', fallback='/data/models/bge-m3')

        # llm_handler의 RAG 시스템에서 이미 로드된 BGE 모델 재사용
        existing_model = None
        if llm_handler and hasattr(llm_handler, 'rag_system') and llm_handler.rag_system:
            existing_model = getattr(llm_handler.rag_system, 'bge_model', None)
            if existing_model:
                print(f"♻️ BGE 어댑터: RAG 시스템의 기존 모델 재사용")

        _bge_adapter = BGEEmbeddingAdapter(model_path=bge_model_path, existing_model=existing_model)
        print(f"✅ BGE 어댑터 초기화 완료")
    return _bge_adapter

def _get_bge_db_manager():
    """BGE DB 매니저 싱글톤"""
    global _bge_db_manager
    if _bge_db_manager is None:
        from aibot_embedding_BGE import BGEVectorDBManager
        from aibot_db_manager import AibotDBManager
        from aibot_db_command import SQL_QUERIES
        db_manager = AibotDBManager(config=_config, query_properties=SQL_QUERIES)
        _bge_db_manager = BGEVectorDBManager(_config, db_manager)
        print(f"✅ BGE DB 매니저 초기화 완료")
    return _bge_db_manager

def _get_qdrant_client():
    """Qdrant 클라이언트 싱글톤"""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        host = _config.config.get('qdrant', 'host', fallback='localhost')
        port = int(_config.config.get('qdrant', 'port', fallback='6333'))
        timeout = int(_config.config.get('qdrant', 'timeout', fallback='120'))
        _qdrant_client = QdrantClient(host=host, port=port, timeout=timeout)
        print(f"✅ Qdrant 클라이언트 초기화 완료 ({host}:{port})")
    return _qdrant_client

# ============================================================
# BGE 모드 프롬프트 함수들
# ============================================================
def create_prompt_bge(user_info, guid, prompt, llm_handler, skip_reload=False) -> Dict[str, Embedding]:
    """BGE 모드 프롬프트 생성

    Args:
        skip_reload: True면 RAG 리로드를 건너뜀 (bulk 처리 시 마지막에 1회만 리로드)
    """
    sub_id = user_info['id']
    print(f"🚀 [BGE] 프롬프트 생성 시작 (sub_id={sub_id}, guid={guid})")

    # 1. BGE 임베딩 어댑터 가져오기 (기존 모델 재사용)
    bge_adapter = _get_bge_adapter(llm_handler)

    # 2. 프롬프트 텍스트 임베딩 생성
    embeddings_data = []
    result_embeddings = {}

    for file_key, (prompt_text, parse_data) in prompt.items():
        try:
            # 임베딩용 텍스트 추출
            embedding_text = _extract_embedding_text(parse_data)

            # BGE 임베딩 생성
            embedding_result = bge_adapter.generate_embedding(embedding_text)
            embedding = embedding_result['data'][0]['embedding']
            token_count = embedding_result['usage']['total_tokens']

            embeddings_data.append({
                'file_key': file_key,
                'text': parse_data,  # YAML 형식 (guid 포함)
                'embedding': embedding
            })

            result_embeddings[file_key] = Embedding(
                token_count=token_count,
                vector=embedding
            )

            print(f"   ✅ 임베딩 생성 완료: {file_key}")

        except Exception as e:
            print(f"   ❌ 임베딩 생성 실패 ({file_key}): {e}")
            raise RuntimeError(f"임베딩 생성 실패 ({file_key}): {e}")

    # 3. Qdrant에 저장
    if embeddings_data:
        try:
            bge_db_manager = _get_bge_db_manager()
            uploaded = bge_db_manager.store_bge_embeddings(sub_id, embeddings_data)
            print(f"   ✅ Qdrant 저장 완료: {uploaded}개")
        except Exception as e:
            print(f"   ❌ Qdrant 저장 실패: {e}")
            raise RuntimeError(f"Qdrant 저장 실패: {e}")

    # 4. RAG 시스템 리로드 (skip_reload=True면 건너뜀)
    if skip_reload:
        return result_embeddings
    try:
        llm_handler.reload_embedding(sub_id=sub_id)
        print(f"   ✅ RAG 리로드 완료")
    except Exception as e:
        print(f"   ⚠️ RAG 리로드 실패: {e}")

    return result_embeddings

def remove_prompt_bge(sub_id: int, guid_list: List[str], llm_handler):
    """BGE 모드 프롬프트 삭제"""
    import re
    from qdrant_client import models

    # UUID 형식만 필터 (숫자 등 잘못된 값 제거)
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
    guid_list = [str(g) for g in guid_list if UUID_PATTERN.match(str(g))]

    print(f"🗑️ [BGE] 프롬프트 삭제 시작 (sub_id={sub_id}, guids={guid_list})")

    if not guid_list:
        print(f"   ⚠️ 유효한 UUID가 없습니다")
        return

    client = _get_qdrant_client()
    deleted_count = 0

    for guid in guid_list:
        try:
            # guid로 문서 검색
            result = client.scroll(
                collection_name=qdrant_collection(),
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="sub_id", match=models.MatchValue(value=sub_id)),
                        models.FieldCondition(key="guid", match=models.MatchValue(value=guid)),
                    ]
                ),
                with_payload=False,
                with_vectors=False,
                limit=10
            )

            # point ID로 삭제
            point_ids = [p.id for p in result[0]]
            if point_ids:
                client.delete(
                    collection_name=qdrant_collection(),
                    points_selector=models.PointIdsList(points=point_ids)
                )
                deleted_count += len(point_ids)
                print(f"   ✅ 삭제 완료: guid={guid}, points={point_ids}")
            else:
                print(f"   ⚠️ 문서 없음: guid={guid}")

        except Exception as e:
            print(f"   ❌ 삭제 실패 (guid={guid}): {e}")

    # RAG 시스템 리로드
    try:
        llm_handler.reload_embedding(sub_id=sub_id)
        print(f"   ✅ RAG 리로드 완료")
    except Exception as e:
        print(f"   ⚠️ RAG 리로드 실패: {e}")

    print(f"🗑️ [BGE] 삭제 완료: 총 {deleted_count}개")

def toggle_prompt_bge(sub_id: int, guid: str, enabled: bool):
    """BGE 모드 프롬프트 토글"""
    from qdrant_client import models

    print(f"🔄 [BGE] 프롬프트 토글 (sub_id={sub_id}, guid={guid}, enabled={enabled})")

    client = _get_qdrant_client()

    try:
        # guid로 문서 검색
        result = client.scroll(
            collection_name=qdrant_collection(),
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="sub_id", match=models.MatchValue(value=sub_id)),
                    models.FieldCondition(key="guid", match=models.MatchValue(value=guid)),
                ]
            ),
            with_payload=False,
            with_vectors=False,
            limit=1
        )

        # payload 업데이트
        if result[0]:
            point_id = result[0][0].id
            client.set_payload(
                collection_name=qdrant_collection(),
                payload={"enabled": enabled},
                points=[point_id]
            )
            print(f"   ✅ 토글 완료: point_id={point_id}, enabled={enabled}")
        else:
            print(f"   ⚠️ 문서 없음: guid={guid}")

    except Exception as e:
        print(f"   ❌ 토글 실패: {e}")

def update_prompt_bge(user_info, guid, prompt, llm_handler) -> Dict[str, Embedding]:
    """BGE 모드 프롬프트 업데이트 (벡터 + payload 모두 업데이트)"""
    from qdrant_client import models

    sub_id = user_info['id']
    print(f"🔄 [BGE] 프롬프트 업데이트 시작 (sub_id={sub_id}, guid={guid})")

    client = _get_qdrant_client()
    bge_adapter = _get_bge_adapter(llm_handler)
    bge_db_manager = _get_bge_db_manager()
    result_embeddings = {}

    for file_key, (prompt_text, parse_data) in prompt.items():
        try:
            # 1. 기존 point ID 찾기
            search_result = client.scroll(
                collection_name=qdrant_collection(),
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="sub_id", match=models.MatchValue(value=sub_id)),
                        models.FieldCondition(key="guid", match=models.MatchValue(value=guid)),
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=1
            )

            if not search_result[0]:
                print(f"   ⚠️ 기존 문서 없음, 새로 생성: guid={guid}")
                return create_prompt_bge(user_info, guid, prompt, llm_handler)

            point_id = search_result[0][0].id
            existing_payload = search_result[0][0].payload

            # 2. 새 임베딩 생성 (dense, sparse, colbert)
            embedding_text = _extract_embedding_text(parse_data)
            embedding_result = bge_adapter.generate_embedding(embedding_text)
            packed_vector = embedding_result['data'][0]['embedding']
            token_count = embedding_result['usage']['total_tokens']

            # 3. BGEVectorDBManager.update_document() 호출
            existing_rev = existing_payload.get("rev", 0)
            doc = {
                'text': parse_data,
                'file_key': file_key,
                'guid': guid,
                'enabled': existing_payload.get("enabled", True),
                'rev': existing_rev + 1,
            }

            success = bge_db_manager.update_document(sub_id, point_id, packed_vector, doc)

            if success:
                result_embeddings[file_key] = Embedding(
                    token_count=token_count,
                    vector=packed_vector.get('dense', [])
                )
                print(f"   ✅ 업데이트 완료: point_id={point_id}")
            else:
                print(f"   ❌ 업데이트 실패: point_id={point_id}")
                raise RuntimeError(f"Qdrant 업데이트 실패: point_id={point_id}")

        except Exception as e:
            print(f"   ❌ 업데이트 실패 ({file_key}): {e}")
            raise RuntimeError(f"프롬프트 업데이트 실패 ({file_key}): {e}")

    # RAG 시스템 리로드
    try:
        llm_handler.reload_embedding(sub_id=sub_id)
        print(f"   ✅ RAG 리로드 완료")
    except Exception as e:
        print(f"   ⚠️ RAG 리로드 실패: {e}")

    return result_embeddings

def _extract_embedding_text(yaml_content: str) -> str:
    """YAML에서 임베딩용 텍스트 추출"""
    try:
        yaml_data = yaml.safe_load(yaml_content)
        if not isinstance(yaml_data, dict):
            return yaml_content

        parts = []
        for field, prefix in [
            ('question', '질문'),
            ('cot', '추론과정'),
            ('spec', '명세'),
            ('answer', '답변'),
            ('description', '설명'),
        ]:
            value = yaml_data.get(field, '')
            if value:
                if isinstance(value, list):
                    value = ' '.join(str(item) for item in value)
                parts.append(f"{prefix}: {str(value).strip()}")

        return ' '.join(parts) if parts else yaml_content

    except:
        return yaml_content

# ============================================================
# 기존 함수들
# ============================================================
class LiteralStr(str): pass
def literal_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
yaml.add_representer(LiteralStr, literal_str_representer)


def create_prompt(user_info, guid, prompt, model, llm_handler, db_manager=None, skip_reload=False):
    # BGE 모드 분기
    if _is_bge_mode():
        return create_prompt_bge(user_info, guid, prompt, llm_handler, skip_reload=skip_reload)

    # 기존 FAISS 모드
    generator = _generator_cache[model]
    sub_id = user_info['id']

    # prompt에서 tags 추출
    tags = None
    for file_key, (prompt_text, parse_data) in prompt.items():
        try:
            parsed_yaml = yaml.safe_load(parse_data)
            tags = parsed_yaml.get('tags', None)
            if tags:
                print(f"✅ Tags 추출 성공: {tags}")
                break
        except Exception as e:
            print(f"⚠️ Tags 추출 실패: {e}")
            
    ctx = generator.embed_changes_only(sub_id, prompt, tags=tags)
    ctx['tags'] = tags
    start_embedding(sub_id, ctx, model, llm_handler)
    return ctx['new_embedding']
    


def remove_prompt(sub_id, model, llm_handler, guid_list=None):
    # BGE 모드 분기
    if _is_bge_mode() and guid_list:
        return remove_prompt_bge(sub_id, guid_list, llm_handler)

    # 기존 FAISS 모드
    generator = _generator_cache[model]
    ctx = generator.embed_changes_only(sub_id)

    start_embedding(sub_id, ctx, model, llm_handler)
    return

def start_embedding(sub_id, ctx, model, llm_handler):
    try:
        loop = asyncio.get_event_loop()

        if loop.is_running():
            loop.create_task(async_embedding_prompt(sub_id, ctx, model, llm_handler))
        else:
            loop.run_until_complete(async_embedding_prompt(sub_id, ctx, model, llm_handler))

    except Exception as e:
        raise RuntimeError(f"Cannot create prompt: {e}")
    
async def async_embedding_prompt(sub_id, ctx, model, llm_handler):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, embedding_prompt, sub_id, ctx, model, llm_handler)

_generator_cache = {}
def embedding_prompt(sub_id, ctx, model, llm_handler):
    global _generator_cache
    print(f"[임베딩 시작] 대상 모델 {model}")
    start_time = time.time()

    generator = _generator_cache[model]
    generator.postprocess_with_faiss_and_kg(sub_id, ctx)
    print(f"[임베딩 걸린 시간] : {time.time() - start_time}")
    
    print(f"[자원 리로드 시작]")
    start_time = time.time()
    
    llm_handler.set_model(model)
    llm_handler.reload_embedding(sub_id=sub_id)
    print(f"[리로드 걸린 시간] : {time.time() - start_time}")

def init_generator(model):
    global _generator_cache

    # BGE 모드에서는 EmbeddingGenerator 불필요 (create_prompt_bge/update_prompt_bge 사용)
    if _is_bge_mode():
        print(f"♻️ [{model.upper()}] BGE 모드: EmbeddingGenerator 초기화 건너뜀 (RAGSystemBGE에서 모델 공유)")
        return

    local_model_path = "../models/bge-m3"

    try:
        print(f"🚀 [{model.upper()}] 임베딩 제너레이터 초기화 시작...")
        
        # 등록된 핸들러면 모두 지원 — 임베딩은 모델 무관하게 BGE-M3(로컬)를 쓰고,
        # 'gpt'(OpenAI API)만 로컬 모델을 쓰지 않는다. 새 핸들러 추가 시 분기 수정 불필요.
        from handler_registry import HANDLER_CLASSES
        if model not in HANDLER_CLASSES:
            raise ValueError(f"지원하지 않는 모델: {model}")
        use_local = model != 'gpt'
        print(f"[{model.upper()} 임베딩 제너레이터 생성]")
        generator = EmbeddingGenerator(
            save_db=True,
            use_local_model=use_local,
            local_model_path=local_model_path if use_local else None,
            batch_db_operations=True
        )
        
        if generator is not None:
            generator.set_performance_profile("ultra")
            _generator_cache[model] = generator
            print(f"✅ [{model.upper()}] 임베딩 제너레이터 초기화 완료")
        else:
            raise Exception(f"{model} Generator 초기화 실패")
            
    except Exception as e:
        print(f"❌ [{model.upper()}] 임베딩 제너레이터 초기화 실패: {str(e)}")
        if model in _generator_cache:
            del _generator_cache[model]
        raise RuntimeError(f"Cannot initialize {model} generator: {e}")

def save_the_yaml(prompt_type, name, description, prompt, guid):
    data = {    
        "type": prompt_type,
        "name": name,
        "description": description,
        "guid": guid,
        "enabled": True,
    }
    
    try:
        parsed_prompt = yaml.safe_load(prompt)
        for k in ("question", "cot", "answer", "spec", "post_action", "tags", "app_code", "require", "aliases"):
            if k in parsed_prompt:
                # aliases, tags는 리스트 유지, 나머지는 문자열 변환
                if k in ("aliases", "tags") and isinstance(parsed_prompt[k], list):
                    data[k] = parsed_prompt[k]
                else:
                    data[k] = str(parsed_prompt[k])
    except Exception as e:
        raise ValueError(f"Invalid prompt format: {e}")

    yaml_str = yaml.safe_dump(
        data, 
        sort_keys=False,
        allow_unicode=True
    )
    return yaml_str
    


def parse_prompt(prompt_type, name, description, prompt, guid):
    parse_data = save_the_yaml(prompt_type.lower(), name, description, prompt, guid)
    return {
        f"{prompt_type.lower()}/{name}.yaml" : (
            prompt,
            parse_data
        )
    }
    
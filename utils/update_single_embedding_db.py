#!/usr/bin/env python3
"""
1번 구독키의 특정 파일 임베딩만 DB에서 직접 업데이트
BGE-M3 모드: Qdrant 벡터 upsert
레거시 모드: DB + FAISS/지식그래프
"""

import sys
import os
import time
from pathlib import Path
import json
import hashlib
import yaml

# 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import ConfigManager, qdrant_collection


def _detect_bge_mode(config: ConfigManager) -> bool:
    """config.ini에서 BGE-M3 모드인지 감지"""
    return config.config.get('embedding', 'use_bge_mode', fallback='False').lower() == 'true'


def _get_docs_dir() -> Path:
    """프로젝트 docs 경로 반환"""
    base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return base / "docs" / "aibot" / "yaml"


def _extract_embedding_text_from_yaml(content: str) -> str:
    """YAML에서 임베딩용 텍스트 추출 (aibot_embedding.py 로직 동일)"""
    try:
        yaml_data = yaml.safe_load(content)
        if not isinstance(yaml_data, dict):
            return content

        embedding_parts = []

        for field, prefix in [
            ('question', '질문'),
            ('cot', '추론과정'),
            ('spec', '명세'),
            ('answer', '답변'),
            ('app_code', '앱코드'),
            ('require', '필수파라미터'),
        ]:
            value = yaml_data.get(field, '')
            if value:
                if field == 'require' and isinstance(value, list):
                    require_text = []
                    for req in value:
                        if isinstance(req, dict):
                            name = req.get('name', '')
                            type_info = req.get('type', '')
                            desc = req.get('description', '')
                            is_optional = req.get('optional', False)
                            default_val = req.get('default', '')
                            example_val = req.get('example', '')
                            param_info = f"{name}({type_info})"
                            if is_optional:
                                param_info += "[선택사항]"
                            if default_val:
                                param_info += f"[기본값:{default_val}]"
                            param_info += f": {desc}"
                            if example_val and not isinstance(example_val, list):
                                param_info += f" 예시:{example_val}"
                            require_text.append(param_info)
                    value = ' '.join(require_text)
                elif isinstance(value, list):
                    value = ' '.join(str(item) for item in value)
                embedding_parts.append(f"{prefix}: {str(value).strip()}")

        for field in ['keywords', 'category', 'tags', 'description']:
            if field in yaml_data and yaml_data[field]:
                value = yaml_data[field]
                if isinstance(value, list):
                    value = ' '.join(str(item) for item in value)
                embedding_parts.append(f"{field}: {str(value)}")

        embedding_text = ' '.join(embedding_parts) if embedding_parts else content

        if len(embedding_text) > 6000:
            embedding_text = embedding_text[:6000]

        return embedding_text

    except yaml.YAMLError:
        return content


def _detect_doc_type(file_path: str) -> str:
    """파일 경로에서 doc_type 추출 (action/xxx.yaml → action)"""
    parts = file_path.split("/")
    if len(parts) >= 2:
        first = parts[0].lower()
        if first in ("action", "plan", "qna", "cve"):
            return first
    return "qna"


def update_single_file_bge(file_path: str, sub_id: int, config: ConfigManager):
    """
    BGE-M3 모드: 단일 파일 임베딩 업데이트 (Qdrant upsert)
    """
    from aibot_embedding_BGE import BGEEmbeddingAdapter, BGEVectorDBManager
    from aibot_embedding_BGE import extract_mode_brand_from_filename, parse_security_ids
    from qdrant_client import models
    from aibot_db_manager import AibotDBManager
    from aibot_db_command import SQL_QUERIES

    docs_dir = _get_docs_dir()
    full_path = docs_dir / file_path
    doc_type = _detect_doc_type(file_path)

    print(f"\n📋 BGE-M3 모드 업데이트:")
    print(f"  - 파일: {file_path}")
    print(f"  - 문서 타입: {doc_type}")
    print(f"  - 전체 경로: {full_path}")

    if not full_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {full_path}")
        return False

    print(f"  - 파일 크기: {full_path.stat().st_size} bytes")

    # 1. 파일 읽기
    with open(full_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # 2. YAML 파싱으로 메타데이터 추출
    parsed_yaml = {}
    try:
        parsed_yaml = yaml.safe_load(file_content) or {}
    except:
        pass

    filename = os.path.basename(file_path)
    name = parsed_yaml.get('name', filename)
    description = parsed_yaml.get('description', '')
    doc_type_from_yaml = parsed_yaml.get('type', doc_type)
    parsed_guid = parsed_yaml.get('guid')
    parsed_tags = parsed_yaml.get('tags')

    # 임베딩용 텍스트 생성 (aibot_embedding.py와 동일한 로직)
    embedding_text = _extract_embedding_text_from_yaml(file_content)
    print(f"  - 임베딩 텍스트 길이: {len(embedding_text)} 문자")

    # 3. BGE-M3 임베딩 생성
    print("\n🧠 BGE-M3 임베딩 생성 중...")
    bge_model_path = config.config.get('embedding', 'bge_model_path', fallback='/data/models/bge-m3')
    start_time = time.time()

    adapter = BGEEmbeddingAdapter(model_path=bge_model_path)
    embedding_result = adapter.generate_embedding(embedding_text)

    embedding_time = time.time() - start_time
    print(f"  - 생성 시간: {embedding_time:.2f}초")

    packed_vector = embedding_result['data'][0]['embedding']
    print(f"  - dense 벡터 차원: {len(packed_vector.get('dense', []))}")

    # 4. Qdrant에 upsert (BGEVectorDBManager.update_document 사용 — API와 동일 경로)
    print("\n📤 Qdrant upsert 중...")

    db_manager = AibotDBManager(config, SQL_QUERIES)
    bge_db = BGEVectorDBManager(config_manager=config, db_manager=db_manager)

    if not bge_db.qdrant_enabled:
        print("❌ Qdrant가 비활성화 상태입니다")
        return False

    client = bge_db.qdrant_client

    # 기존 문서 검색 (file_key로)
    existing_points = client.scroll(
        collection_name=qdrant_collection(),
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="file_key", match=models.MatchValue(value=file_path)),
            ]
        ),
        with_payload=True,
        with_vectors=False,
        limit=10
    )

    existing_ids = [p.id for p in existing_points[0]]

    if existing_ids:
        point_id = existing_ids[0]
        print(f"  - 기존 문서 발견: point_id={point_id}")
    else:
        # 새 ID 할당
        collection_info = client.get_collection(qdrant_collection())
        point_id = (collection_info.points_count or 0) + 1
        print(f"  - 새 문서 추가: point_id={point_id}")

    # update_document 호출 (API update_prompt_bge와 동일)
    t0 = time.time()
    doc = {
        'text': file_content,
        'file_key': file_path,
        'doc_type': doc_type,
        'guid': parsed_guid,
        'enabled': True,
    }
    print(f"  [1] update_document 호출 중...")
    success = bge_db.update_document(sub_id, point_id, packed_vector, doc)
    print(f"  [1] update_document 완료: {time.time()-t0:.2f}초, 성공={success}")

    if not success:
        print("❌ Qdrant upsert 실패")
        return False

    total_time = time.time() - start_time
    print(f"\n✅ 전체 완료!")
    print(f"  - point_id: {point_id}")
    print(f"  - collection: {collection_name}")
    print(f"  - 전체 시간: {total_time:.2f}초")

    # 5. DB (openai_prompts)에도 저장 (체크섬 관리용)
    print("\n💾 DB 체크섬 업데이트...")
    try:
        new_checksum = hashlib.md5(file_content.encode()).hexdigest()
        from aibot_embedding import Embedding
        token_count = embedding_result.get('usage', {}).get('total_tokens', len(embedding_text.split()))

        new_embedding = Embedding(
            token_count=token_count,
            vector=dense_vector  # dense 벡터만 DB에 저장
        )

        existing_db = db_manager.get_single_embedding(str(sub_id), file_path)
        if existing_db:
            db_manager.update_single_embedding(
                subscription_id=str(sub_id), source=file_path,
                embedding=new_embedding, content=file_content, checksum=new_checksum)
        else:
            db_manager.insert_single_embedding(
                subscription_id=str(sub_id), source=file_path,
                embedding=new_embedding, content=file_content, checksum=new_checksum)

        print(f"  ✅ DB 체크섬 저장: {new_checksum[:16]}...")
    except Exception as e:
        print(f"  ⚠️ DB 저장 실패 (Qdrant는 성공): {e}")

    return True


def update_single_file_legacy(file_path: str, sub_id: str, config: ConfigManager, use_gpt: bool):
    """
    레거시 모드: DB + FAISS/지식그래프 업데이트
    """
    from aibot_embedding import EmbeddingGenerator, Embedding
    from aibot_db_manager import AibotDBManager
    from aibot_db_command import SQL_QUERIES

    docs_dir = _get_docs_dir()
    full_path = docs_dir / file_path

    print(f"\n📋 레거시 모드 업데이트:")
    print(f"  - 파일: {file_path}")
    print(f"  - 전체 경로: {full_path}")

    if not full_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {full_path}")
        return False

    print(f"  - 파일 크기: {full_path.stat().st_size} bytes")

    config_file = "config.ini.gpt" if use_gpt else "config.ini"
    db_manager = AibotDBManager(config, SQL_QUERIES)

    # 기존 임베딩 확인
    existing = db_manager.get_single_embedding(sub_id, file_path)

    # 파일 읽기 & 체크섬
    with open(full_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    new_checksum = hashlib.md5(file_content.encode()).hexdigest()

    if existing and existing.get('checksum') == new_checksum:
        print("\n✅ 파일이 변경되지 않았습니다 (체크섬 동일)")
        return True

    # EmbeddingGenerator 초기화
    print("\n🚀 임베딩 생성기 초기화...")
    if use_gpt:
        generator = EmbeddingGenerator(config_path=config_file, save_db=True, use_local_model=False)
    else:
        generator = EmbeddingGenerator(config_path=config_file, save_db=True,
                                       use_local_model=True, local_model_path="/data/models/bge-m3")
    generator.db_manager = db_manager

    # 텍스트 추출
    if file_path.endswith(('.yaml', '.yml')):
        embedding_text = generator._extract_embedding_text_from_yaml(file_content, file_path)
    else:
        embedding_text = file_content

    print(f"  - 추출된 텍스트 길이: {len(embedding_text)} 문자")

    # 임베딩 생성
    print("\n🧠 임베딩 생성 중...")
    start_time = time.time()

    if use_gpt:
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {generator.api_key}",
        }
        data = {"input": embedding_text, "model": generator.model, "encoding_format": "float"}
        response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            embedding_data = {
                'data': [{'embedding': result["data"][0]["embedding"]}],
                'usage': result["usage"],
                'model': generator.model
            }
        else:
            print(f"❌ API 에러: {response.status_code}")
            return False
    else:
        embedding_data = generator.embedding_adapter.generate_embedding(embedding_text)
        if not embedding_data:
            print("❌ 임베딩 생성 실패")
            return False

    embedding_time = time.time() - start_time
    print(f"  - 생성 시간: {embedding_time:.2f}초")

    vector = embedding_data.get('data', [{}])[0].get('embedding')
    if not vector:
        print("❌ 벡터 데이터를 찾을 수 없습니다")
        return False

    # DB 업데이트
    print("\n💾 DB 업데이트 중...")
    token_count = embedding_data.get('usage', {}).get('total_tokens', len(embedding_text.split()))
    new_embedding = Embedding(token_count=token_count, vector=vector)

    if existing:
        success = db_manager.update_single_embedding(sub_id, file_path, new_embedding, file_content, new_checksum)
    else:
        success = db_manager.insert_single_embedding(sub_id, file_path, new_embedding, file_content, new_checksum)

    if not success:
        print("❌ DB 업데이트 실패")
        return False

    # FAISS & 지식그래프 업데이트
    print("\n🔗 FAISS 및 지식그래프 업데이트 중...")
    ctx = {
        "all_files": [file_path], "has_deletions": False, "has_modifications": True,
        "new_files": [], "modified_files": [file_path], "deleted_files": [], "user_prompt": None
    }

    all_subscription_ids = db_manager.get_active_subscription_ids() or ["1"]
    success_count = 0
    for sid in all_subscription_ids:
        try:
            generator.postprocess_with_faiss_and_kg(sid, ctx)
            success_count += 1
            print(f"    ✅ 구독키 {sid} 완료")
        except Exception as e:
            print(f"    ❌ 구독키 {sid} 오류: {e}")

    total_time = time.time() - start_time
    print(f"\n📊 완료: DB {'✅' if success else '❌'}, FAISS/KG {success_count}/{len(all_subscription_ids)}, {total_time:.2f}초")
    return success


def update_single_file_embedding_in_db(file_path: str, sub_id: str = "1", use_gpt: bool = False):
    """
    단일 파일 임베딩 업데이트 (BGE-M3 / 레거시 자동 감지)
    """
    print("=" * 60)
    print("🔄 단일 파일 임베딩 업데이트")
    print("=" * 60)

    config = ConfigManager("config.ini")
    is_bge = _detect_bge_mode(config)

    print(f"  - 모드: {'BGE-M3 (Qdrant)' if is_bge else '레거시 (FAISS/KG)'}")

    if is_bge:
        return update_single_file_bge(file_path, int(sub_id), config)
    else:
        return update_single_file_legacy(file_path, sub_id, config, use_gpt)


if __name__ == "__main__":
    file_path = "action/ticket-open-list.yaml"
    use_gpt = False

    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2].lower() in ['gpt', 'true', '1']:
        use_gpt = True

    print(f"🎯 대상 파일: {file_path}")
    print(f"🤖 모델: {'GPT API' if use_gpt else '로컬/BGE'}")

    success = update_single_file_embedding_in_db(file_path, sub_id="1", use_gpt=use_gpt)

    print("\n" + "=" * 60)
    if success:
        print("✅ 작업 완료")
    else:
        print("❌ 작업 실패")
    print("=" * 60)

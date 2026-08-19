#!/usr/bin/env python3

import os
import sys
import uuid
import asyncio
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import ConfigManager
from aibot_db_manager import AibotDBManager
from aibot_embedding import EmbeddingGenerator

async def init_system():
    print("=== 시스템 초기화 시작 ===")

    config = ConfigManager()
    db_config = config.get_db_config()

    from aibot_db_command import SQL_QUERIES
    db_manager = AibotDBManager(
        config=config,
        query_properties=SQL_QUERIES
    )

    print("1. 데이터베이스 연결 및 테이블 초기화...")
    if not db_manager.initialize_tables():
        print("❌ 데이터베이스 테이블 초기화 실패")
        return False
    print("✅ 데이터베이스 테이블 초기화 완료")

    print("2. 기본 구독키 확인/생성...")

    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, api_key FROM ai_subscriptions WHERE name = %s', ('default',))
                result = cursor.fetchone()

                if result:
                    sub_id = result['id'] if isinstance(result, dict) else result[0]
                    api_key = result['api_key'] if isinstance(result, dict) else result[1]
                    print(f"✅ 기존 구독키 사용 - ID: {sub_id}, API Key: {api_key}")
                else:
                    guid = str(uuid.uuid4())
                    api_key = str(uuid.uuid4())
                    now = datetime.now()
                    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.000")
                    expires_at = (now + timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S.000")

                    insert_sql = """
                        INSERT INTO ai_subscriptions (guid, name, account, description, api_key, acl, created_at, updated_at, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    cursor.execute(insert_sql, (
                        guid, "default", "default", "default", api_key, None,
                        formatted_now, formatted_now, expires_at
                    ))
                    sub_id = cursor.lastrowid
                    conn.commit()
                    print(f"✅ 새 구독키 생성 완료 - ID: {sub_id}, API Key: {api_key}")

    except Exception as e:
        print(f"❌ 구독키 처리 실패: {e}")
        return False

    print("3. 초기 임베딩 생성...")
    try:
        model_config = config.get_model_config()
        model = model_config.get('model', 'gpt-oss')
        use_local_model = model != 'gpt'
        print(f"🤖 Model: {model}, Local model: {use_local_model}")

        local_model_path = None
        if use_local_model:
            paths_config = config.get_paths_config()
            local_model_path = '../models/bge-m3'

        embedding_generator = EmbeddingGenerator(
            use_local_model=use_local_model,
            local_model_path=local_model_path,
            save_db=True
        )
        embedding_generator.set_ultra_performance_profile()

        embedding_generator.generate_embeddings_ultra_fast(sub_id)
        print("✅ 초기 임베딩 생성 완료")

    except Exception as e:
        print(f"❌ 임베딩 생성 중 오류: {e}")
        return False

    print("=== 시스템 초기화 완료 ===")
    return True

if __name__ == "__main__":
    if asyncio.run(init_system()):
        sys.exit(0)
    else:
        sys.exit(1)
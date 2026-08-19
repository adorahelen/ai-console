#!/usr/bin/env python3
"""
DB에서 특정 파일의 임베딩 확인
"""

import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aibot_db_manager import AibotDBManager
from config_utils import ConfigManager

def check_embedding(file_path: str, sub_id: str = "1", use_gpt: bool = False):
    """특정 파일의 임베딩 정보 확인"""

    print("=" * 60)
    print(f"🔍 DB 임베딩 조회")
    print("=" * 60)

    # GPT 모드에 따라 다른 설정 파일 사용
    if use_gpt:
        config_file = "config.ini.gpt"
        print(f"설정 파일: {config_file} (GPT 모드)")
    else:
        config_file = "config.ini"
        print(f"설정 파일: {config_file} (로컬 모델)")

    config = ConfigManager(config_file)
    db_manager = AibotDBManager(config, {})

    # DB에서 조회
    result = db_manager.get_single_embedding(sub_id, file_path)

    if result:
        print(f"\n✅ 임베딩 발견:")
        print(f"  - ID: {result.get('id')}")
        print(f"  - Source: {result.get('source')}")
        print(f"  - Token Count: {result.get('token_count')}")
        print(f"  - Vector Size: {result.get('vector_size')} bytes")
        print(f"  - Vector Dim: {result.get('vector_dim')}")
        print(f"  - Created: {result.get('created_at')}")
        print(f"  - Updated: {result.get('updated_at')}")
        print(f"  - Checksum: {result.get('checksum', 'N/A')[:16]}...")

        # 저장된 컨텐츠 일부 확인 (원본 영어 필드명 확인용)
        prompt = result.get('prompt', '')
        if prompt:
            print(f"\n📄 저장된 컨텐츠 (첫 200자):")
            print(f"  {prompt[:200]}...")
            # 영어 필드명 확인
            if 'question:' in prompt and '질문:' not in prompt:
                print(f"  ✅ 원본 영어 필드명이 올바르게 저장됨")
            elif '질문:' in prompt:
                print(f"  ⚠️ 번역된 필드명이 저장됨 (수정 필요)")
    else:
        print(f"\n❌ 임베딩 없음: {file_path}")

    # 전체 action 폴더의 임베딩 확인
    print("\n" + "=" * 60)
    print("📊 action 폴더 임베딩 통계")
    print("=" * 60)

    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # action 폴더 파일 수 확인
                query = """
                    SELECT COUNT(*) as count
                    FROM openai_prompts
                    WHERE subscription_id = %s
                    AND source LIKE 'action/%%'
                """
                cursor.execute(query, (sub_id,))
                count_result = cursor.fetchone()
                print(f"  - action 폴더 임베딩 수: {count_result['count']}개")

                # 최근 업데이트된 action 파일들
                query = """
                    SELECT source, token_count, updated_at
                    FROM openai_prompts
                    WHERE subscription_id = %s
                    AND source LIKE 'action/%%'
                    ORDER BY updated_at DESC
                    LIMIT 5
                """
                cursor.execute(query, (sub_id,))
                recent = cursor.fetchall()

                if recent:
                    print("\n  최근 업데이트된 action 파일:")
                    for r in recent:
                        print(f"    - {r['source']} (토큰: {r['token_count']}, 업데이트: {r['updated_at']})")

    except Exception as e:
        print(f"❌ 통계 조회 실패: {e}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "action/ticket-open-list.yaml"
    sub_id = sys.argv[2] if len(sys.argv) > 2 else "1"
    use_gpt = len(sys.argv) > 3 and sys.argv[3].lower() in ['gpt', 'true', '1']

    check_embedding(file_path, sub_id, use_gpt)
#!/usr/bin/env python3
"""
1번 구독키의 증분 모드 작동 확인
"""

import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aibot_embedding import EmbeddingGenerator
from pathlib import Path

def test_incremental_mode():
    print("=" * 60)
    print("🔍 1번 구독키 증분 모드 테스트")
    print("=" * 60)

    # 구독키 1번 설정
    sub_id = "1"

    # EmbeddingGenerator 초기화
    generator = EmbeddingGenerator(
        save_db=True,
        use_local_model=True,
        local_model_path="/data/models/bge-m3"
    )

    # 메타데이터 파일 경로 확인
    model = 'general' if generator.use_local_model else 'gpt'
    metadata_file = generator.embedding_dir / f"metadata_{model}_{sub_id}.json"

    print(f"\n📋 테스트 정보:")
    print(f"  - 구독 ID: {sub_id}")
    print(f"  - 모델: {model}")
    print(f"  - 메타데이터 파일: {metadata_file}")
    print(f"  - 파일 존재: {'✅ 예' if metadata_file.exists() else '❌ 아니오'}")

    # analyze_db_changes 실행해서 증분 모드 확인
    print("\n🔄 변경사항 분석 중...")
    new_files, deleted_files, modified_files, current_metadata = generator.analyze_db_changes(sub_id, model)

    print(f"\n📊 분석 결과:")
    print(f"  - 새 파일: {len(new_files)}개")
    print(f"  - 삭제된 파일: {len(deleted_files)}개")
    print(f"  - 수정된 파일: {len(modified_files)}개")
    print(f"  - 현재 메타데이터: {len(current_metadata)}개")

    # 증분 모드 작동 여부 판단
    is_initial_run = not metadata_file.exists()

    if is_initial_run:
        print("\n⚠️ 초기 실행 모드:")
        print("  - 메타데이터 파일이 없어 모든 파일을 처리합니다")
        print("  - 증분 모드가 작동하지 않습니다")
    else:
        print("\n✅ 증분 모드 활성화:")
        print("  - 메타데이터 파일이 존재합니다")
        print("  - 변경된 파일만 처리됩니다")

    # embed_changes_only 테스트
    print("\n🚀 embed_changes_only 실행 테스트...")
    try:
        result = generator.embed_changes_only(sub_id)

        print("\n📦 실행 결과:")
        print(f"  - 처리된 임베딩: {len(result.get('new_embedding', {}))}개")
        print(f"  - has_modifications: {result.get('has_modifications')}")
        print(f"  - has_deletions: {result.get('has_deletions')}")

        # 메타데이터 파일 생성 확인
        if metadata_file.exists():
            print(f"\n✅ 메타데이터 파일 생성됨: {metadata_file}")

            # 파일 내용 확인
            with open(metadata_file, 'r') as f:
                import json
                metadata = json.load(f)
                print(f"  - 저장된 파일 수: {len(metadata)}개")

    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_incremental_mode()
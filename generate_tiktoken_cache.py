#!/usr/bin/env python3
"""
generate_tiktoken_cache.py — tiktoken 인코딩을 OfflineTiktoken 형식의 pickle 로 미리 캐시.

오프라인 환경에서 install.sh 의 7단계 init_system 가 BGE 인덱싱 시 tiktoken 을
사용하는데, default tiktoken 은 처음 호출 시 네트워크에서 BPE merge 데이터를
다운로드한다. 인터넷 끊긴 환경에선 이 단계가 fail 한다.

build_package.sh 는 ``tiktoken_cache/`` 를 source.tar.gz 에 포함시키므로,
빌드머신에서 한 번 이 스크립트를 실행해서 캐시를 채워두면 타겟에 그대로 전달된다.

사용법:
  python generate_tiktoken_cache.py
  python generate_tiktoken_cache.py --encodings cl100k_base o200k_base
  python generate_tiktoken_cache.py --output-dir /custom/path
"""

import argparse
import pickle
import sys
from pathlib import Path

DEFAULT_ENCODINGS = ["cl100k_base", "o200k_base", "p50k_base", "r50k_base"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encodings", nargs="+", default=DEFAULT_ENCODINGS,
        help=f"수집할 tiktoken 인코딩 (default: {' '.join(DEFAULT_ENCODINGS)})",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent / "tiktoken_cache"),
        help="결과 디렉토리 (default: <repo>/tiktoken_cache)",
    )
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError:
        print("tiktoken 미설치. conda env 활성화 후 재실행:", file=sys.stderr)
        print("  conda activate deploy && pip install tiktoken", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for name in args.encodings:
        out_file = out_dir / f"{name}.pkl"
        if out_file.exists():
            print(f"  · {name}: 이미 존재 — 건너뜀")
            continue
        try:
            enc = tiktoken.get_encoding(name)
            with open(out_file, "wb") as f:
                pickle.dump({
                    "name": enc.name,
                    "_pat_str": enc._pat_str,
                    "_mergeable_ranks": enc._mergeable_ranks,
                    "_special_tokens": enc._special_tokens,
                }, f)
            print(f"  ✓ {name} → {out_file}")
        except Exception as e:
            print(f"  ✗ {name}: {e}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"\n실패: {failed} (인터넷 연결/tiktoken 버전 확인)", file=sys.stderr)
        return 1
    print(f"\n완료: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Gemma 4 프롬프트 적응 검증용 룰 기반 verdict 검출기.

사용법:
    python verdict_gemma.py --intent qna --file answer.txt
    cat answer.txt | python verdict_gemma.py --intent action --stdin
    python verdict_gemma.py --selftest

모듈로:
    from verdict_gemma import check
    failures = check(answer_text, intent="qna")  # [] 면 pass

검출 룰 (codex 권고 정규식 6종):
    1. token_leak        — <think>, <start_of_turn>, <|channel>thought 등 챗 템플릿/사고 토큰 누수
    2. action_json_in_qna — QNA 답변에 ACTION-style JSON ("action":"query|rest-api|clarify") 침범
    3. where_in_query    — query 블록에서 'where' 사용 (대상 쿼리 언어는 'search' 사용)
    4. uppercase_boolean — query 블록에서 'AND'/'OR' 대문자 (대상 쿼리 언어는 소문자)
    5. forbidden_command — query 블록에서 head/order/strftime
    6. string_ip_compare — query 블록에서 ip() 없이 문자열로 IP 비교
"""
import re
import sys
import json
import argparse
from typing import List, Tuple

TOKEN_LEAK_PATTERN = re.compile(
    r"<think>|</think>|<thought>|</thought>|<\|/?think\|?>|"
    r"<start_of_turn>|<end_of_turn>|<bos>|<eos>|"
    r"<\|?channel\|?>thought|<channel\|?>",
    re.IGNORECASE,
)

ACTION_JSON_PATTERN = re.compile(
    r'"action"\s*:\s*"(query|rest-api|clarify|alert|incident)"',
    re.IGNORECASE,
)

QUERY_BLOCK_PATTERN = re.compile(r"```query\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

WHERE_IN_QUERY_PATTERN = re.compile(r"\bwhere\s+\w", re.IGNORECASE)
UPPERCASE_BOOL_PATTERN = re.compile(r"(?:\s|[)\"'])(AND|OR)(?=\s)")
FORBIDDEN_CMD_PATTERN = re.compile(r"\b(head|order)\s+|strftime\(", re.IGNORECASE)
STRING_IP_PATTERN = re.compile(
    r'(?:src|dst|client|server|peer)_ip\s*==\s*"\d{1,3}(?:\.\d{1,3}){3}"'
)


def check(text: str, intent: str = "qna") -> List[Tuple[str, str]]:
    """답변 텍스트를 검사. (rule_name, snippet) 리스트 반환. [] 면 pass."""
    failures: List[Tuple[str, str]] = []
    intent = (intent or "qna").lower()

    m = TOKEN_LEAK_PATTERN.search(text)
    if m:
        failures.append(("token_leak", m.group(0)))

    if intent == "qna":
        m = ACTION_JSON_PATTERN.search(text)
        if m:
            failures.append(("action_json_in_qna", m.group(0)))

    for qm in QUERY_BLOCK_PATTERN.finditer(text):
        qbody = qm.group(1)
        first_line = qbody.strip().split("\n")[0][:120]
        if WHERE_IN_QUERY_PATTERN.search(qbody):
            failures.append(("where_in_query", first_line))
        if UPPERCASE_BOOL_PATTERN.search(qbody):
            failures.append(("uppercase_boolean", first_line))
        if FORBIDDEN_CMD_PATTERN.search(qbody):
            failures.append(("forbidden_command", first_line))
        if STRING_IP_PATTERN.search(qbody):
            failures.append(("string_ip_compare", first_line))

    return failures


def _selftest() -> int:
    cases = [
        # (label, intent, text, expected_failure_rule_or_None)
        ("clean_qna", "qna", "방화벽 통신 조회는 ```query\ntable duration=1h FW | search src_ip == ip(\"1.2.3.4\")\n```", None),
        ("token_leak_think", "qna", "답변 <think>reasoning</think> 본문", "token_leak"),
        ("token_leak_turn", "qna", "<start_of_turn>user 본문", "token_leak"),
        ("action_in_qna", "qna", '응답: {"action": "query", "query": "..."}', "action_json_in_qna"),
        ("where_in_q", "qna", "```query\ntable FW | where count > 3\n```", "where_in_query"),
        ("uppercase_AND", "qna", "```query\nsearch a == 1 AND b == 2\n```", "uppercase_boolean"),
        ("head_command", "qna", "```query\ntable FW | head 10\n```", "forbidden_command"),
        ("string_ip", "qna", "```query\nsearch src_ip == \"1.2.3.4\"\n```", "string_ip_compare"),
        ("ip_func_ok", "qna", "```query\nsearch src_ip == ip(\"1.2.3.4\")\n```", None),
        ("action_mode_ok", "action", '{"action": "query", "query": "..."}', None),  # ACTION 모드는 OK
    ]
    fail = 0
    for label, intent, text, expected in cases:
        result = check(text, intent=intent)
        rules = [r for r, _ in result]
        if expected is None:
            if rules:
                print(f"❌ {label}: expected pass, got {rules}")
                fail += 1
            else:
                print(f"✅ {label}: pass")
        else:
            if expected in rules:
                print(f"✅ {label}: detected {expected}")
            else:
                print(f"❌ {label}: expected {expected!r}, got {rules}")
                fail += 1
    return fail


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma 룰 기반 verdict 검출기")
    parser.add_argument("--intent", choices=["qna", "action", "plan"], default="qna")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--file", help="검사할 답변 텍스트 파일 경로")
    src.add_argument("--stdin", action="store_true", help="stdin 으로 입력")
    src.add_argument("--selftest", action="store_true", help="내부 셀프 테스트 실행")
    parser.add_argument("--json", action="store_true", help="JSON 형식 출력")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.stdin:
        text = sys.stdin.read()
    else:
        parser.error("--file, --stdin, --selftest 중 하나는 필요")

    failures = check(text, intent=args.intent)

    if args.json:
        print(json.dumps(
            {"intent": args.intent, "failures": failures, "pass": not failures},
            ensure_ascii=False,
        ))
        return 1 if failures else 0

    if failures:
        print(f"❌ FAIL ({len(failures)} rule violation(s)) intent={args.intent}")
        for name, snippet in failures:
            print(f"   - {name}: {snippet}")
        return 1
    print(f"✅ PASS intent={args.intent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

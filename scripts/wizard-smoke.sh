#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# wizard-smoke.sh — Phase 3 온보딩 위저드 종단 스모크 (testing-guide V4·V5의 기계화)
#
#   ./scripts/wizard-smoke.sh            # 4단계 순차 호출 후 테스트 카트리지 정리
#   ./scripts/wizard-smoke.sh --keep     # 생성된 테스트 카트리지 보존(수동 확인용)
#   WIZARD_SMOKE_TIMEOUT=600 ./scripts/wizard-smoke.sh   # CPU 8B(2~5분/건)용 타임아웃 상향
#
# 목적: 클릭 없이 CLI로 위저드 4엔드포인트(prompt-draft→test-chat→knowledge-convert
#       →cartridge-save)를 활성 핸들러로 실제 구동해, "위저드 글루가 도는가"를 실측한다.
# 전제: 콘솔이 기동돼 있어야 함(./run.sh start, repo 루트에서). 모델 로딩 중이면 503 재시도로 대기.
# 범위: 활성 핸들러(설치 프리셋)만 검증. handler_llama(CPU 8B) 경로는 그 프리셋으로 실행해야 커버됨.
# 정리: 테스트 카트리지(cartridges/wizard-smoke)는 trap으로 항상 삭제(--keep면 보존).
#       .gitignore에도 등재돼 잔존 시에도 오커밋 안 됨. 커밋 대상 아님.
# ═══════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "repo 루트 이동 실패"; exit 1; }

say()  { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*"; exit 1; }

PY=python3; [ -x .venv/bin/python3 ] && PY=.venv/bin/python3
BASE="https://localhost:8443"
MAXT="${WIZARD_SMOKE_TIMEOUT:-300}"   # CPU 8B는 draft 1건 2~5분 — 낮추면 false-fail (guide V5)
KEEP=0; [ "${1:-}" = "--keep" ] && KEEP=1
SMOKE_NAME="wizard-smoke"

# 모든 핸들러 계열의 chat template 누출 토큰 — gemma(<start_of_turn>…) · gpt-oss(<|channel|>…) ·
# llama(<|eot_id|>…). 좁게 <|헤더만 잡으면 배포 gemma 누출을 놓친다(리뷰 finding).
LEAK_RE='<(start_of_turn|end_of_turn|eos|bos)>|<\|(start_header_id|end_header_id|eot_id|start|end|message|channel|return|endoftext)'

# 잔존 카트리지 정리 — trap으로 중단 시에도 실행(오커밋·다음 실행 409 방지)
cleanup() { [ "$KEEP" = 1 ] && return 0; [ -d "cartridges/$SMOKE_NAME" ] && rm -rf "cartridges/${SMOKE_NAME:?}"; }
trap cleanup EXIT INT TERM

KEY=$(cat api_keys/default.key 2>/dev/null | tr -d '\r\n' | tr -d '[:space:]') || true
[ -n "$KEY" ] || die "api_keys/default.key 없음/공백 — 콘솔 설치 후 실행"
AUTH="Authorization: Bearer $KEY"
CT="Content-Type: application/json"

# 콘솔 준비 대기(FastAPI 바인드) — 단발 대신 짧은 재시도(로딩 직후 실행 대비)
for _ in $(seq 1 12); do
  [ "$(curl -sk -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/docs")" = "200" ] && { READY=1; break; }
  sleep 5
done
[ "${READY:-0}" = 1 ] || die "콘솔 미기동($BASE/docs 무응답) — ./run.sh start 후 재실행"

FAIL=0
LAST_CODE=""
bad_field() { case "$1" in ""|__MISSING__|__PARSEFAIL__*) return 0;; *) return 1;; esac; }

# 단일 키 추출(플랫 키만 사용) — 파싱 실패/누락은 센티널
jget() { $PY -c "import sys,json
try: d=json.load(sys.stdin)
except Exception as e: print('__PARSEFAIL__'); sys.exit()
v=d.get('$1') if isinstance(d,dict) else None
print(v if v is not None else '__MISSING__')"; }

# POST: 본문을 stdout, HTTP 코드를 LAST_CODE로. 503(모델 로딩)은 짧게 재시도.
# FAIL 집계는 여기서 하지 않는다 — 호출부 필드검사가 유일 집계점(이중집계 방지).
# 본문·코드는 전역 RESP/LAST_CODE로 반환한다 — $(POST) 서브셸에 두면 LAST_CODE가
# 부모로 전파되지 않아 항상 빈값이 된다(리뷰 반영본의 런타임 버그, 실측으로 발견).
POST() {
  local body i
  for i in 1 2 3 4 5; do
    body=$(curl -sk -X POST "$BASE$1" -H "$AUTH" -H "$CT" -d "$2" --max-time "$MAXT" -w $'\n%{http_code}')
    LAST_CODE="${body##*$'\n'}"; body="${body%$'\n'*}"
    [ "$LAST_CODE" = 503 ] || break
    [ "$i" = 5 ] && break
    warn "$1 → 503(모델 로딩 중) 재시도 $i/4"; sleep 10
  done
  RESP="$body"
}

# 응답 요약 경고(비200 시) — 키 노출 없이 본문만
http_warn() { [ "$LAST_CODE" != 200 ] && warn "$1 → HTTP $LAST_CODE: $(printf '%s' "$2" | head -c 200)"; }

# ── 1) prompt-draft: 5문답 → 시스템 프롬프트 초안 ──
say "1/4 prompt-draft (5문답 → 프롬프트 초안 생성)"
POST /api/wizard/prompt-draft '{"role":"사내 보안 관제 도우미","audience":"보안팀 신입","tone":"정중하고 간결하게","rules":"확실하지 않으면 모른다고 답한다","needs_action":false}'; R1="$RESP"
http_warn /api/wizard/prompt-draft "$R1"
DRAFT=$(printf '%s' "$R1" | jget draft)
if bad_field "$DRAFT"; then warn "prompt-draft: draft 없음(HTTP $LAST_CODE)"; FAIL=$((FAIL+1)); DRAFT="당신은 보안 관제 도우미입니다."
elif printf '%s' "$DRAFT" | grep -qE "$LEAK_RE"; then warn "prompt-draft: 초안에 특수토큰 누출 — chat template 확인"; FAIL=$((FAIL+1))
else ok "초안 ${#DRAFT}자 생성, 토큰 누출 없음"; fi

# ── 2) test-chat: 초안 프롬프트로 실제 대화 ──
say "2/4 test-chat (초안으로 테스트 대화)"
CHAT_REQ=$($PY -c "import json,sys; print(json.dumps({'system_prompt':sys.argv[1],'message':'너는 무슨 일을 도와줘?','history':[]}))" "$DRAFT")
POST /api/wizard/test-chat "$CHAT_REQ"; R2="$RESP"
http_warn /api/wizard/test-chat "$R2"
REPLY=$(printf '%s' "$R2" | jget reply)
if bad_field "$REPLY"; then warn "test-chat: reply 없음(HTTP $LAST_CODE)"; FAIL=$((FAIL+1))
elif printf '%s' "$REPLY" | grep -qE "$LEAK_RE"; then warn "test-chat: 응답에 특수토큰 누출 — chat template 확인"; FAIL=$((FAIL+1))
else ok "응답 ${#REPLY}자, 토큰 누출 없음"; fi

# ── 3) knowledge-convert: 문서 → qna 지식 초안 (본문 1회만 파싱) ──
say "3/4 knowledge-convert (문서 → qna 변환)"
DOC='이 콘솔에서 stats 명령은 필드별 집계에 쓴다. 예: stats count by src_ip 는 출발지 IP별 건수를 센다. sort 명령은 결과 정렬에 쓰며 sort -count 는 내림차순이다.'
KC_REQ=$($PY -c "import json,sys; print(json.dumps({'text':sys.argv[1],'count':3}))" "$DOC")
POST /api/wizard/knowledge-convert "$KC_REQ"; R3="$RESP"
http_warn /api/wizard/knowledge-convert "$R3"
KOUT=$(printf '%s' "$R3" | $PY -c "import sys,json
try: d=json.load(sys.stdin); it=d.get('items',[])
except Exception: it=[]
if not isinstance(it,list): it=[]   # 소형모델이 dict/str 뱉는 경우 방어
print(len(it)); print(json.dumps(it[:3], ensure_ascii=False))")
NITEMS=$(printf '%s' "$KOUT" | sed -n 1p); ITEMS_JSON=$(printf '%s' "$KOUT" | sed -n 2p)
[ -n "$ITEMS_JSON" ] || ITEMS_JSON='[]'
case "$NITEMS" in ''|*[!0-9]*) NITEMS=0;; esac
if [ "$LAST_CODE" != 200 ]; then warn "knowledge-convert: 서버 오류(HTTP $LAST_CODE), qna 변환 실패"; FAIL=$((FAIL+1))
elif [ "$NITEMS" -ge 1 ]; then ok "qna ${NITEMS}건 추출"
else warn "knowledge-convert: 200이나 qna 0건 — 파싱/변환 실패"; FAIL=$((FAIL+1)); fi

# ── 4) cartridge-save: 프롬프트+지식(3단계 결과 재사용)을 카트리지로 저장 ──
say "4/4 cartridge-save (카트리지 파일 생성)"
[ -d "cartridges/$SMOKE_NAME" ] && rm -rf "cartridges/${SMOKE_NAME:?}"   # 서버측 잔존 대비 사전 정리
SAVE_REQ=$($PY -c "import json,sys
name, items_json = sys.argv[1], sys.argv[2]
try: items = json.loads(items_json)
except Exception: items = []
print(json.dumps({'name':name,'description':'wizard smoke','system_prompt':'스모크 테스트 프롬프트','knowledge':items,'model_preset':''}))" "$SMOKE_NAME" "$ITEMS_JSON")
POST /api/wizard/cartridge-save "$SAVE_REQ"; R4="$RESP"
http_warn /api/wizard/cartridge-save "$R4"
SAVED=$(printf '%s' "$R4" | jget saved)
if bad_field "$SAVED"; then warn "cartridge-save: 저장 실패(HTTP $LAST_CODE)"; FAIL=$((FAIL+1))
else
  # 저장은 서버 프로세스 cwd 기준(repo 루트 가정) — 파일·YAML 유효성까지 확인
  CART="cartridges/$SMOKE_NAME"
  if [ ! -f "$CART/cartridge.yaml" ] || [ ! -f "$CART/prompts/system.txt" ]; then
    warn "cartridge-save: 200이나 파일 없음(서버 cwd 불일치 가능 — run.sh를 repo 루트에서 기동?)"; FAIL=$((FAIL+1))
  else
    NYAML=$(ls "$CART"/knowledge/*.yaml 2>/dev/null | wc -l)
    EXPECT=$([ "$NITEMS" -lt 3 ] && echo "$NITEMS" || echo 3)
    if ! $PY -c "import yaml,sys,glob
yaml.safe_load(open('$CART/cartridge.yaml',encoding='utf-8'))
for f in glob.glob('$CART/knowledge/*.yaml'): yaml.safe_load(open(f,encoding='utf-8'))" 2>/dev/null; then
      warn "cartridge-save: 생성된 YAML 파싱 실패(손상)"; FAIL=$((FAIL+1))
    elif [ "$EXPECT" -ge 1 ] && [ "$NYAML" -lt "$EXPECT" ]; then
      warn "cartridge-save: 지식 ${EXPECT}건 기대인데 ${NYAML}건만 저장 — 페이로드 누락"; FAIL=$((FAIL+1))
    else ok "카트리지 생성됨 (manifest+프롬프트 유효, 지식 ${NYAML}건)"; fi
  fi
fi

# 정리(trap도 보장하나 정상 경로 메시지용)
if [ "$KEEP" = 1 ]; then warn "테스트 카트리지 보존: cartridges/$SMOKE_NAME (수동 삭제 필요)"
else cleanup && ok "테스트 카트리지 정리됨"; fi

echo
if [ "$FAIL" = 0 ]; then
  ok "위저드 종단 스모크 통과 (4/4) — 활성 핸들러 기준 Phase 3 web-UI 실동작 확인"
  warn "범위: handler_llama(CPU 8B) chat template은 미포함 — 그 프리셋으로 재실행해야 V5 완전 커버"
else die "위저드 스모크 실패 $FAIL건 — 위 warn 확인 (핸들러 로그: qa_llm_*.log)"; fi

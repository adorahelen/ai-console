#!/bin/bash
# =====================================================================
# install_compose.sh — 타겟 서버용 (air-gap 가능)
#
# build_images.sh 가 만든 패키지(이미지 tar + compose + config)를 받아
# 타겟 서버에서 한 번에 설치 + 기동 + 인덱싱.
#
# 사전 요구 (이 스크립트 실행 전 host 에 설치 필요):
#   - NVIDIA driver 535+ (nvidia-smi 동작)
#   - Docker 24+ (docker version)
#   - nvidia-container-toolkit (sudo nvidia-ctk runtime configure --runtime=docker)
#   - docker compose plugin v2+ (docker compose version)
#
# 사용:
#   sudo bash install_compose.sh                          # 기본
#   sudo bash install_compose.sh --skip-load              # 이미지 이미 load 됨 (재실행)
#   sudo bash install_compose.sh --skip-init              # init_system 안 돌림
#   sudo bash install_compose.sh --models-dir /data/models
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_LOAD=false
SKIP_INIT=false
LOAD_ALL_TAGS=false
MODELS_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-load)  SKIP_LOAD=true; shift ;;
        --skip-init)  SKIP_INIT=true; shift ;;
        # --all-tags: .env 의 AI_CONSOLE_TAG 무관하게 images/*.tar 전체 load (디스크 + 시간 소모)
        --all-tags)   LOAD_ALL_TAGS=true; shift ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --help|-h)
            sed -n '3,/^# ==/p' "${BASH_SOURCE[0]}" | sed 's/^# *//;s/^#$//'
            exit 0 ;;
        *) echo "알 수 없는 인자: $1"; exit 1 ;;
    esac
done

log()  { echo -e "\033[1;34m[install]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[ ✓ ]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m   $*" >&2; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; exit 1; }

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------
# (1) 사전 요구 검증
# ---------------------------------------------------------------------
log "[1/6] 사전 요구사항 검증"

# .env 가 있으면 AI_CONSOLE_TAG 보고 driver minimum 결정 (cu130 → 575+, cu128 → 535+)
AI_CONSOLE_TAG_GUESS=cu128
if [ -f .env ]; then
    v=$(grep -E '^AI_CONSOLE_TAG=' .env | cut -d= -f2 | tr -d '"' | tr -d "'")
    [ -n "$v" ] && AI_CONSOLE_TAG_GUESS="$v"
fi
DRIVER_MIN=535
[ "$AI_CONSOLE_TAG_GUESS" = "cu130" ] && DRIVER_MIN=575

PREREQ_HINT="설치 자동화는 이 디렉토리의 install_host_prereqs.sh:
    sudo bash install_host_prereqs.sh --cuda $AI_CONSOLE_TAG_GUESS -y
  (오프라인이면) bash install_host_prereqs.sh --info  로 다운로드 목록 출력"

command -v nvidia-smi >/dev/null 2>&1 || fail "NVIDIA driver 없음.
  $PREREQ_HINT"

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
[ "$DRIVER_MAJOR" -ge "$DRIVER_MIN" ] || fail "driver $DRIVER_VER < $DRIVER_MIN ($AI_CONSOLE_TAG_GUESS 이미지 최소요건). 업그레이드 필요.
  $PREREQ_HINT"
ok  "NVIDIA driver $DRIVER_VER ✓ (>= $DRIVER_MIN, $AI_CONSOLE_TAG_GUESS)"

command -v docker >/dev/null 2>&1 || fail "docker 미설치.
  $PREREQ_HINT"

if ! docker compose version >/dev/null 2>&1; then
    fail "docker compose plugin 미설치.
  $PREREQ_HINT"
fi
ok  "docker $(docker version --format '{{.Server.Version}}') + compose plugin ✓"

# nvidia-container-toolkit 검증 (docker 가 nvidia runtime 인식)
if ! docker info 2>/dev/null | grep -q -i "nvidia"; then
    fail "docker 가 nvidia runtime 등록 안 됨 (nvidia-container-toolkit 미설치 또는 미구성).
  $PREREQ_HINT"
fi
ok  "nvidia-container-toolkit ✓"

# ---------------------------------------------------------------------
# (2) 이미지 load
# ---------------------------------------------------------------------
if [ "$SKIP_LOAD" = false ]; then
    log "[2/6] 이미지 load (docker load) — AI_CONSOLE_TAG=$AI_CONSOLE_TAG_GUESS"
    [ -d "images" ] || fail "images/ 디렉토리 없음"

    if [ "$LOAD_ALL_TAGS" = true ]; then
        # 전부 load (--all-tags)
        for tar in images/*.tar; do
            [ -f "$tar" ] || continue
            log "  → $(basename "$tar")"
            docker load -i "$tar" | tail -1
        done
    else
        # 인프라(mariadb / qdrant) 는 항상 load
        for base in mariadb.tar qdrant.tar; do
            f="images/$base"
            if [ -f "$f" ]; then
                log "  → $base"
                docker load -i "$f" | tail -1
            fi
        done
        # 선택된 AI_CONSOLE_TAG 만 load
        loaded=0
        for f in "images/ai-console-${AI_CONSOLE_TAG_GUESS}.tar" "images/ai-console-llama-server-${AI_CONSOLE_TAG_GUESS}.tar"; do
            if [ -f "$f" ]; then
                log "  → $(basename "$f")"
                docker load -i "$f" | tail -1
                loaded=$((loaded + 1))
            else
                warn "  $(basename "$f") 없음 (다른 tag 만 패키지에 있을 수 있음)"
            fi
        done
        [ "$loaded" -eq 2 ] || fail "$AI_CONSOLE_TAG_GUESS 이미지 누락. 패키지에 다른 tag만 있다면:
    --all-tags 로 전체 load 후 .env 의 AI_CONSOLE_TAG 다시 확인"
        # 다른 tag 의 tar 가 있는지 안내
        for other in images/ai-console-*.tar; do
            [ -f "$other" ] || continue
            base=$(basename "$other")
            case "$base" in
                "ai-console-${AI_CONSOLE_TAG_GUESS}.tar"|"ai-console-llama-server-${AI_CONSOLE_TAG_GUESS}.tar") ;;
                *) log "  (skip $base — AI_CONSOLE_TAG=$AI_CONSOLE_TAG_GUESS 와 무관, --all-tags 면 함께 load)" ;;
            esac
        done
    fi
    ok  "이미지 load 완료"
else
    log "[2/6] --skip-load: 이미지 load 건너뜀"
fi

# ---------------------------------------------------------------------
# (3) .env / config.ini.docker 검증 (없으면 .example 에서 복사 후 안내)
# ---------------------------------------------------------------------
log "[3/6] config 검증"

if [ ! -f .env ]; then
    cp .env.example .env
    warn ".env 새로 생성. MARIADB_ROOT_PASSWORD / MODELS_DIR 등 편집 후 재실행하세요"
    exit 1
fi

# 비밀번호가 CHANGE_ME 그대로면 기동 거부 (기본 비밀번호 방치 방지)
if grep -qE '^MARIADB_(ROOT_PASSWORD|DB_PASSWORD)=.*CHANGE_ME' .env; then
    warn "MARIADB_*_PASSWORD가 CHANGE_ME 그대로입니다 — .env를 편집한 뒤 재실행하세요"
    exit 1
fi

if [ ! -f config.ini.docker ]; then
    cp config.ini.docker.example config.ini.docker
    warn "config.ini.docker 새로 생성 (plain text — 첫 docker compose up 시 ENC 자동 변환)"
fi

# .encryption_key 부트스트랩
# - 컨테이너의 config_utils.py 가 utils/.encryption_key 로 32-byte XOR 키를 읽음.
# - host 의 utils-docker/.encryption_key 가 file-mount 되는데:
#   · 없으면 docker 가 자동으로 디렉토리로 생성해버림 → IsADirectoryError
#   · 빈 파일이면 len(key)=0 → ZeroDivisionError
#   둘 다 위험하니 32-byte 랜덤 키로 미리 채워둠. (이미 채워져 있으면 그대로 둠)
mkdir -p utils-docker
if [ ! -s utils-docker/.encryption_key ] || [ -d utils-docker/.encryption_key ]; then
    rm -rf utils-docker/.encryption_key
    python3 -c "import os; open('utils-docker/.encryption_key','wb').write(os.urandom(32))" \
        || head -c 32 /dev/urandom > utils-docker/.encryption_key
    chmod 600 utils-docker/.encryption_key
    ok  "  utils-docker/.encryption_key 생성 (32 bytes)"
fi

# .env 의 MODELS_DIR 결정
if [ -z "$MODELS_DIR" ]; then
    MODELS_DIR=$(grep -E '^MODELS_DIR=' .env | cut -d= -f2 | tr -d '"' || echo /service/models)
    MODELS_DIR="${MODELS_DIR:-/service/models}"
fi
ok  ".env / config.ini.docker ✓ (MODELS_DIR=$MODELS_DIR)"

# ---------------------------------------------------------------------
# (4) 모델 디렉토리 검증
# ---------------------------------------------------------------------
log "[4/6] 모델 디렉토리 검증 ($MODELS_DIR)"

[ -d "$MODELS_DIR" ] || fail "$MODELS_DIR 없음. 모델 파일 (~22GB) 별도 옮긴 후 재실행:
    mkdir -p $MODELS_DIR
    cp -r /usb/models/* $MODELS_DIR/"

for sub in bge-m3 gpt-oss-20b-GGUF; do
    [ -d "$MODELS_DIR/$sub" ] || warn "$MODELS_DIR/$sub 없음 (선택적이지만 일부 기능 제한)"
done
ok  "모델 디렉토리 ✓"

# ---------------------------------------------------------------------
# (5) docker compose up
# ---------------------------------------------------------------------
log "[5/6] docker compose up -d"
docker compose up -d
ok  "스택 기동 완료"

# 모델 로드 / DB 초기화 대기
log "    ai-console-app 기동 대기 (uvicorn ready 까지)"
for i in $(seq 1 60); do
    if docker logs ai-console-app 2>&1 | grep -q "Uvicorn running"; then
        ok "    ai-console-app uvicorn ready (${i}초 대기)"
        break
    fi
    sleep 2
done

# ---------------------------------------------------------------------
# (6) init_system (BGE 인덱싱) — 첫 실행 시만
# ---------------------------------------------------------------------
if [ "$SKIP_INIT" = false ]; then
    log "[6/6] init_system (BGE 인덱싱, GPU 사용 — 약 3~5분)"
    docker exec ai-console-app python /app/utils/init_system.py 2>&1 | tail -10
    ok  "init_system 완료"
else
    log "[6/6] --skip-init: init_system 건너뜀"
fi

# ---------------------------------------------------------------------
# 안내
# ---------------------------------------------------------------------
SERVER_PORT=$(grep -E '^SERVER_PORT=' .env | cut -d= -f2 | tr -d '"' || echo 5443)
SERVER_PORT="${SERVER_PORT:-5443}"

# default 구독 api_key 표시 (DB 에서 조회)
API_KEY=$(docker exec ai-console-app python3 -c "
import pymysql, os
c=pymysql.connect(host='mariadb',port=3306,user=os.environ.get('MARIADB_DB_USER','agent'),password=os.environ.get('MARIADB_DB_PASSWORD',''),database=os.environ.get('MARIADB_DB_NAME','agent'))
cur=c.cursor()
cur.execute(\"SELECT api_key FROM ai_subscriptions WHERE name='default'\")
row=cur.fetchone(); print(row[0] if row else '')
" 2>/dev/null || echo "")

echo ""
echo "============================================================"
echo "   ✅ 설치 완료"
echo "============================================================"
echo "   서비스 상태  : docker compose ps"
echo "   로그         : docker compose logs -f app"
echo "   외부 접속    : https://<host>:$SERVER_PORT/docs"
[ -n "$API_KEY" ] && echo "   API key      : $API_KEY"
echo ""
echo "   curl 예시:"
echo "     curl -k -H \"Authorization: Bearer $API_KEY\" \\"
echo "          https://localhost:$SERVER_PORT/api/ai/hello"
echo "============================================================"

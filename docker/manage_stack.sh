#!/bin/bash
# =====================================================================
# manage_stack.sh — docker 스택 운영 명령 단축
#
# INSTALL_GUIDE.md §6 의 docker compose 명령들을 서브커맨드로 묶음.
# 같은 디렉토리의 docker-compose.yml 을 사용.
#
# 사용:
#   ./manage_stack.sh <command> [args...]
#   ./manage_stack.sh --help
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 서비스 이름들 (docker-compose.yml 의 services 와 일치)
SERVICES="app llama-main llama-translation mariadb qdrant"

err()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; }
ok()   { echo -e "\033[1;32m[ ✓ ]\033[0m $*"; }
log()  { echo -e "\033[1;34m[stack]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m  $*" >&2; }

# .env 의 SERVER_PORT 읽기 (health 용)
read_port() {
    local port=8443
    [ -f .env ] && {
        local p
        p=$(grep -E '^SERVER_PORT=' .env | cut -d= -f2 | tr -d '"' || true)
        [ -n "${p:-}" ] && port="$p"
    }
    echo "$port"
}

# ---------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------

cmd_status() {
    docker compose ps
}

cmd_start() {
    log "전체 스택 기동 (docker compose up -d)"
    docker compose up -d
}

cmd_stop() {
    log "스택 중지 (docker compose stop) — 컨테이너/볼륨 유지"
    docker compose stop
}

cmd_restart() {
    if [ $# -ge 1 ]; then
        log "$1 재시작"
        docker compose restart "$1"
    else
        log "전체 재시작"
        docker compose restart
    fi
}

cmd_down() {
    log "스택 down (컨테이너 제거, 볼륨 유지)"
    docker compose down
}

cmd_purge() {
    warn "이 작업은 mariadb / qdrant 볼륨까지 모두 삭제합니다 (DB / 임베딩 인덱싱 데이터 손실)"
    if [ -t 0 ]; then
        read -r -p "정말 진행할까요? (yes 입력 시 진행): " yn
        [ "$yn" = "yes" ] || { echo "취소"; exit 0; }
    fi
    log "docker compose down -v"
    docker compose down -v
}

cmd_logs() {
    local svc="${1:-}"
    shift || true
    if [ -z "$svc" ]; then
        log "전체 서비스 로그 (Ctrl-C 로 중단)"
        docker compose logs -f --tail 100
    else
        log "$svc 로그 (Ctrl-C 로 중단)"
        docker compose logs -f --tail 100 "$svc"
    fi
}

cmd_shell() {
    local svc="${1:-app}"
    log "$svc 컨테이너 shell 진입 (exit 로 빠져나옴)"
    case "$svc" in
        app)             docker compose exec app bash ;;
        llama-main|llama-translation)
                         docker compose exec "$svc" /bin/bash 2>/dev/null \
                            || docker compose exec "$svc" /bin/sh ;;
        mariadb|qdrant)  docker compose exec "$svc" /bin/bash 2>/dev/null \
                            || docker compose exec "$svc" /bin/sh ;;
        *) err "알 수 없는 서비스: $svc (가능: $SERVICES)"; exit 1 ;;
    esac
}

cmd_db() {
    log "mariadb shell (agent 유저, agent DB)"
    # .env 의 비번 읽기
    local pw=""
    [ -f .env ] && {
        local p
        p=$(grep -E '^MARIADB_DB_PASSWORD=' .env | cut -d= -f2 | tr -d '"' || true)
        [ -n "${p:-}" ] && pw="$p"
    }
    docker compose exec mariadb mariadb -uagent -p"$pw" agent
}

cmd_health() {
    local port host
    port=$(read_port)
    host="${AI_CONSOLE_HOST:-localhost}"
    log "GET https://${host}:${port}/api/ai/hello"
    local code body
    body=$(curl -k -s -m 5 -o /dev/stdout -w "\n[HTTP %{http_code}]" \
        "https://${host}:${port}/api/ai/hello" || true)
    echo "$body"
}

cmd_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        err "nvidia-smi 없음 (NVIDIA driver 미설치?)"
        return 1
    fi
    log "GPU 상태"
    nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu \
               --format=csv
    echo
    log "프로세스별 VRAM"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
               --format=csv
}

cmd_init() {
    if [ -t 0 ]; then
        warn "init_system 은 BGE 모델을 GPU 에 다시 로드하고 임베딩을 재생성합니다 (기존 컬렉션 보존)."
        warn "RAM 64GB+ 권장 (4K docs 기준 peak ~33GB)."
        read -r -p "진행? (y/N): " yn
        [[ "$yn" =~ ^[Yy]$ ]] || { echo "취소"; exit 0; }
    fi
    log "init_system 실행"
    docker compose exec app python /app/utils/init_system.py
}

cmd_load_tars() {
    local dir="${1:-images}"
    [ -d "$dir" ] || { err "디렉토리 없음: $dir"; exit 1; }
    log "$dir/*.tar 모두 docker load"
    local found=0
    for tar in "$dir"/*.tar; do
        [ -f "$tar" ] || continue
        found=1
        log "  → $(basename "$tar")"
        docker load -i "$tar" | tail -1
    done
    [ "$found" = 1 ] || warn "tar 파일 없음"
}

cmd_update() {
    cat <<EOF
업데이트 절차 (수동):

  1. 새 패키지를 host 의 임시 위치에 풀기:
        tar xzf ai-console-compose-NEW.tar.gz -C /tmp/

  2. 스택 중지:
        $0 down

  3. 새 이미지 tar 와 스크립트만 교체 (.env / config / DB / Qdrant 데이터는 보존):
        cp /tmp/ai-console-compose-NEW/images/*.tar  ./images/
        cp /tmp/ai-console-compose-NEW/{docker-compose.yml,install_compose.sh,manage_keys.sh,manage_stack.sh,INSTALL_GUIDE.md} ./

  4. 새 이미지 load:
        $0 load-tars images

  5. 재기동:
        $0 start

  6. 확인:
        $0 status
        $0 health
EOF
}

# ---------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------
case "${1:-}" in
    status|ps)        shift || true; cmd_status ;;
    start|up)         shift || true; cmd_start ;;
    stop)             shift || true; cmd_stop ;;
    restart)          shift; cmd_restart "$@" ;;
    down)             shift || true; cmd_down ;;
    purge|down-v)     shift || true; cmd_purge ;;
    logs|log)         shift; cmd_logs "$@" ;;
    shell|exec)       shift; cmd_shell "$@" ;;
    db|mysql|mariadb) shift || true; cmd_db ;;
    health|hello)     shift || true; cmd_health ;;
    gpu|nvidia)       shift || true; cmd_gpu ;;
    init|reindex)     shift || true; cmd_init ;;
    load-tars|load)   shift; cmd_load_tars "$@" ;;
    update|upgrade)   shift || true; cmd_update ;;
    -h|--help|"")
        cat <<EOF
사용법: $0 <command> [args...]

LIFECYCLE:
  status                  docker compose ps
  start                   docker compose up -d
  stop                    docker compose stop (컨테이너/볼륨 유지)
  restart [service]       전체 또는 특정 서비스 재시작
  down                    컨테이너 제거 (볼륨/데이터 유지)
  purge                   ⚠️ 볼륨까지 모두 제거 (DB/Qdrant 데이터 삭제, 확인 프롬프트)

LOGS:
  logs [service]          전체 또는 특정 서비스 follow (Ctrl-C 로 중단)
                          services: $SERVICES

EXEC:
  shell [service]         컨테이너 bash (기본: app)
  db                      mariadb -uagent -p<pw> agent

OPS:
  health                  /api/ai/hello 확인 (HTTP code 표시)
  gpu                     nvidia-smi 요약 (GPU + 프로세스별 VRAM)
  init                    init_system 재실행 (BGE 인덱싱 재생성)

UPDATE:
  load-tars [dir]         dir/*.tar (default: images/) 모두 docker load
  update                  업데이트 절차 안내 (수동 명령 출력)

환경변수:
  AI_CONSOLE_HOST     health 명령에서 사용할 host (기본 localhost)

자세한 운영 가이드: INSTALL_GUIDE.md §6
EOF
        ;;
    *)
        err "알 수 없는 명령: $1"
        echo "  $0 --help 로 사용법 확인"
        exit 1
        ;;
esac

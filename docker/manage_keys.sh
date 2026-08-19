#!/bin/bash
# =====================================================================
# manage_keys.sh — docker 배포 환경용 API 키 관리
#
# 컨테이너 안의 aibotctl 을 docker exec 로 호출하는 얇은 wrapper.
# 풍부한 옵션(--model / --length / --reasoning / --json / -y 등) 은
# aibotctl 이 처리하므로 그대로 pass-through.
#
# 사용:
#   ./manage_keys.sh generate [name] [account] [description] [acl] [...]
#   ./manage_keys.sh list [name] [account] [...]
#   ./manage_keys.sh verify <api_key>
#   ./manage_keys.sh delete <api_key> [-y]
#   ./manage_keys.sh renew  <api_key> [duration] [unit]
#   ./manage_keys.sh set    <api_key> [--model X] [--length Y] [--reasoning Z]
#   ./manage_keys.sh show-default
#
# 추가 옵션은 그대로 aibotctl 로 전달됨:
#   ./manage_keys.sh generate john dev_team --model gpt-5-mini --length high
#   ./manage_keys.sh list --json
#   ./manage_keys.sh set xxxx-... --clear-model
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${AI_CONSOLE_CONTAINER:-ai-console-app}"

err()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; }

check_container() {
    if ! docker exec "$CONTAINER" true 2>/dev/null; then
        err "컨테이너 '$CONTAINER' 가 실행 중이 아닙니다.
  - docker compose ps 로 상태 확인
  - AI_CONSOLE_CONTAINER 환경변수로 컨테이너명 override 가능"
        exit 1
    fi
}

# aibotctl 을 컨테이너 안에서 실행
# 인터랙티브 prompt 도 작동하도록 -t 플래그 (TTY) 부착
ctl() {
    check_container
    if [ -t 0 ] && [ -t 1 ]; then
        docker exec -it -e "ADMIN_KEY=${ADMIN_KEY:-}" "$CONTAINER" /app/aibotctl "$@"
    else
        docker exec -e "ADMIN_KEY=${ADMIN_KEY:-}" "$CONTAINER" /app/aibotctl "$@"
    fi
}

# ---------------------------------------------------------------------
# 명령어 — aibotctl 위임 + 백워드호환 positional 변환
# ---------------------------------------------------------------------

cmd_generate() {
    local FWD=()
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=("$1"); shift; fi   # name
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=("$1"); shift; fi   # account
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
        [ -n "$1" ] && FWD+=(--description "$1"); shift
    fi
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
        [ -n "$1" ] && FWD+=(--acl "$1"); shift
    fi
    FWD+=("$@")
    ctl keys generate "${FWD[@]}"
}

cmd_list() {
    local FWD=()
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
        [ -n "$1" ] && FWD+=(--name "$1"); shift
    fi
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
        [ -n "$1" ] && FWD+=(--account "$1"); shift
    fi
    FWD+=("$@")
    ctl keys list "${FWD[@]}"
}

cmd_verify() { ctl keys show "$@"; }
cmd_delete() { ctl keys delete "$@"; }

cmd_renew() {
    local FWD=()
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=("$1"); shift; fi   # api_key
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=("$1"); shift; fi   # duration
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=(--unit "$1"); shift; fi
    FWD+=("$@")
    ctl keys renew "${FWD[@]}"
}

cmd_set() { ctl keys set "$@"; }

cmd_show_default() {
    # install_compose.sh 가 만든 default 구독 키 표시
    check_container
    docker exec "$CONTAINER" python3 -c "
import pymysql, os
c=pymysql.connect(host='mariadb',port=3306,user=os.environ.get('MARIADB_DB_USER','agent'),password=os.environ.get('MARIADB_DB_PASSWORD',''),database=os.environ.get('MARIADB_DB_NAME','agent'))
cur=c.cursor()
cur.execute(\"SELECT id, name, api_key, expires_at FROM ai_subscriptions WHERE name='default'\")
for r in cur.fetchall(): print(r)
" 2>/dev/null || err "조회 실패 — $CONTAINER 컨테이너 동작 중인지 확인"
}

# ---------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------
case "${1:-}" in
    generate)        shift; cmd_generate "$@" ;;
    list)            shift; cmd_list "$@" ;;
    verify)          shift; cmd_verify "$@" ;;
    delete)          shift; cmd_delete "$@" ;;
    renew)           shift; cmd_renew "$@" ;;
    set)             shift; cmd_set "$@" ;;
    show-default)    shift; cmd_show_default "$@" ;;
    -h|--help|"")
        cat <<EOF
사용법: $0 <command> [args...]

명령어:
  generate [name] [account] [description] [acl] [추가 옵션...]
                                          새 API 키 발급 (누락 필드는 인터랙티브 prompt)
  list [name] [account] [추가 옵션...]    키 목록 (필터)
  verify <api_key>                        키 유효성 / 만료일 확인 (= show)
  delete <api_key> [-y]                   키 삭제 (-y 로 confirm 스킵)
  renew  <api_key> [duration] [unit]      만료일 연장 (unit: day/month/year)
  set    <api_key> [--model X|--clear-model] [--length Y|--clear-length] [--reasoning Z|--clear-reasoning]
                                          model/length/reasoning_effort 갱신
  show-default                            default 구독 키 (mariadb 직조회)

추가 옵션 예시:
  $0 generate john dev_team --model gpt-5-mini --length high --reasoning low
  $0 list --json
  $0 set xxxx-... --clear-model
  $0 generate                             # 전부 인터랙티브 prompt

환경 변수:
  ADMIN_KEY        관리자 키 (운영 admin 검증 켰을 때만 필요)
  AI_CONSOLE_CONTAINER   컨테이너 이름 (기본: ai-console-app)

aibotctl 직접 호출:
  docker exec -it $CONTAINER /app/aibotctl keys --help
EOF
        ;;
    *)
        err "알 수 없는 명령: $1"
        echo "  $0 --help 로 사용법 확인"
        exit 1 ;;
esac

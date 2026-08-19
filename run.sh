#!/bin/bash

# 스크립트 이름: run.sh

PYTHON_SCRIPT="qa_llm.py"
LOG_FILE="qa_llm.log"
API_BASE_URL="https://localhost:5443"

# install.sh가 만든 .venv가 있으면 그 파이썬을 우선 사용 (venv 미활성 셸에서 ./run.sh 직행 지원)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -x "$SCRIPT_DIR/.venv/bin/python3" ] && PATH="$SCRIPT_DIR/.venv/bin:$PATH"

# admin key 확보 — install.sh 가 발급한 api_keys/admin.key 를 읽는다.
# 예전 '0'32자 더미 기본값은 제거했다: 서버 검증을 되살린 이상(security-review.md S-2/S-7)
# 기본값이 남아 있으면 검증이 형식뿐이 된다. 키가 없으면 조용히 넘어가지 말고 실패시킨다.
get_admin_key() {
    if [ -n "$ADMIN_KEY" ]; then return 0; fi
    if [ -f "$SCRIPT_DIR/api_keys/admin.key" ]; then
        ADMIN_KEY="$(head -n1 "$SCRIPT_DIR/api_keys/admin.key" | tr -d '\r\n')"
    fi
    if [ -z "$ADMIN_KEY" ]; then
        echo "❌ 관리자 키가 없습니다 — api_keys/admin.key 부재." >&2
        echo "   ./install.sh --config-only 로 재발급하거나 ADMIN_KEY 환경변수를 주세요." >&2
        return 1
    fi
}

start() {
    echo "Starting $PYTHON_SCRIPT..."

    # 필요한 로그 디렉토리 생성
    mkdir -p logs/subscription

    # 날짜별 로그 파일 이름 생성
    DATE_STR=$(date +"%Y%m%d")
    DATED_LOG_FILE="qa_llm_${DATE_STR}.log"

    # append 모드로 날짜별 로그 파일에 기록
    nohup python3 "$PYTHON_SCRIPT" >> "$DATED_LOG_FILE" 2>&1 &
    echo "Started with PID $!"
    echo "Log file: $DATED_LOG_FILE"
}

_cleanup_orphan_llama_servers() {
    # qa_llm 이 죽으면 자식 llama-server 도 atexit 으로 정리되지만, 그게 안 됐을 때 orphan 정리.
    # 바이너리 경로로 매칭 — 모델 경로 기준(구 /data/models)은 이 포크의 models/ 경로와 안 맞아 사문이었음
    LLAMA_PIDS=$(pgrep -f "build/bin/llama-server -m ")
    if [ -n "$LLAMA_PIDS" ]; then
        echo "🦙 Orphan llama-server cleanup: $LLAMA_PIDS"
        kill $LLAMA_PIDS 2>/dev/null
        sleep 2
        STILL=$(pgrep -f "build/bin/llama-server -m ")
        if [ -n "$STILL" ]; then
            echo "🦙 Force kill llama-server: $STILL"
            kill -9 $STILL 2>/dev/null
        fi
    fi
}

stop() {
    echo "Stopping $PYTHON_SCRIPT..."
    # python 프로세스만 찾기 (VSCode ripgrep 등 제외)
    PIDS=$(pgrep -f "python.*$PYTHON_SCRIPT")
    if [ -z "$PIDS" ]; then
        echo "No process found running $PYTHON_SCRIPT."
        _cleanup_orphan_llama_servers
        return
    fi

    echo "SIGTERM → $PIDS"
    kill $PIDS 2>/dev/null

    # Graceful shutdown 대기 (최대 15초)
    for i in $(seq 1 15); do
        sleep 1
        REMAINING=$(pgrep -f "python.*$PYTHON_SCRIPT")
        if [ -z "$REMAINING" ]; then
            echo "✅ Stopped gracefully in ${i}s"
            _cleanup_orphan_llama_servers
            return
        fi
    done

    # 15초 후에도 살아있으면 SIGKILL
    echo "⚠️ SIGTERM 무시 — SIGKILL 진행 (PIDs: $REMAINING)"
    kill -9 $REMAINING 2>/dev/null
    sleep 1
    _cleanup_orphan_llama_servers
    echo "✅ Force stopped"
}

status() {
    # python 프로세스만 찾기 (VSCode ripgrep 등 제외)
    PIDS=$(pgrep -f "python.*$PYTHON_SCRIPT")
    if [ -z "$PIDS" ]; then
        echo "$PYTHON_SCRIPT is not running."
    else
        echo "$PYTHON_SCRIPT is running:"
        echo "----------------------------------------"
        # ps 명령으로 상세 정보 표시 (USER, PID, START TIME, COMMAND)
        ps -f -p $(echo $PIDS | tr ' ' ',') | tail -n +2 | while read line; do
            USER=$(echo $line | awk '{print $1}')
            PID=$(echo $line | awk '{print $2}')
            STIME=$(echo $line | awk '{print $5}')
            echo "  PID: $PID | User: $USER | Started: $STIME"
        done
        echo "----------------------------------------"
    fi
}

restart() {
    echo "Restarting $PYTHON_SCRIPT..."
    stop
    sleep 2
    start
}

init() {
    echo "Initializing system..."
    python3 utils/init_system.py
    if [ $? -eq 0 ]; then
        echo "System initialization completed successfully."
    else
        echo "System initialization failed."
        exit 1
    fi
}

# 서버 상태 확인 함수
check_server() {
    curl -k -s "$API_BASE_URL/" > /dev/null 2>&1
    return $?
}

# API 키 생성 — aibotctl 위임
# 백워드호환: $1=name $2=account $3=description $4=acl
# 그 외 옵션(--model 등)은 그대로 pass-through. 인자 없으면 인터랙티브 prompt.
generate_key() {
    local DIR="$(dirname "$0")"
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
    exec "$DIR/aibotctl" keys generate "${FWD[@]}"
}

# API 키 목록 조회 — aibotctl 위임
# 백워드호환: $1=name $2=account
list_keys() {
    local DIR="$(dirname "$0")"
    local FWD=()
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
        [ -n "$1" ] && FWD+=(--name "$1"); shift
    fi
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
        [ -n "$1" ] && FWD+=(--account "$1"); shift
    fi
    FWD+=("$@")
    exec "$DIR/aibotctl" keys list "${FWD[@]}"
}

# 임베딩 상태 확인
check_embedding() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 check-embedding <sub_id>"
        echo "       $0 check-embedding all"
        exit 1
    fi

    if ! check_server; then
        echo "❌ Server is not running. Please start the server first with: $0 start"
        exit 1
    fi

    SUB_ID="$1"

    if [ "$SUB_ID" = "all" ]; then
        # config.ini에서 admin key 읽기 (admin 전용 기능)
        get_admin_key
        echo "🔍 Checking all embedding jobs..."
        RESPONSE=$(curl -k -s -X POST "$API_BASE_URL/api/embedding-status-all" \
            -H "Content-Type: application/json" \
            -d "{
                \"admin_key\": \"$ADMIN_KEY\"
            }" 2>/dev/null)
    else
        echo "🔍 Checking embedding status for subscription ID: $SUB_ID"
        RESPONSE=$(curl -k -s "$API_BASE_URL/api/embedding-status/$SUB_ID" 2>/dev/null)
    fi

    if [ $? -eq 0 ]; then
        echo "$RESPONSE" | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read()), ensure_ascii=False, indent=2))" 2>/dev/null || echo "$RESPONSE"
    else
        echo "❌ Failed to check embedding status:"
        echo "$RESPONSE" | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read()), ensure_ascii=False, indent=2))" 2>/dev/null || echo "$RESPONSE"
        exit 1
    fi
}

# 단일 파일 임베딩 업데이트
update_embedding() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 update-embedding <file_path> [sub_id] [gpt]"
        echo ""
        echo "Parameters:"
        echo "  file_path  : Relative path from docs/aibot/yaml (e.g., action/ticket-open-list.yaml)"
        echo "  sub_id     : Subscription ID (default: 1)"
        echo "  gpt        : Use 'gpt' to use OpenAI API instead of local model"
        echo ""
        echo "Examples:"
        echo "  $0 update-embedding action/ticket-open-list.yaml"
        echo "  $0 update-embedding qna/test.yaml 2"
        echo "  $0 update-embedding action/test.yaml 1 gpt"
        exit 1
    fi

    # 파라미터 파싱
    FILE_PATH="$1"
    SUB_ID="${2:-1}"  # 기본값 1
    USE_GPT="${3:-}"  # 기본값 빈 문자열

    echo "🔄 단일 파일 임베딩 업데이트"
    echo "================================================"
    echo "📁 파일: $FILE_PATH"
    echo "🆔 구독 ID: $SUB_ID"
    echo "🤖 모델: $([ "$USE_GPT" = "gpt" ] && echo "OpenAI GPT API" || echo "로컬 모델")"
    echo "================================================"

    # 파일 존재 확인
    FULL_FILE_PATH="docs/aibot/yaml/$FILE_PATH"
    if [ ! -f "$FULL_FILE_PATH" ]; then
        echo "❌ 파일을 찾을 수 없습니다: $FULL_FILE_PATH"
        exit 1
    fi

    echo "✅ 파일 확인: $FULL_FILE_PATH ($(stat -c%s "$FULL_FILE_PATH") bytes)"

    # Python 스크립트 실행
    echo ""
    echo "🚀 임베딩 업데이트 시작..."

    if [ "$USE_GPT" = "gpt" ]; then
        python3 utils/update_single_embedding_db.py "$FILE_PATH" "$USE_GPT"
    else
        python3 utils/update_single_embedding_db.py "$FILE_PATH"
    fi

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 임베딩 업데이트 완료!"
        echo "🔍 확인: ./run.sh check-embedding-file $FILE_PATH"
    else
        echo ""
        echo "❌ 임베딩 업데이트 실패"
        exit 1
    fi
}

# 단일 파일 임베딩 확인
check_embedding_file() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 check-embedding-file <file_path> [sub_id]"
        echo ""
        echo "Parameters:"
        echo "  file_path  : Relative path from docs/aibot/yaml"
        echo "  sub_id     : Subscription ID (default: 1)"
        echo ""
        echo "Example:"
        echo "  $0 check-embedding-file action/ticket-open-list.yaml"
        exit 1
    fi

    FILE_PATH="$1"
    SUB_ID="${2:-1}"

    echo "🔍 파일 임베딩 확인: $FILE_PATH (구독 ID: $SUB_ID)"
    echo ""

    python3 utils/check_embedding_db.py "$FILE_PATH"
}

# API 키 검증 — aibotctl 위임 (인자 없으면 인터랙티브 prompt)
verify_key() {
    exec "$(dirname "$0")/aibotctl" keys show "$@"
}

# API 키 삭제 — aibotctl 위임 (CLI 가 confirm 처리)
delete_key() {
    exec "$(dirname "$0")/aibotctl" keys delete "$@"
}

# API 키 갱신 — aibotctl 위임
# 백워드호환: $1=api_key $2=duration $3=unit (day|month|year)
renew_key() {
    local DIR="$(dirname "$0")"
    local FWD=()
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=("$1"); shift; fi   # api_key
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=("$1"); shift; fi   # duration
    if [ $# -ge 1 ] && [[ "$1" != -* ]]; then FWD+=(--unit "$1"); shift; fi
    FWD+=("$@")
    exec "$DIR/aibotctl" keys renew "${FWD[@]}"
}

# 구독 설정 갱신 — aibotctl 위임 (NULL → --clear-* 매핑)
set_config() {
    if [ $# -lt 1 ]; then
        cat <<EOF
Usage: $0 set-config <api_key> [--model X|NULL] [--length low|medium|high|NULL] [--reasoning minimal|low|medium|high|NULL]
  Tip: ./aibotctl keys set --help 가 더 풍부한 인터페이스 제공
EOF
        exit 1
    fi
    local KEY="$1"; shift
    local ARGS=("$KEY")
    while [ $# -gt 0 ]; do
        case "$1" in
            --model)
                if [ "$2" = "NULL" ]; then ARGS+=(--clear-model); else ARGS+=(--model "$2"); fi; shift 2 ;;
            --length)
                if [ "$2" = "NULL" ]; then ARGS+=(--clear-length); else ARGS+=(--length "$2"); fi; shift 2 ;;
            --reasoning)
                if [ "$2" = "NULL" ]; then ARGS+=(--clear-reasoning); else ARGS+=(--reasoning "$2"); fi; shift 2 ;;
            *) echo "❌ Unknown option: $1"; exit 1 ;;
        esac
    done
    exec "$(dirname "$0")/aibotctl" keys set "${ARGS[@]}"
}

# 임베딩 재생성
clean() {
    echo "🧹 Cleaning full_messages logs..."

    # logs/full_messages 폴더가 존재하는지 확인
    if [ -d "logs/full_messages" ]; then
        # 폴더 안의 모든 내용 삭제 (폴더는 유지)
        rm -rf logs/full_messages/*
        echo "✅ Cleaned all files in logs/full_messages/"
    else
        echo "❌ Directory logs/full_messages/ does not exist"
    fi
}

regenerate_base_embedding() {
    # 확인 생략 옵션 확인
    SKIP_CONFIRM=false

    # 인자 처리
    while [[ $# -gt 0 ]]; do
        case $1 in
            --yes|-y)
                SKIP_CONFIRM=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 regenerate-base-embedding [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --yes, -y    Skip confirmation prompt"
                echo "  --help, -h   Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 regenerate-base-embedding"
                echo "  $0 regenerate-base-embedding --yes"
                echo "  $0 regenerate-base-embedding -y"
                return 0
                ;;
            *)
                echo "❌ Unknown option: $1"
                echo "Use '$0 regenerate-base-embedding --help' for usage information"
                exit 1
                ;;
        esac
    done

    echo "🔄 Regenerating base embedding system..."

    # 1. 서버 상태 확인 - 꺼져있어야 함
    if check_server; then
        echo "❌ Server must be stopped before regenerating embeddings (GPU intensive operation)"
        echo "Please stop the server first: $0 stop"
        exit 1
    fi

    # 2. 확인 프롬프트 (SKIP_CONFIRM이 false일 때만)
    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "⚠️  This will regenerate the entire base embedding system. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Cancelled."
            exit 0
        fi
    fi

    echo "🔧 Starting embedding regeneration process..."

    # 3. Python 스크립트로 전체 프로세스 실행
    python3 -c "
import sys, os
sys.path.append('.')
from config_utils import ConfigManager
from aibot_db_manager import AibotDBManager
from aibot_db_command import SQL_QUERIES
from aibot_embedding import EmbeddingGenerator
import asyncio

async def regenerate_all_embeddings():
    try:
        print('📋 Loading configuration...')
        config = ConfigManager()

        # DB 매니저 생성 (서버 없이 직접 연결)
        db_manager = AibotDBManager(
            config=config,
            query_properties=SQL_QUERIES
        )

        # 기본 구독 ID 가져오기 (name='default')
        print('🔍 Finding default subscription...')
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id FROM ai_subscriptions WHERE name = %s', ('default',))
                result = cursor.fetchone()
                if not result:
                    print('❌ Default subscription not found')
                    return False
                default_sub_id = result['id'] if isinstance(result, dict) else result[0]
                print(f'📋 Default subscription ID: {default_sub_id}')

        # 모델 타입에 따른 use_local_model 설정
        model_config = config.get_model_config()
        model = model_config.get('model', 'gpt-oss')
        use_local_model = model != 'gpt'
        use_bge = config.config.get('embedding', 'use_bge_mode', fallback='False')

        # BGE 모드이면 로컬 모델(BGE-M3) 사용 강제
        if use_bge.lower() == 'true':
            use_local_model = True

        print(f'🤖 Model: {model}, Local model: {use_local_model}')
        print(f'🧠 BGE M3 embedding mode: {use_bge}')

        # BGE 모드에서 기존 Qdrant 컬렉션 먼저 정리
        if use_bge.lower() == 'true':
            print('🧹 BGE 모드 - 기존 Qdrant 컬렉션 정리 중...')
            try:
                import requests
                # 포트를 하드코딩하면 인스턴스 B에서 재색인할 때 기본 인스턴스(6333)의
                # 컬렉션을 지운다 — 한 호스트에 콘솔 N대를 띄우는 배치에서 치명적. config가 단일 소스.
                _qh = config.config.get('qdrant', 'host', fallback='localhost')
                _qp = config.config.get('qdrant', 'port', fallback='6333')
                qdrant_url = f'http://{_qh}:{_qp}'

                # 컬렉션명도 config가 단일 소스 — 하드코딩 'bge'면 커스텀 컬렉션 인스턴스가
                # 엉뚱한 'bge'를 지운다(qdrant_collection()과 동일 규칙: [qdrant] collection, 기본 bge)
                _qc = config.config.get('qdrant', 'collection', fallback='bge')
                collections_to_delete = [_qc]

                for collection_name in collections_to_delete:
                    try:
                        response = requests.delete(f'{qdrant_url}/collections/{collection_name}')
                        if response.status_code == 200:
                            print(f'🗑️  기존 컬렉션 삭제: {collection_name}')
                        else:
                            print(f'ℹ️  컬렉션 없음: {collection_name}')
                    except Exception as e:
                        print(f'ℹ️  컬렉션 삭제 건너뜀 ({collection_name}): {e}')

                print('✅ Qdrant 컬렉션 정리 완료')
            except Exception as e:
                print(f'⚠️  Qdrant 컬렉션 정리 실패: {e}')

        # 임베딩 제너레이터 생성
        print('🔧 Initializing embedding generator...')
        local_model_path = None
        if use_local_model:
            # config.ini에서 로컬 모델 경로 가져오기
            paths_config = config.get_paths_config()
            local_model_path = '../models/bge-m3'

        generator = EmbeddingGenerator(
            use_local_model=use_local_model,
            local_model_path=local_model_path,
            save_db=True
        )

        # Step 0: 기본 구독의 기존 임베딩 데이터 삭제
        print('🗑️  Deleting existing embeddings for default subscription...')
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ai_faiss_index_parts에서 모든 데이터 삭제 (기본 임베딩 변경으로 모든 인덱스 무효화)
                    cursor.execute('DELETE FROM ai_faiss_index_parts')
                    deleted_faiss_parts = cursor.rowcount
                    print(f'   Deleted {deleted_faiss_parts} FAISS index parts (all users)')

                    # ai_faiss_indices에서 모든 FAISS 인덱스 삭제
                    cursor.execute('DELETE FROM ai_faiss_indices')
                    deleted_faiss = cursor.rowcount
                    print(f'   Deleted {deleted_faiss} FAISS indices (all users)')

                    # ai_knowledge_graph_parts에서 모든 데이터 삭제 (기본 임베딩 변경으로 모든 그래프 무효화)
                    cursor.execute('DELETE FROM ai_knowledge_graph_parts')
                    deleted_kg_parts = cursor.rowcount
                    print(f'   Deleted {deleted_kg_parts} knowledge graph parts (all users)')

                    # ai_knowledge_graph에서 모든 지식그래프 삭제
                    cursor.execute('DELETE FROM ai_knowledge_graph')
                    deleted_kg = cursor.rowcount
                    print(f'   Deleted {deleted_kg} knowledge graph entries (all users)')

                    # openai_prompts에서 기본 구독의 임베딩만 삭제 (사용자 데이터 보존)
                    cursor.execute('DELETE FROM openai_prompts WHERE subscription_id = %s', (default_sub_id,))
                    deleted_prompts = cursor.rowcount
                    print(f'   Deleted {deleted_prompts} prompts from openai_prompts (default only)')

                conn.commit()
                print('✅ Existing embeddings deleted successfully')
        except Exception as e:
            print(f'❌ Failed to delete existing embeddings: {e}')
            return False

        # Step 1: 기본 파일 기반 임베딩 생성
        print('📄 Generating file-based embeddings for default subscription...')
        try:
            generator.set_ultra_performance_profile()
            generator.generate_embeddings_ultra_fast(default_sub_id)
            print('✅ File-based embeddings generated successfully')
        except Exception as e:
            print(f'❌ Failed to generate file-based embeddings: {e}')
            return False

        if use_bge.lower() == 'true':
            print('🧠 BGE M3 임베딩 모드 감지 - Step 2 건너뜀')
            print('✅ Base embedding system regeneration completed successfully (BGE mode)')
        else:    
            # Step 2: 모든 사용자 구독의 FAISS와 지식그래프 생성 (기본 구독 제외)
            print('👥 Regenerating FAISS indexes and knowledge graphs for all users (excluding default)...')

            # 모든 구독 목록 가져오기 (기본 구독 제외)
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('SELECT id FROM ai_subscriptions WHERE id != %s', (default_sub_id,))
                    results = cursor.fetchall()
                    user_subs = [row['id'] if isinstance(row, dict) else row[0] for row in results]

            if not user_subs:
                print('ℹ️  No user subscriptions found, skipping Step 2')
            else:
                print(f'👥 Found {len(user_subs)} user subscriptions to process')

                for sub_id in user_subs:
                    print(f'🔨 Processing user subscription {sub_id}...')
                    try:
                        # 변경사항만 임베딩 (실제로는 기존 데이터 기반으로 처리)
                        ctx = generator.embed_changes_only(sub_id)

                        # BGE 모드 확인 (generator의 model_type 체크)
                        is_bge_mode = hasattr(generator, 'model_type') and generator.model_type == 'bge_m3'

                        if is_bge_mode:
                            print(f'🔍 구독 {sub_id}: BGE 모드 감지 - postprocess_with_faiss_and_kg 건너뛰기')
                            print(f'✅ User subscription {sub_id} processed successfully (BGE mode)')
                        else:
                            # BGE 모드가 아닌 경우에만 FAISS 인덱스 및 지식그래프 생성
                            generator.postprocess_with_faiss_and_kg(sub_id, ctx, initial_run=True)
                            print(f'✅ User subscription {sub_id} processed successfully (with FAISS/KG)')

                    except Exception as e:
                        print(f'❌ Failed to process user subscription {sub_id}: {e}')
                        continue

            print('✅ Base embedding system regeneration completed successfully')
        return True

    except Exception as e:
        print(f'❌ Error during regeneration: {e}')
        return False

# 실행
success = asyncio.run(regenerate_all_embeddings())
sys.exit(0 if success else 1)
"

    if [ $? -eq 0 ]; then
        echo "✅ Base embedding regeneration completed successfully"
        echo "ℹ️  You can now start the server: $0 start"
    else
        echo "❌ Base embedding regeneration failed"
        exit 1
    fi
}

# 임베딩 재생성 (복사 방식)
regenerate_base_embedding_with_copy() {
    # 확인 생략 옵션 확인
    SKIP_CONFIRM=false

    # 인자 처리
    while [[ $# -gt 0 ]]; do
        case $1 in
            --yes|-y)
                SKIP_CONFIRM=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 regenerate-base-embedding-with-copy [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --yes, -y    Skip confirmation prompt"
                echo "  --help, -h   Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 regenerate-base-embedding-with-copy"
                echo "  $0 regenerate-base-embedding-with-copy --yes"
                echo "  $0 regenerate-base-embedding-with-copy -y"
                echo ""
                echo "Description:"
                echo "  Regenerates base embeddings and copies FAISS/KG data to all user subscriptions"
                echo "  instead of generating them individually (faster and more consistent)"
                return 0
                ;;
            *)
                echo "❌ Unknown option: $1"
                echo "Use '$0 regenerate-base-embedding-with-copy --help' for usage information"
                exit 1
                ;;
        esac
    done

    echo "🔄 Regenerating base embedding system with copy mode..."

    # 1. 서버 상태 확인 - 꺼져있어야 함
    if check_server; then
        echo "❌ Server must be stopped before regenerating embeddings (GPU intensive operation)"
        echo "Please stop the server first: $0 stop"
        exit 1
    fi

    # 2. 확인 프롬프트 (SKIP_CONFIRM이 false일 때만)
    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "⚠️  This will regenerate base embeddings and copy to all users. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Cancelled."
            exit 0
        fi
    fi

    echo "🔧 Starting embedding regeneration with copy process..."

    # 3. Python 스크립트로 전체 프로세스 실행
    python3 -c "
import sys, os
sys.path.append('.')
from config_utils import ConfigManager
from aibot_db_manager import AibotDBManager
from aibot_db_command import SQL_QUERIES
from aibot_embedding import EmbeddingGenerator
import asyncio

async def regenerate_with_copy():
    try:
        print('📋 Loading configuration...')
        config = ConfigManager()

        # DB 매니저 생성 (서버 없이 직접 연결)
        db_manager = AibotDBManager(
            config=config,
            query_properties=SQL_QUERIES
        )

        # 기본 구독 ID 가져오기 (name='default')
        print('🔍 Finding default subscription...')
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id FROM ai_subscriptions WHERE name = %s', ('default',))
                result = cursor.fetchone()
                if not result:
                    print('❌ Default subscription not found')
                    return False
                default_sub_id = result['id'] if isinstance(result, dict) else result[0]
                print(f'📋 Default subscription ID: {default_sub_id}')

        # 모델 타입에 따른 use_local_model 설정
        model_config = config.get_model_config()
        model = model_config.get('model', 'gpt-oss')
        use_local_model = model != 'gpt'
        use_bge = config.config.get('embedding', 'use_bge_mode', fallback='False')
        print(f'🤖 Model: {model}, Local model: {use_local_model}')
        print(f'🧩 BGE mode: {use_bge}')

        # 임베딩 제너레이터 생성
        print('🔧 Initializing embedding generator...')
        local_model_path = None
        if use_local_model:
            # config.ini에서 로컬 모델 경로 가져오기
            paths_config = config.get_paths_config()
            local_model_path = '../models/bge-m3'

        generator = EmbeddingGenerator(
            use_local_model=use_local_model,
            local_model_path=local_model_path,
            save_db=True
        )

        # Step 0: 기본 구독의 기존 임베딩 데이터 삭제
        print('🗑️  Deleting existing embeddings for all subscriptions...')
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ai_faiss_index_parts에서 모든 데이터 삭제
                    cursor.execute('DELETE FROM ai_faiss_index_parts')
                    deleted_faiss_parts = cursor.rowcount
                    print(f'   Deleted {deleted_faiss_parts} FAISS index parts (all users)')

                    # ai_faiss_indices에서 모든 FAISS 인덱스 삭제
                    cursor.execute('DELETE FROM ai_faiss_indices')
                    deleted_faiss = cursor.rowcount
                    print(f'   Deleted {deleted_faiss} FAISS indices (all users)')

                    # ai_knowledge_graph_parts에서 모든 데이터 삭제
                    cursor.execute('DELETE FROM ai_knowledge_graph_parts')
                    deleted_kg_parts = cursor.rowcount
                    print(f'   Deleted {deleted_kg_parts} knowledge graph parts (all users)')

                    # ai_knowledge_graph에서 모든 지식그래프 삭제
                    cursor.execute('DELETE FROM ai_knowledge_graph')
                    deleted_kg = cursor.rowcount
                    print(f'   Deleted {deleted_kg} knowledge graph entries (all users)')

                    # openai_prompts에서 기본 구독의 임베딩만 삭제
                    cursor.execute('DELETE FROM openai_prompts WHERE subscription_id = %s', (default_sub_id,))
                    deleted_prompts = cursor.rowcount
                    print(f'   Deleted {deleted_prompts} prompts from openai_prompts (default only)')

                conn.commit()
                print('✅ Existing embeddings deleted successfully')
        except Exception as e:
            print(f'❌ Failed to delete existing embeddings: {e}')
            return False

        # Step 1: 기본 파일 기반 임베딩 생성 및 FAISS/KG 생성
        print('📄 Generating embeddings, FAISS and Knowledge Graph for default subscription...')
        try:
            generator.set_ultra_performance_profile()
            generator.generate_embeddings_ultra_fast(default_sub_id)

            print('✅ Default subscription embeddings, FAISS and KG generated successfully')
        except Exception as e:
            print(f'❌ Failed to generate default embeddings: {e}')
            return False

        # Step 2: 모든 사용자 구독에 FAISS와 지식그래프 복사
        print('👥 Copying FAISS indexes and knowledge graphs to all user subscriptions...')

        if use_bge.lower() == 'true':
            print('ℹ️  BGE mode enabled - skipping Step 2 copy process')
        else:
            # 모든 구독 목록 가져오기 (기본 구독 제외)
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('SELECT id FROM ai_subscriptions WHERE id != %s', (default_sub_id,))
                    results = cursor.fetchall()
                    user_subs = [row['id'] if isinstance(row, dict) else row[0] for row in results]

            if not user_subs:
                print('ℹ️  No user subscriptions found, skipping Step 2')
            else:
                print(f'👥 Found {len(user_subs)} user subscriptions to copy to')

                for sub_id in user_subs:
                    print(f'📋 Copying data to subscription {sub_id}...')
                    try:
                        with db_manager.get_connection() as conn:
                            with conn.cursor() as cursor:
                                # 1. FAISS 인덱스 복사
                                cursor.execute('''
                                    INSERT INTO ai_faiss_indices (subscription_id, index_name, index_data, metadata, artifact_metadata, created_at, updated_at)
                                    SELECT %s, index_name, index_data, metadata, artifact_metadata, NOW(), NOW()
                                    FROM ai_faiss_indices
                                    WHERE subscription_id = %s
                                ''', (sub_id, default_sub_id))

                                # 2. FAISS 인덱스 파트 복사
                                cursor.execute('''
                                    INSERT INTO ai_faiss_index_parts (subscription_id, index_name, part_no, chunk, created_at)
                                    SELECT %s, index_name, part_no, chunk, NOW()
                                    FROM ai_faiss_index_parts
                                    WHERE subscription_id = %s
                                ''', (sub_id, default_sub_id))

                                # 3. 지식그래프 복사 및 ID 매핑
                                cursor.execute('''
                                    SELECT id, data_name, metadata
                                    FROM ai_knowledge_graph
                                    WHERE subscription_id = %s
                                ''', (default_sub_id,))
                                default_kg_entries = cursor.fetchall()

                                for kg_entry in default_kg_entries:
                                    old_kg_id = kg_entry['id']

                                    # 새로운 지식그래프 엔트리 생성
                                    cursor.execute('''
                                        INSERT INTO ai_knowledge_graph (subscription_id, data_name, metadata, created_at, updated_at)
                                        VALUES (%s, %s, %s, NOW(), NOW())
                                    ''', (sub_id, kg_entry['data_name'], kg_entry['metadata']))

                                    new_kg_id = cursor.lastrowid

                                    # 4. 해당 지식그래프 파트 복사
                                    cursor.execute('''
                                        INSERT INTO ai_knowledge_graph_parts (kg_id, part_no, codec, chunk, checksum, created_at)
                                        SELECT %s, part_no, codec, chunk, checksum, NOW()
                                        FROM ai_knowledge_graph_parts
                                        WHERE kg_id = %s
                                    ''', (new_kg_id, old_kg_id))

                            conn.commit()
                            print(f'✅ Successfully copied data to subscription {sub_id}')

                    except Exception as e:
                        print(f'❌ Failed to copy data to subscription {sub_id}: {e}')
                        continue

        print('✅ Base embedding system regeneration with copy completed successfully')
        return True

    except Exception as e:
        print(f'❌ Error during regeneration: {e}')
        return False

# 실행
success = asyncio.run(regenerate_with_copy())
sys.exit(0 if success else 1)
"

    if [ $? -eq 0 ]; then
        echo "✅ Base embedding regeneration with copy completed successfully"
        echo "ℹ️  You can now start the server: $0 start"
    else
        echo "❌ Base embedding regeneration with copy failed"
        exit 1
    fi
}

# 데이터베이스 설정
configure_db() {
    echo "🔧 Database Configuration Setup"
    echo "================================="
    echo ""

    # DB 설정 복호화 함수
    get_db_config() {
        if [ -f "config.ini" ] && command -v python3 > /dev/null 2>&1; then
            python3 -c "
from config_utils import ConfigManager
try:
    config = ConfigManager()
    db_config = config.get_db_config()
    print(f\"URL={db_config.get('url', 'localhost')}\")
    print(f\"PORT={db_config.get('port', 3306)}\")
    print(f\"USER={db_config.get('user', '')}\")
    print(f\"PASSWORD={db_config.get('password', '')}\")
    print(f\"DATABASE={db_config.get('database', '')}\")
except Exception as e:
    print(f\"ERROR=Failed to decrypt DB config: {e}\")
"
        else
            echo "ERROR=config.ini not found or python3 not available"
        fi
    }

    # 현재 설정 읽기 (복호화된 값)
    if [ -f "config.ini" ]; then
        DB_CONFIG=$(get_db_config)

        if echo "$DB_CONFIG" | grep -q "ERROR="; then
            echo "⚠️  Database configuration error:"
            echo "$DB_CONFIG" | grep "ERROR=" | cut -d'=' -f2-
            CURRENT_HOST="localhost"
            CURRENT_PORT="3306"
            CURRENT_DB="agent"
        else
            CURRENT_HOST=$(echo "$DB_CONFIG" | grep "^URL=" | cut -d'=' -f2)
            CURRENT_PORT=$(echo "$DB_CONFIG" | grep "^PORT=" | cut -d'=' -f2)
            CURRENT_DB=$(echo "$DB_CONFIG" | grep "^DATABASE=" | cut -d'=' -f2)
        fi
    fi
}

show_help() {
    cat << 'EOF'
🚀 AI QA System Management Script

BASIC COMMANDS (No server required):
  start                    Start the QA server
  stop                     Stop the QA server
  status                   Check server status
  restart                  Restart the QA server
  init                     Initialize system (DB, default subscription, embeddings)

API MANAGEMENT COMMANDS (Requires server to be running):
  💡 Tip: 풍부한 옵션/자동완성/JSON 출력을 원하면 ./aibotctl keys --help
     자동완성 설치 (bash):  eval "$(_AIBOTCTL_COMPLETE=bash_source ./aibotctl)"

  generate-key <name> <account> [description] [acl]
                          Generate new API key
                          Example: ./run.sh generate-key john_user dev_team "Dev user" "192.168.1.0/24"

  list-keys [name] [account]
                          List API keys (all or by name/account)
                          Example: ./run.sh list-keys
                          Example: ./run.sh list-keys john_user
                          Example: ./run.sh list-keys "" dev_team

  verify-key <api_key>    Verify API key validity
                          Example: ./run.sh verify-key 12345678-1234-1234-1234-123456789abc

  delete-key <api_key>    Delete API key (with confirmation)
                          Example: ./run.sh delete-key 12345678-1234-1234-1234-123456789abc

  renew-key <api_key> [duration] [unit]
                          Extend API key expiration. duration 1~100, unit day|month|year (default 1 year)
                          Example: ./run.sh renew-key 12345678-1234-1234-1234-123456789abc
                          Example: ./run.sh renew-key 12345678-... 100         # +100 years
                          Example: ./run.sh renew-key 12345678-... 30 day      # +30 days

  set-config <api_key> [--model X|NULL] [--length low|medium|high|NULL] [--reasoning minimal|low|medium|high|NULL]
                          Update per-subscription model/length/reasoning_effort.
                          Only specified fields are updated. NULL clears the field (fallback mode).
                          Example: ./run.sh set-config 12345678-... --model gpt-5-mini --length high
                          Example: ./run.sh set-config 12345678-... --reasoning low
                          Example: ./run.sh set-config 12345678-... --model NULL        # back to config default

  check-embedding <sub_id>|all
                          Check embedding generation status
                          Example: ./run.sh check-embedding 123
                          Example: ./run.sh check-embedding all

EMBEDDING MANAGEMENT COMMANDS (Requires server to be stopped):
  regenerate-base-embedding [OPTIONS]
                          Regenerate all base embeddings and indexes
                          Note: This is a GPU-intensive operation

                          Options:
                            --yes, -y     Skip confirmation prompt
                            --help, -h    Show help for this command

                          Examples:
                            ./run.sh regenerate-base-embedding
                            ./run.sh regenerate-base-embedding --yes
                            ./run.sh regenerate-base-embedding -y

  regenerate-base-embedding-with-copy [OPTIONS]
                          Regenerate base embeddings and copy FAISS/KG to all users
                          Note: Faster than regenerate-base-embedding (copies instead of regenerating)

                          Options:
                            --yes, -y     Skip confirmation prompt
                            --help, -h    Show help for this command

                          Examples:
                            ./run.sh regenerate-base-embedding-with-copy
                            ./run.sh regenerate-base-embedding-with-copy --yes
                            ./run.sh regenerate-base-embedding-with-copy -y

  update-embedding <file_path> [sub_id] [gpt]
                          Update embedding for a single file
                          file_path: Relative path from docs/aibot/yaml (e.g., action/ticket-open-list.yaml)
                          sub_id: Subscription ID (default: 1)
                          gpt: Use 'gpt' to use OpenAI API instead of local model

                          Examples:
                            ./run.sh update-embedding action/ticket-open-list.yaml
                            ./run.sh update-embedding qna/test.yaml 2
                            ./run.sh update-embedding action/test.yaml 1 gpt

  check-embedding-file <file_path> [sub_id]
                          Check embedding information for a single file
                          file_path: Relative path from docs/aibot/yaml
                          sub_id: Subscription ID (default: 1)

                          Examples:
                            ./run.sh check-embedding-file action/ticket-open-list.yaml
                            ./run.sh check-embedding-file qna/test.yaml 2

  clean                   Clean all files in logs/full_messages/ directory
                          Example: ./run.sh clean

  test                    Test ai-console completions2 API endpoint
                          Example: ./run.sh test

  help                    Show this help message

NOTES:
  • API commands require the server to be running (./run.sh start)
  • All API operations require admin authentication (read from config.ini)
  • Use 'curl' and 'python3' must be available in PATH
  • Server runs on HTTPS (localhost:443)

EXAMPLES:
  ./run.sh start
  ./run.sh generate-key alice marketing "Marketing team user"
  ./run.sh list-keys alice
  ./run.sh renew-key 12345678-1234-1234-1234-123456789abc
  ./run.sh check-embedding 123
  ./run.sh update-embedding action/ticket-open-list.yaml
  ./run.sh check-embedding-file action/ticket-open-list.yaml
  ./run.sh stop
  ./run.sh regenerate-base-embedding

EOF
}

case "$1" in
    start|Start)
        start
        ;;
    stop|Stop)
        stop
        ;;
    status|Status)
        status
        ;;
    restart|Restart)
        restart
        ;;
    init|Init)
        init
        ;;
    generate-key)
        shift
        generate_key "$@"
        ;;
    list-keys)
        shift
        list_keys "$@"
        ;;
    verify-key)
        shift
        verify_key "$@"
        ;;
    delete-key)
        shift
        delete_key "$@"
        ;;
    renew-key)
        shift
        renew_key "$@"
        ;;
    set-config)
        shift
        set_config "$@"
        ;;
    check-embedding)
        shift
        check_embedding "$@"
        ;;
    regenerate-base-embedding)
        shift
        regenerate_base_embedding "$@"
        ;;
    regenerate-base-embedding-with-copy)
        shift
        regenerate_base_embedding_with_copy "$@"
        ;;
    update-embedding)
        shift
        update_embedding "$@"
        ;;
    check-embedding-file)
        shift
        check_embedding_file "$@"
        ;;
    clean|Clean)
        clean
        ;;
    test|Test)
        echo "🧪 Testing ai-console completions2 API..."
        # [server] port 를 config.ini에서 읽음 (섹션 인식 — 다른 섹션의 port 오인 방지)
        _PORT=$(awk -F'= *' '/^\[server\]/{s=1;next} /^\[/{s=0} s&&$1~/^port/{print $2;exit}' config.ini 2>/dev/null)
        curl -k -X POST "https://localhost:${_PORT:-443}/agent/chat/completions2" \
          -H "Content-Type: application/json" \
          ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
          -d '{
            "messages": [
              {
                "role": "system",
                "content": "You are a helpful assistant."
              },
              {
                "role": "user",
                "content": "이 콘솔은 무엇을 하는 도구인지 간단히 설명해주세요."
              }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": false
          }'
        ;;
    help|Help|-h|--help)
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

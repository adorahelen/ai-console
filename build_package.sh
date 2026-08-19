#!/bin/bash
# ===========================================================================
# build_package.sh — 배포 tar.gz 패키지 생성 (소스 / 오프라인 패키지 분리)
#
# 산출물 (기본):
#   dist/ai-console-deploy-<version>-source.tar.gz                   (~150 MB, 코드 + 스크립트)
#   dist/ai-console-deploy-<version>-offline-<target>-x86_64.tar.gz  (~9.5 GB, OS별 패키지)
#
# 두 tarball 의 inner 디렉토리는 동일 (ai-console-deploy-<version>/) 이라
# 같은 위치에 풀면 자동으로 머지됨:
#   tar xzf ai-console-deploy-<ver>-source.tar.gz
#   tar xzf ai-console-deploy-<ver>-offline-ubuntu-22.04-x86_64.tar.gz
#   cd ai-console-deploy-<ver>/   # 소스 + offline/ 둘 다 있음
#
# 사용법:
#   ./build_package.sh                          # 소스 + 오프라인 (현재 OS 기준)
#   ./build_package.sh --target ubuntu-22.04    # 오프라인 타겟 선택
#   ./build_package.sh --no-offline             # 소스만 (오프라인 tarball 스킵)
#   ./build_package.sh --no-source              # 오프라인만 (코드는 이미 배포된 경우)
#   ./build_package.sh --offline-skip-torch     # 오프라인 포함하되 torch wheels 제외
#   ./build_package.sh --version 1.0.0          # 명시적 버전
#   ./build_package.sh --with-llama-binary      # 빌드된 llama-server 포함
#   ./build_package.sh --output-dir /tmp/out    # 출력 경로 변경
#   ./build_package.sh --dry-run                # 포함될 파일 목록만 출력
#
# 워크플로우 (대상 서버에서):
#   tar xzf ai-console-deploy-<ver>-source.tar.gz
#   tar xzf ai-console-deploy-<ver>-offline-<target>-x86_64.tar.gz   # 오프라인 모드면
#   cd ai-console-deploy-<ver>/
#   cp shell/install.conf.example shell/install.conf && vi shell/install.conf
#   ./shell/install.sh --config shell/install.conf
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
# 출력 위치:
#   - source.tar.gz      → dist/         (가벼운 코드 패키지)
#   - offline.tar.gz     → dist/package/ (대용량 OS deb / wheel / docker image)
#   - SHA256SUMS         → dist/         (root 에 두 tarball 의 sha 모두 기록)
OUTPUT_DIR="${REPO_ROOT}/dist"
OFFLINE_DIR="${OUTPUT_DIR}/package"

# 버전: git tag > git short sha > 날짜
if git -C "$REPO_ROOT" describe --tags --exact-match 2>/dev/null; then
    DEFAULT_VERSION=$(git -C "$REPO_ROOT" describe --tags --exact-match 2>/dev/null)
elif GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null); then
    DEFAULT_VERSION="$(date +%Y%m%d)-${GIT_SHA}"
else
    DEFAULT_VERSION="$(date +%Y%m%d)"
fi
VERSION="$DEFAULT_VERSION"

WITH_LLAMA_BINARY=false
DRY_RUN=false
INCLUDE_SOURCE=true
INCLUDE_OFFLINE=true
OFFLINE_SKIP_TORCH=false
TARGET_OS=""

# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
log()  { echo -e "\033[1;32m[pkg]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --version)              VERSION="$2"; shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; OFFLINE_DIR="${OUTPUT_DIR}/package"; shift 2 ;;
        --with-llama-binary)    WITH_LLAMA_BINARY=true; shift ;;
        --no-source)            INCLUDE_SOURCE=false; shift ;;
        --no-offline)           INCLUDE_OFFLINE=false; shift ;;
        --offline-skip-torch)   OFFLINE_SKIP_TORCH=true; shift ;;
        --target)               TARGET_OS="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=true; shift ;;
        --help|-h)
            sed -n '3,/^# ==/p' "${BASH_SOURCE[0]}" | sed 's/^# *//;s/^#$//'
            exit 0
            ;;
        *) fail "알 수 없는 인자: $1 (--help 참조)" ;;
    esac
done

if [ "$INCLUDE_SOURCE" = false ] && [ "$INCLUDE_OFFLINE" = false ]; then
    fail "--no-source 와 --no-offline 를 같이 쓰면 만들 게 없습니다"
fi

# TARGET_OS 미지정 시 빌드 머신 OS 자동 감지
if [ -z "$TARGET_OS" ]; then
    if [ -r /etc/os-release ]; then
        BUILD_ID=$(. /etc/os-release && echo "${ID:-}")
        BUILD_VER=$(. /etc/os-release && echo "${VERSION_ID:-}")
        TARGET_OS="${BUILD_ID}-${BUILD_VER}"
    else
        TARGET_OS="linux"
    fi
fi
TARGET_VER="${TARGET_OS#*-}"

# 두 tarball 의 inner 디렉토리는 동일 → 풀 때 자동 머지.
# STAGE 는 offline tarball 과 같이 dist/package/ 안에 둔다 (cleanup 후 사라지지만 용량/위치 일관).
INNER_NAME="ai-console-deploy-${VERSION}"
STAGE="${OFFLINE_DIR}/${INNER_NAME}"

SOURCE_TARBALL="${OUTPUT_DIR}/ai-console-deploy-${VERSION}-source.tar.gz"
OFFLINE_TARBALL="${OFFLINE_DIR}/ai-console-deploy-${VERSION}-offline-${TARGET_OS}-x86_64.tar.gz"

log "REPO_ROOT:        $REPO_ROOT"
log "VERSION:          $VERSION"
log "TARGET_OS:        $TARGET_OS (offline 만 영향)"
log "OUTPUT_DIR:       $OUTPUT_DIR (source.tar.gz)"
log "OFFLINE_DIR:      $OFFLINE_DIR (offline.tar.gz)"
log "include source:   $INCLUDE_SOURCE"
log "include offline:  $INCLUDE_OFFLINE (skip-torch=$OFFLINE_SKIP_TORCH)"
log "with llama bin:   $WITH_LLAMA_BINARY"

# ---------------------------------------------------------------------------
# 제외 패턴 (소스 staging 용)
# ---------------------------------------------------------------------------
EXCLUDES=(
    --exclude='.git/'
    --exclude='.gitignore'
    --exclude='dist/'
    --exclude='build_package.sh'
    --exclude='*.pyc'
    --exclude='__pycache__/'
    --exclude='*/__pycache__/'

    --exclude='.install_state/'
    --exclude='shell/install.conf'
    --exclude='offline/'
    --exclude='config.ini'
    --exclude='config.ini.gpt'
    --exclude='config.ini.bak*'
    --exclude='config_gpt.ini'
    --exclude='api_keys/*'
    --exclude='ssl/selfsigned.*'
    --exclude='utils/.encryption_key'   # 인스턴스별 자동 생성. 타겟에 들고가면 안 됨.

    --exclude='logs/'
    --exclude='cache/'
    # tiktoken_cache/ 는 의도적으로 포함 — install 시 인터넷 없이 OfflineTiktoken 동작에 필요
    --exclude='qa_embeddings/'
    --exclude='qa_embeddings_gpt/'
    --exclude='qa_llm_*.log'
    --exclude='*.log'
    --exclude='*.bak.*'

    --exclude='_archive/'
    --exclude='benchmark/'
    --exclude='regression/'
    --exclude='tickets_sample/'
    --exclude='utils/test_only/'
    --exclude='handler_base.py.bak.*'

    # 내부 개발 노트 (배포 패키지에 포함 안 함)
    --exclude='dev_notes/'

    --exclude='llama-cpp-python/build*/'
    --exclude='llama-cpp-python/vendor/llama.cpp/build*/'
    --exclude='llama-cpp-python/.git/'
    --exclude='llama-cpp-python/vendor/llama.cpp/.git/'

    --exclude='query-validator/tests/'
    --exclude='query-validator/TEST_RESULTS.md'

    --exclude='.vscode/'
    --exclude='.idea/'
    --exclude='*.swp'
    --exclude='.claude/'
)

# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = true ]; then
    log "=== Dry-run: 포함될 소스 파일 목록 ==="
    rsync -avn --stats "${EXCLUDES[@]}" "$REPO_ROOT/" /tmp/pkg-dryrun/ 2>&1 | tail -30
    if [ "$INCLUDE_OFFLINE" = true ]; then
        TARGET_DEB_DIR="$REPO_ROOT/offline/deb-${TARGET_VER}"
        if [ -d "$TARGET_DEB_DIR" ]; then
            log "=== offline 으로 포함될 항목 ==="
            log "  $TARGET_DEB_DIR (→ offline/deb/)"
            for shared in pip-wheels miniconda docker-images MANIFEST.txt; do
                [ -e "$REPO_ROOT/offline/$shared" ] && log "  $REPO_ROOT/offline/$shared"
            done
        else
            warn "  offline/deb-${TARGET_VER}/ 없음 — 먼저 ./shell/collect_ubuntu_in_docker.sh ${TARGET_VER}"
        fi
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Staging 시작
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR" "$OFFLINE_DIR"
rm -rf "$STAGE"
mkdir -p "$STAGE"

command -v rsync >/dev/null 2>&1 || fail "rsync 필요. sudo apt install rsync"

# ---------------------------------------------------------------------------
# (1) 소스 코드 staging
# ---------------------------------------------------------------------------
log "[1] 소스 파일 staging"

# tiktoken_cache 사전 생성 — 비어있으면 generate_tiktoken_cache.py 실행해서 채움.
# 오프라인 타겟에서 BGE 인덱싱(7단계 init_system) 시 tiktoken 이 인터넷 다운로드 시도하므로
# 빌드 시점에 미리 캐시를 만들어 source.tar.gz 에 포함시킨다.
TIKTOKEN_CACHE_DIR="$REPO_ROOT/tiktoken_cache"
if [ ! -d "$TIKTOKEN_CACHE_DIR" ] || [ -z "$(ls -A "$TIKTOKEN_CACHE_DIR" 2>/dev/null)" ]; then
    if [ -f "$REPO_ROOT/generate_tiktoken_cache.py" ]; then
        log "    tiktoken_cache 비어있음 — generate_tiktoken_cache.py 실행 (인터넷 필요)"
        if python3 "$REPO_ROOT/generate_tiktoken_cache.py" 2>&1 | sed 's/^/      /'; then
            :
        else
            warn "    tiktoken_cache 생성 실패 — 오프라인 타겟에서 7단계가 fail 할 수 있음"
            warn "    수동: conda activate <env> && python3 generate_tiktoken_cache.py"
        fi
    else
        warn "    generate_tiktoken_cache.py 없음 — tiktoken_cache 누락된 채로 패키징"
    fi
fi

rsync -a "${EXCLUDES[@]}" "$REPO_ROOT/" "$STAGE/"

# 빈 디렉토리 보장 (런타임 필요)
mkdir -p "$STAGE/api_keys" "$STAGE/ssl" "$STAGE/logs" "$STAGE/cache" "$STAGE/qa_embeddings"

# llama-server 바이너리 포함
LLAMA_BIN_SRC="$REPO_ROOT/llama-cpp-python/vendor/llama.cpp/build-server/bin"
LLAMA_BIN_DST="$STAGE/llama-cpp-python/vendor/llama.cpp/build-server/bin"
if [ "$WITH_LLAMA_BINARY" = true ]; then
    if [ -x "$LLAMA_BIN_SRC/llama-server" ]; then
        log "    llama-server 바이너리 포함"
        mkdir -p "$LLAMA_BIN_DST"
        cp -a "$LLAMA_BIN_SRC/"llama-server "$LLAMA_BIN_DST/"
        cp -a "$LLAMA_BIN_SRC/"libggml-*.so* "$LLAMA_BIN_DST/" 2>/dev/null || true
        cp -a "$LLAMA_BIN_SRC/"libggml.so* "$LLAMA_BIN_DST/" 2>/dev/null || true
    else
        warn "--with-llama-binary 지정했지만 빌드된 바이너리 없음: $LLAMA_BIN_SRC/llama-server"
    fi
fi

# 메타데이터
echo "$VERSION" > "$STAGE/VERSION"
{
    echo "Version:          $VERSION"
    echo "Built at:         $(date -Iseconds)"
    echo "Built on:         $(hostname)"
    echo "Built by:         $(id -un)"
    echo "Git commit:       $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo 'n/a')"
    echo "Git branch:       $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
    echo "Target OS:        $TARGET_OS (offline tarball)"
    echo "llama-server:     $([ "$WITH_LLAMA_BINARY" = true ] && echo 'bundled' || echo 'build-on-target')"
    echo "Source tarball:   $([ "$INCLUDE_SOURCE" = true ] && echo "yes ($(basename "$SOURCE_TARBALL"))" || echo 'no')"
    echo "Offline tarball:  $([ "$INCLUDE_OFFLINE" = true ] && echo "yes ($(basename "$OFFLINE_TARBALL"))" || echo 'no')"
    [ "$INCLUDE_OFFLINE" = true ] && echo "skip-torch:       $OFFLINE_SKIP_TORCH"
} > "$STAGE/BUILD_INFO"

# README_INSTALL.md
cat > "$STAGE/README_INSTALL.md" <<MD
# ai-console 배포 패키지 설치 가이드

## 패키지 구성

이 배포는 **두 개의 tarball** 으로 분리됩니다:

1. \`ai-console-deploy-<ver>-source.tar.gz\` — 코드 + 설치 스크립트 (~150 MB)
2. \`ai-console-deploy-<ver>-offline-<target>-x86_64.tar.gz\` — OS별 .deb/.rpm + Docker 이미지 + Miniconda + pip wheels (~9.5 GB)

코드만 필요한 경우(빠른 코드 갱신 등)엔 source tarball 만 받으면 됩니다.
타겟 머신에 인터넷이 없으면 offline tarball 도 함께 풉니다.

## 요구사항 (타겟 머신)
- Ubuntu 20.04 / 22.04 / 24.04 (offline tarball 의 \`<target>\` 과 일치)
- NVIDIA 드라이버 ≥ 550, CUDA Toolkit 12.x
- sudo 권한, 디스크 50GB+, GPU 1장

## 모델 파일 배치
\`\`\`
<풀 곳>/
├── ai-console-deploy-<ver>/    ← source + offline 같이 풀린 디렉토리
└── models/               ← 별도 수급
    ├── gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf
    ├── meta-llama-3.1-8b-instruct-q4_k_m.gguf
    └── bge-m3/
\`\`\`

## 설치 순서

### A. 풀 오프라인 (인터넷 없음)
\`\`\`bash
tar xzf ai-console-deploy-<ver>-source.tar.gz
tar xzf ai-console-deploy-<ver>-offline-<target>-x86_64.tar.gz
cd ai-console-deploy-<ver>/
cp shell/install.conf.example shell/install.conf
vi shell/install.conf            # MARIADB_ROOT_PASSWORD 등 입력
./shell/install.sh --config shell/install.conf
\`\`\`

### B. 인터넷 가능 (offline tarball 불필요)
\`\`\`bash
tar xzf ai-console-deploy-<ver>-source.tar.gz
cd ai-console-deploy-<ver>/
cp shell/install.conf.example shell/install.conf && vi shell/install.conf
./shell/install.sh --config shell/install.conf
\`\`\`

\`offline/\` 디렉토리가 없으면 install 스크립트들이 인터넷 모드로 동작합니다.

## 코드만 갱신할 때
타겟에서 기존 \`offline/\` 디렉토리를 그대로 두고 새 source tarball 만 풉니다:
\`\`\`bash
cd <기존 ai-console-deploy 디렉토리 부모>/
tar xzf ai-console-deploy-<new-ver>-source.tar.gz
# offline/ 은 이전 디렉토리에서 복사하거나 그대로 두기
\`\`\`

## Tarball 검증
빌드 시 SHA-256 출력 → 옮긴 후 \`sha256sum\` 으로 비교.

## 자세한 가이드
설치/운영 가이드는 소스 리포지토리의 \`dev_notes/\` 폴더 참조 (배포 패키지에는 포함되지 않음):
- \`dev_notes/INSTALL_GUIDE.md\`
- \`dev_notes/INSTALL_GUIDE_RHEL.md\` (RHEL 계열)
MD

log "    ✓ 소스 staging 완료 ($(du -sh "$STAGE" --exclude=offline | cut -f1))"

# ---------------------------------------------------------------------------
# (2) Offline staging
# ---------------------------------------------------------------------------
if [ "$INCLUDE_OFFLINE" = true ]; then
    log "[2] offline staging ($STAGE/offline)"

    EXISTING_OFFLINE="$REPO_ROOT/offline"
    TARGET_DEB_DIR="$EXISTING_OFFLINE/deb-${TARGET_VER}"

    if [ -d "$TARGET_DEB_DIR" ]; then
        log "    multi-version 레이아웃 감지 — deb-${TARGET_VER}/ 사용"
        mkdir -p "$STAGE/offline"
        rsync -a "$TARGET_DEB_DIR/" "$STAGE/offline/deb/"
        for shared in pip-wheels miniconda docker-images MANIFEST.txt; do
            if [ -e "$EXISTING_OFFLINE/$shared" ]; then
                rsync -a "$EXISTING_OFFLINE/$shared" "$STAGE/offline/"
            fi
        done
    elif [ -d "$EXISTING_OFFLINE" ] && [ -f "$EXISTING_OFFLINE/MANIFEST.txt" ] && [ -d "$EXISTING_OFFLINE/deb" ]; then
        log "    구 레이아웃 (offline/deb/) 재사용"
        rsync -a "$EXISTING_OFFLINE/" "$STAGE/offline/"
    else
        log "    기존 offline/ 없음 — collect_packages.sh 실행"
        COLLECT_ARGS=(--output "$STAGE/offline")
        [ "$OFFLINE_SKIP_TORCH" = true ] && COLLECT_ARGS+=(--skip-torch)
        if ! "$REPO_ROOT/shell/collect_packages.sh" "${COLLECT_ARGS[@]}"; then
            fail "collect_packages.sh 실패. 수동 수집 후 재시도:
        ./shell/collect_packages.sh --output offline/
        ./shell/collect_ubuntu_in_docker.sh ${TARGET_VER}   # 타겟 OS 다른 경우"
        fi
    fi

    if [ ! -d "$STAGE/offline/deb" ] || [ -z "$(ls -A "$STAGE/offline/deb" 2>/dev/null)" ]; then
        fail "offline deb 디렉토리 비어있음 ($STAGE/offline/deb).
      먼저:  ./shell/collect_ubuntu_in_docker.sh ${TARGET_VER}"
    fi

    log "    ✓ offline staging 완료 ($(du -sh "$STAGE/offline" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# 비밀 누출 스캔 (소스만 — offline 은 .deb/wheel 바이너리)
# ---------------------------------------------------------------------------
log "[3] 비밀 누출 스캔 (소스 파일만)"
LEAK_PATTERNS=(
    'sk-svcacct-[A-Za-z0-9_-]\{40,\}'
    'sk-proj-[A-Za-z0-9_-]\{40,\}'
    'sk-None-[A-Za-z0-9_-]\{40,\}'
    'sk-ant-api[0-9]\{2\}-[A-Za-z0-9_-]\{40,\}'
    'sk-[A-Za-z0-9]\{48,\}'
    'xoxb-[0-9]\{10,\}-[0-9]\{10,\}-[A-Za-z0-9]\{24,\}'
    'xapp-[0-9]-[A-Z0-9]\{10,\}-[0-9]\{10,\}-[a-z0-9]\{50,\}'
    'AKIA[0-9A-Z]\{16\}'
    '-----BEGIN \(RSA\|EC\|OPENSSH\|DSA\) PRIVATE KEY-----'
)
SCAN_EXCLUDES=(
    --exclude-dir='examples'
    --exclude-dir='tests'
    --exclude-dir='test'
    --exclude-dir='offline'
    --exclude='*.ipynb'
)
LEAKS_FOUND=0
for pat in "${LEAK_PATTERNS[@]}"; do
    if MATCHES=$(grep -rlIE "${SCAN_EXCLUDES[@]}" "$pat" "$STAGE" 2>/dev/null); then
        if [ -n "$MATCHES" ]; then
            LEAKS_FOUND=1
            warn "비밀 패턴 감지 [$pat]:"
            echo "$MATCHES" | sed 's|^|    |' >&2
        fi
    fi
done
[ "$LEAKS_FOUND" = "1" ] && fail "비밀 정보 포함됨. exclude 보강 후 재시도. (스테이징: $STAGE)"
log "    ✓ 비밀 패턴 검출 없음"

# ---------------------------------------------------------------------------
# tarball 생성
# ---------------------------------------------------------------------------
# STAGE 가 OFFLINE_DIR 안에 있으므로 거기서 tar 작업
cd "$OFFLINE_DIR"

PRODUCED=()
if [ "$INCLUDE_SOURCE" = true ]; then
    log "[4a] 소스 tar.gz 생성 (offline/ 제외) → $OUTPUT_DIR"
    tar czf "$SOURCE_TARBALL" --exclude="${INNER_NAME}/offline" "${INNER_NAME}"
    PRODUCED+=("$SOURCE_TARBALL")
fi

if [ "$INCLUDE_OFFLINE" = true ]; then
    log "[4b] offline tar.gz 생성 → $OFFLINE_DIR"
    tar czf "$OFFLINE_TARBALL" "${INNER_NAME}/offline"
    PRODUCED+=("$OFFLINE_TARBALL")
fi

# 스테이징 정리
rm -rf "$STAGE"

# ---------------------------------------------------------------------------
# 결과 요약 + SHA256
# ---------------------------------------------------------------------------
log "완료"
echo ""
SHASUMS_FILE="${OUTPUT_DIR}/ai-console-deploy-${VERSION}.SHA256SUMS"
> "$SHASUMS_FILE"
for f in "${PRODUCED[@]}"; do
    SZ=$(du -h "$f" | cut -f1)
    SHA=$(sha256sum "$f" | cut -d' ' -f1)
    echo "$SHA  $(basename "$f")" >> "$SHASUMS_FILE"
    echo "📦 $(basename "$f")"
    echo "   크기:    $SZ"
    echo "   SHA-256: $SHA"
    echo ""
done
echo "SHA256SUMS 파일: $SHASUMS_FILE"

echo ""
echo "타겟 서버에서:"
if [ "$INCLUDE_SOURCE" = true ] && [ "$INCLUDE_OFFLINE" = true ]; then
    echo "  tar xzf $(basename "$SOURCE_TARBALL")"
    echo "  tar xzf $(basename "$OFFLINE_TARBALL")"
elif [ "$INCLUDE_SOURCE" = true ]; then
    echo "  tar xzf $(basename "$SOURCE_TARBALL")"
elif [ "$INCLUDE_OFFLINE" = true ]; then
    echo "  # 기존 ${INNER_NAME}/ 디렉토리에서:"
    echo "  tar xzf $(basename "$OFFLINE_TARBALL")"
fi
echo "  cd ${INNER_NAME}/"
echo "  cp shell/install.conf.example shell/install.conf && vi shell/install.conf"
echo "  ./shell/install.sh --config shell/install.conf"

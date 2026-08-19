#!/bin/bash
# =====================================================================
# build_images.sh — 빌드머신용
#
# Docker image 들을 빌드하고 docker save 로 tar 화 + compose / config /
# install 스크립트와 함께 packaging. 결과는 USB / scp 로 air-gap 타겟에 옮김.
#
# 사전 요구:
#   - 빌드머신에 docker + nvidia-container-toolkit (image build 시 GPU 필요)
#
# 사용:
#   ./build_images.sh                          # 기본: cu128 만 빌드 + packaging
#   ./build_images.sh --tags cu128,cu130       # 두 버전 모두 빌드 (driver 575+ 필요시 cu130 도)
#   ./build_images.sh --no-build               # 빌드 skip, save 만
#   ./build_images.sh --output-dir /tmp/x      # 다른 출력 경로
#   ./build_images.sh --cleanup-after-save     # save 끝난 후 docker rmi ai-console:* (디스크 절약)
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # docker/ 디렉토리
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # repo root (docker/ 의 부모)

VERSION="$(date +%Y%m%d)"
if GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null); then
    VERSION="${VERSION}-${GIT_SHA}"
fi

OUTPUT_DIR="${REPO_ROOT}/dist/package/ai-console-compose-${VERSION}"
DO_BUILD=true
DO_SAVE=true
DO_CLEANUP=false
TAGS="cu128"   # comma-separated. 예: cu128,cu130

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --no-build)           DO_BUILD=false;  shift ;;
        --tags)               TAGS="$2";       shift 2 ;;
        # --package-only: image build / save 둘 다 skip, 스크립트/설정/README 만 다시 생성.
        # 기존 패키지에 sync 할 때 유용 (`--output-dir <기존경로>` 와 같이).
        --package-only)       DO_BUILD=false; DO_SAVE=false; shift ;;
        # --cleanup-after-save: docker save 끝난 후 ai-console:*, ai-console-llama-server:* image 를 docker rmi.
        # mariadb / qdrant 는 작아서 유지. 빌드머신 / 파티션 디스크 압박 시 사용.
        --cleanup-after-save) DO_CLEANUP=true; shift ;;
        --help|-h)
            sed -n '3,/^# ==/p' "${BASH_SOURCE[0]}" | sed 's/^# *//;s/^#$//'
            exit 0 ;;
        *) echo "알 수 없는 인자: $1"; exit 1 ;;
    esac
done

# tag → CUDA_VER 매핑
cuda_ver_for() {
    case "$1" in
        cu128) echo "12.8.1" ;;
        cu130) echo "13.0.0" ;;
        *)     echo ""       ;;  # 알 수 없는 tag
    esac
}

log()  { echo -e "\033[1;34m[build]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[ ✓ ]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

# --- 사전 검증 -------------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "docker 미설치"
docker compose version >/dev/null 2>&1 || fail "docker compose plugin 미설치"

mkdir -p "$OUTPUT_DIR/images"

# --- (1) 이미지 빌드 (tag 별로 두 번까지) ------------------------------
IFS=',' read -ra TAG_LIST <<< "$TAGS"

if [ "$DO_BUILD" = true ]; then
    for tag in "${TAG_LIST[@]}"; do
        ver=$(cuda_ver_for "$tag")
        [ -n "$ver" ] || fail "알 수 없는 tag: $tag (cu128 / cu130 만 지원)"
        log "[1/3] docker build — ai-console:$tag / ai-console-llama-server:$tag (CUDA $ver)"

        docker build -t "ai-console:$tag" \
            --build-arg "CUDA_VER=$ver" \
            --build-arg "TORCH_CUDA=$tag" \
            -f "$SCRIPT_DIR/Dockerfile" \
            "$REPO_ROOT"

        docker build -t "ai-console-llama-server:$tag" \
            --build-arg "CUDA_VER=$ver" \
            -f "$SCRIPT_DIR/Dockerfile.llama-server" \
            "$REPO_ROOT"
    done
    ok "이미지 빌드 완료 (tags: ${TAGS})"
fi

# --- (2) 이미지 save → tar -------------------------------------------
if [ "$DO_SAVE" = true ]; then
    log "[2/3] docker save (이미지 → tar)"

    # mariadb / qdrant 는 tag 무관 공통
    docker pull mariadb:11               >/dev/null 2>&1 || true
    docker pull qdrant/qdrant:v1.12.1    >/dev/null 2>&1 || true
    docker save mariadb:11               -o "$OUTPUT_DIR/images/mariadb.tar"
    docker save qdrant/qdrant:v1.12.1    -o "$OUTPUT_DIR/images/qdrant.tar"
    ok  "  mariadb / qdrant 저장"

    for tag in "${TAG_LIST[@]}"; do
        log "  → ai-console-${tag}.tar / ai-console-llama-server-${tag}.tar"
        docker save "ai-console:$tag"               -o "$OUTPUT_DIR/images/ai-console-${tag}.tar"
        docker save "ai-console-llama-server:$tag"  -o "$OUTPUT_DIR/images/ai-console-llama-server-${tag}.tar"
        a=$(du -h "$OUTPUT_DIR/images/ai-console-${tag}.tar" | cut -f1)
        b=$(du -h "$OUTPUT_DIR/images/ai-console-llama-server-${tag}.tar" | cut -f1)
        ok "  ai-console:$tag ($a) / ai-console-llama-server:$tag ($b)"
    done
else
    log "[2/3] --package-only: docker save 건너뜀 (기존 images/ 그대로)"
fi

# --- (3) compose / config / install 스크립트 packaging ----------------
log "[3/3] config + install 스크립트 packaging"

# docker/ 안 모든 설정 파일 복사. install_compose.sh, *.example, ssl/, api_keys/(빈 디렉토리)도.
cp "$SCRIPT_DIR/docker-compose.yml"          "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/.env.example"                "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/config.ini.docker.example"   "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/install_compose.sh"          "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/install_host_prereqs.sh"     "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/manage_keys.sh"              "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/manage_stack.sh"             "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/INSTALL_GUIDE.md"            "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/QUICK_START.md"              "$OUTPUT_DIR/"
cp -r "$SCRIPT_DIR/ssl"                      "$OUTPUT_DIR/ssl" 2>/dev/null || true
mkdir -p "$OUTPUT_DIR/api_keys" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/cache" "$OUTPUT_DIR/utils-docker"

# README — 짧은 quick-start. 자세한 건 INSTALL_GUIDE.md 참조.
cat > "$OUTPUT_DIR/README.md" <<EOF
# ai-console Compose Deploy ($VERSION)

> 자세한 설치 / 운영 / 트러블슈팅: **[INSTALL_GUIDE.md](INSTALL_GUIDE.md)** 참조.

## 포함된 image tag
- tags 빌드: ${TAGS}
- \`ai-console:<tag>\`, \`ai-console-llama-server:<tag>\` — tag 별 CUDA 버전 (cu128 / cu130)
- \`mariadb:11\`, \`qdrant/qdrant:v1.12.1\` — 공식

## 하드웨어 요구사항
| 항목 | 최소 | 권장 |
|---|---|---|
| GPU VRAM | 24 GB | **48 GB** (L40S, A6000, A100-40) |
| CPU | 4 vCPU | 8 vCPU |
| **RAM (4K docs 기준)** | **48 GB** | **64 GB** ⚠️ |
| 디스크 | 80 GB | 120 GB SSD |

> **RAM 주의**: BGE-M3 인덱싱이 doc 당 ~4.8 MB Python 객체 누적. 데이터셋 클수록 RAM 선형 증가. 32 GB 인스턴스는 4K docs 못 버팀 — INSTALL_GUIDE.md §0 참조.

## SW 사전 요구
- NVIDIA driver 호환 (cu128 → 535+, cu130 → 575+)
- Docker 24+ + nvidia-container-toolkit + docker compose plugin v2+
- → \`install_host_prereqs.sh\` 가 Ubuntu 20.04/22.04, RHEL 9 자동 설치 (오프라인은 \`--info\`).

## Quick Start
\`\`\`bash
cp -r ai-console-compose-${VERSION}/ /opt/ai-console && cd /opt/ai-console

# 0) host SW (driver/docker/toolkit) — 이미 깔려있으면 skip
sudo bash install_host_prereqs.sh -y          # GPU 자동 감지 → 추천 driver/CUDA
# 드라이버 신규 설치면 reboot 후:
sudo bash install_host_prereqs.sh --skip-driver -y

# 1) 모델 (~22GB) 별도 옮기기
sudo mkdir -p /service/models && sudo cp -r /usb/models/* /service/models/

# 2) 환경 편집 (AI_CONSOLE_TAG / CUDA_VER / TORCH_CUDA 셋이 서로 일치해야)
cp .env.example .env && vi .env
cp config.ini.docker.example config.ini.docker

# 3) 설치
sudo bash install_compose.sh
\`\`\`

> **오프라인 (air-gap)**: 인터넷 없으면 \`bash install_host_prereqs.sh --info\` 로 받아야 할 .deb/.rpm 목록 + 추천 CUDA tag 확인.

성공 시 마지막에 API key + 접속 URL 표시됨.

## 자주 쓰는 운영 명령
\`\`\`bash
docker compose ps                     # 상태
docker compose logs -f app            # 로그
docker compose restart app            # 앱만 재시작
docker compose down && docker compose up -d   # 전체 재기동
\`\`\`
EOF

ok  "packaging 완료: $OUTPUT_DIR"
echo ""
log "산출물:"
du -sh "$OUTPUT_DIR"/* 2>/dev/null

echo ""
log "USB 로 옮기기:"
echo "   tar czf ai-console-compose-${VERSION}.tar.gz -C $(dirname "$OUTPUT_DIR") $(basename "$OUTPUT_DIR")"
echo "   또는: cp -r '$OUTPUT_DIR' /mnt/usb/"

# --- (4) cleanup: --cleanup-after-save 시 docker rmi (mariadb/qdrant 는 유지) ---
if [ "$DO_CLEANUP" = true ] && [ "$DO_SAVE" = true ]; then
    echo ""
    log "[cleanup] docker rmi ai-console:*  /  ai-console-llama-server:*  (mariadb / qdrant 는 유지)"
    before=$(docker system df --format "{{.Size}}" 2>/dev/null | head -1 || echo "?")
    for tag in "${TAG_LIST[@]}"; do
        for img in "ai-console:$tag" "ai-console-llama-server:$tag"; do
            if docker image inspect "$img" >/dev/null 2>&1; then
                docker rmi "$img" >/dev/null 2>&1 && ok "  rmi $img" || true
            fi
        done
    done
    docker image prune -f >/dev/null 2>&1 || true
    after=$(docker system df --format "{{.Size}}" 2>/dev/null | head -1 || echo "?")
    ok "정리 완료 (docker images: $before → $after)"
    log "다음 빌드 시 layer cache 일부 손실로 빌드가 느려질 수 있음."
fi

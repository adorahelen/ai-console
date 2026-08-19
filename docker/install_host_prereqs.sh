#!/bin/bash
# =====================================================================
# install_host_prereqs.sh — host 사전 SW 자동 설치
#
# 지원 OS: Ubuntu 20.04, Ubuntu 22.04, RHEL 9 (Rocky/Alma 9 포함)
#
# 설치 항목:
#   1. NVIDIA driver  (GPU 감지 → 자동 선택. Blackwell→575, 그 외→535)
#   2. Docker CE 24+ + compose plugin
#   3. nvidia-container-toolkit
#
# 흐름:
#   - 드라이버 신규 설치 시 reboot 필요 → 스크립트가 멈추고 안내
#   - reboot 후 `--skip-driver` 로 재실행하면 docker/toolkit 이어서 설치
#
# 사용:
#   sudo bash install_host_prereqs.sh                       # 대화형 + GPU 자동 감지 (cu128 기준)
#   sudo bash install_host_prereqs.sh -y                    # 확인 prompt 생략
#   sudo bash install_host_prereqs.sh --cuda cu130 -y       # cu130 이미지 사용 의도 → 575 강제
#   sudo bash install_host_prereqs.sh --driver-version 575  # 자동 감지 무시하고 강제
#   sudo bash install_host_prereqs.sh --skip-driver         # reboot 후 2회차
#   sudo bash install_host_prereqs.sh --skip-test           # GPU 컨테이너 동작 테스트 생략
#   bash install_host_prereqs.sh --info                     # 설치 안하고 다운로드 목록만 출력
#
# 오프라인 (air-gap) 환경:
#   - 인터넷 미감지 시 자동으로 --info 와 동일한 다운로드 가이드 출력 후 종료
#   - 빌드머신에서 .deb/.rpm 받아서 USB 로 옮긴 다음 수동 설치
#
# driver / CUDA 호환:
#   cu128 이미지 → driver 535+ 필요 (Ampere/Ada/Hopper 는 535 충분, Blackwell 은 575)
#   cu130 이미지 → driver 575+ 필수 (GPU 무관)
# =====================================================================

set -euo pipefail

DRIVER_VERSION=""        # 비어있으면 GPU 감지해서 자동 선택 (--driver-version 으로 override)
CUDA_TAG="cu128"         # cu128 (default) | cu130. cu130 은 driver 575+ 강제
SKIP_DRIVER=false
SKIP_DOCKER=false
SKIP_TOOLKIT=false
SKIP_TEST=false
ASSUME_YES=false
INFO_MODE=false          # --info: 설치 안 하고 다운로드 가이드만 출력
TARGET_USER="${SUDO_USER:-$USER}"

log()  { echo -e "\033[1;34m[host-prereqs]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[ ✓ ]\033[0m $*"; }
warn() { echo -e "\033[1;33m[ ! ]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

# --info / 오프라인 가이드 출력에서 패키지명 강조용 ANSI 색상.
# TTY 일 때만 적용 (파이프 / redirect 시 빈 문자열 → ANSI escape 가 로그에 안 섞임).
if [ -t 1 ]; then
    PKG=$'\033[1;33m'   # bold yellow — 다운로드 패키지명
    SEC=$'\033[1;36m'   # bold cyan — 섹션 제목 (driver/docker/toolkit)
    RST=$'\033[0m'
else
    PKG=""; SEC=""; RST=""
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --driver-version) DRIVER_VERSION="$2"; shift 2 ;;
        --cuda)           CUDA_TAG="$2";       shift 2 ;;
        --skip-driver)    SKIP_DRIVER=true;    shift ;;
        --skip-docker)    SKIP_DOCKER=true;    shift ;;
        --skip-toolkit)   SKIP_TOOLKIT=true;   shift ;;
        --skip-test)      SKIP_TEST=true;      shift ;;
        --info)           INFO_MODE=true;      shift ;;
        -y|--yes)         ASSUME_YES=true;     shift ;;
        --help|-h)
            sed -n '3,/^# ==/p' "${BASH_SOURCE[0]}" | sed 's/^# *//;s/^#$//'
            exit 0 ;;
        *) fail "알 수 없는 인자: $1 (--help 참조)" ;;
    esac
done

if [ "$INFO_MODE" = false ]; then
    [ "$EUID" -eq 0 ] || fail "root 권한 필요 (sudo bash $0 ...)"
fi

case "$CUDA_TAG" in
    cu128|cu130) ;;
    *) fail "--cuda 는 cu128 또는 cu130 만 지원 (받은 값: $CUDA_TAG)" ;;
esac

# ========== OS 감지 ==========
[ -f /etc/os-release ] || fail "/etc/os-release 없음"
. /etc/os-release
case "${ID}-${VERSION_ID}" in
    ubuntu-20.04)                    OS=ubuntu;  UBU_CODENAME=focal  ;;
    ubuntu-22.04)                    OS=ubuntu;  UBU_CODENAME=jammy  ;;
    rhel-9*|rocky-9*|almalinux-9*)   OS=rhel9                          ;;
    *) fail "지원하지 않는 OS: $PRETTY_NAME (Ubuntu 20.04/22.04, RHEL 9 만)" ;;
esac
ok "OS: $PRETTY_NAME"

confirm() {
    [ "$ASSUME_YES" = true ] && return 0
    read -r -p "$1 [y/N]: " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

# ========== 인터넷 연결 감지 ==========
# Docker / NVIDIA repo 둘 중 하나라도 닿으면 online. 5초 timeout.
check_internet() {
    for url in https://download.docker.com/linux/ https://nvidia.github.io/libnvidia-container/; do
        if curl -fsS --connect-timeout 5 --max-time 5 -o /dev/null "$url" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

# ========== GPU 세대 감지 (글로벌 변수에 1회 populate) ==========
# 결과: GPU_LINE (lspci raw line), GPU_GEN (blackwell|hopper|ada|ampere|unknown)
# (PCI device id 영역 근사:
#   Blackwell 10de:2900-29ff,2b00-2bff / Hopper 10de:2330-233f
#   Ada       10de:2600-27ff           / Ampere 10de:2080-25ff)
GPU_LINE=""
GPU_GEN="unknown"
populate_gpu_info() {
    if ! command -v lspci >/dev/null 2>&1; then
        # info/오프라인 모드면 패키지 install 시도 안 함 (root 권한 + repo 둘 다 필요)
        if [ "$INFO_MODE" = false ] && [ "${OFFLINE_MODE:-false}" = false ] && [ "$EUID" -eq 0 ]; then
            case "$OS" in
                ubuntu) DEBIAN_FRONTEND=noninteractive apt install -y pciutils >/dev/null 2>&1 || true ;;
                rhel9)  dnf install -y pciutils >/dev/null 2>&1 || true ;;
            esac
        fi
    fi
    command -v lspci >/dev/null 2>&1 || return
    GPU_LINE=$(lspci -nn 2>/dev/null | grep -iE 'vga|3d|display' | grep -i nvidia | head -1)
    [ -z "$GPU_LINE" ] && return

    if   echo "$GPU_LINE" | grep -qiE 'rtx ?50[0-9]{2}|b1[0-9]{2}|b2[0-9]{2}|gb[12][0-9]{2}|blackwell|10de:(29|2b)[0-9a-f]{2}'; then GPU_GEN=blackwell
    elif echo "$GPU_LINE" | grep -qiE 'h100|h200|hopper|10de:233[0-9a-f]'; then GPU_GEN=hopper
    elif echo "$GPU_LINE" | grep -qiE 'rtx ?40[0-9]{2}|l40|l4 |ada|10de:(26|27)[0-9a-f]{2}'; then GPU_GEN=ada
    elif echo "$GPU_LINE" | grep -qiE 'a100|a40|a6000|a10[^0-9]|rtx ?30[0-9]{2}|ampere|10de:(20[89a-f]|21|22|23[0-2]|24|25)[0-9a-f]{1,2}'; then GPU_GEN=ampere
    fi
}

# ========== driver 버전 추천 ==========
# 우선순위:
#   (0) --cuda cu130 → 575 강제
#   (1) Ubuntu: `ubuntu-drivers devices` recommended
#   (2) GPU 세대 (populate_gpu_info 결과): Blackwell→575 / 그 외→535 / unknown→575
detect_driver_version() {
    if [ "$CUDA_TAG" = "cu130" ]; then
        log "--cuda cu130 → driver 575 강제" >&2
        echo "575"; return
    fi

    if [ "$OS" = "ubuntu" ]; then
        DEBIAN_FRONTEND=noninteractive apt install -y ubuntu-drivers-common >/dev/null 2>&1 || true
        if command -v ubuntu-drivers >/dev/null 2>&1; then
            local rec
            rec=$(ubuntu-drivers devices 2>/dev/null \
                | grep -E 'nvidia-driver-[0-9]+ - .*recommended' \
                | grep -oE 'nvidia-driver-[0-9]+' | head -1 | grep -oE '[0-9]+$')
            if [ -n "$rec" ]; then
                log "ubuntu-drivers 추천: nvidia-driver-$rec" >&2
                echo "$rec"; return
            fi
        fi
    fi

    [ -n "$GPU_LINE" ] && log "감지된 GPU: $GPU_LINE" >&2
    case "$GPU_GEN" in
        blackwell) log "세대: Blackwell → 575" >&2;     echo "575" ;;
        hopper)    log "세대: Hopper → 535"    >&2;     echo "535" ;;
        ada)       log "세대: Ada Lovelace → 535" >&2;  echo "535" ;;
        ampere)    log "세대: Ampere → 535"    >&2;     echo "535" ;;
        *)         warn "GPU 세대 판별 실패 — default 575" >&2; echo "575" ;;
    esac
}

# ========== CUDA tag 추천 (스크립트 끝에서 출력) ==========
# 출력 포맷: "cu128|이유" 또는 "cu130|이유"
recommend_cuda_tag() {
    local driver_major=0
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
            | head -1 | cut -d. -f1)
    fi
    driver_major=${driver_major:-0}

    case "$GPU_GEN" in
        blackwell)
            if [ "$driver_major" -ge 575 ] 2>/dev/null; then
                echo "cu130|Blackwell GPU + driver $driver_major.x — sm_120 native, 이미지 30% 작음"
            else
                echo "cu128|Blackwell GPU 지만 driver $driver_major < 575 라 cu130 불가 (driver 업그레이드 시 cu130)"
            fi
            ;;
        hopper|ada|ampere)
            echo "cu128|$GPU_GEN GPU — 안정적, driver 535+ 호환 폭 넓음 (cu130 도 driver 575+ 면 가능)"
            ;;
        *)
            if [ "$driver_major" -ge 575 ] 2>/dev/null; then
                echo "cu128|GPU 세대 미상이나 driver $driver_major.x — 둘 다 가능, 안정성 위해 cu128 권장"
            else
                echo "cu128|GPU/driver 정보 불충분 — default cu128"
            fi
            ;;
    esac
}

# ========== 오프라인 / --info 플랜 출력 ==========
# 인터넷 미감지 또는 --info 시 호출. 다운로드 대상 + URL + 추천 CUDA tag 까지 안내.
print_offline_plan() {
    local rec rec_tag rec_reason rec_ver
    rec=$(recommend_cuda_tag)
    rec_tag="${rec%%|*}"
    rec_reason="${rec#*|}"
    rec_ver=12.8.1; [ "$rec_tag" = "cu130" ] && rec_ver=13.0.0

    # 추천 driver = --driver-version 명시값 / cu130 강제 575 / GPU 세대 매핑
    local rec_drv="${DRIVER_VERSION}"
    if [ -z "$rec_drv" ]; then
        if [ "$CUDA_TAG" = "cu130" ]; then rec_drv=575
        else
            case "$GPU_GEN" in
                blackwell) rec_drv=575 ;;
                hopper|ada|ampere) rec_drv=535 ;;
                *) rec_drv=575 ;;
            esac
        fi
    fi

    cat <<EOF

============================================================
   📋 오프라인 / --info — 사전 다운로드 가이드
============================================================
   감지된 OS         : $PRETTY_NAME
   감지된 GPU        : ${GPU_LINE:-(미감지 — lspci/pciutils 없음)}
   GPU 세대          : $GPU_GEN

   👉 추천 CUDA 이미지: ${rec_tag}
      이유            : ${rec_reason}
      .env 적용:        AI_CONSOLE_TAG=${rec_tag}  CUDA_VER=${rec_ver}  TORCH_CUDA=${rec_tag}

   👉 추천 NVIDIA driver: ${rec_drv}
      (cu128 → 535+ / cu130 → 575+)
============================================================

EOF

    case "$OS" in
        ubuntu) print_offline_plan_ubuntu "$rec_drv" ;;
        rhel9)  print_offline_plan_rhel "$rec_drv"   ;;
    esac

    cat <<EOF

============================================================
   타겟 (air-gap) 설치 순서
============================================================
   1) USB 로 위 .deb/.rpm 들 옮기기
   2) [driver]   sudo dpkg -i nvidia-driver-${rec_drv}*.deb     # ubuntu
                 sudo dnf install ./nvidia-driver*.rpm           # rhel9
                 sudo reboot
   3) [docker]   sudo dpkg -i docker-*.deb containerd.io*.deb
   4) [toolkit]  sudo dpkg -i nvidia-container-toolkit*.deb libnvidia-container*.deb
                 sudo nvidia-ctk runtime configure --runtime=docker
                 sudo systemctl enable --now docker
                 sudo systemctl restart docker
   5) [검증]     docker run --rm --gpus all nvidia/cuda:${rec_ver}-base-ubuntu22.04 nvidia-smi
   6) cd /service/ai-console && sudo bash install_compose.sh
============================================================
EOF
}

print_offline_plan_ubuntu() {
    local drv="$1"
    cat <<EOF
${SEC}[1] NVIDIA driver $drv${RST}  (Ubuntu $VERSION_ID / $UBU_CODENAME)
    옵션 A — graphics-drivers PPA (인터넷 머신에서 download 만):
      sudo add-apt-repository -y ppa:graphics-drivers/ppa
      sudo apt update
      mkdir -p offline/driver && cd offline/driver
      apt download ${PKG}nvidia-driver-$drv${RST} ${PKG}linux-modules-nvidia-$drv-generic${RST} \\
                   ${PKG}libnvidia-compute-$drv${RST} ${PKG}libnvidia-decode-$drv${RST} ${PKG}libnvidia-encode-$drv${RST} \\
                   ${PKG}libnvidia-fbc1-$drv${RST} ${PKG}libnvidia-gl-$drv${RST} ${PKG}nvidia-compute-utils-$drv${RST} \\
                   ${PKG}nvidia-kernel-common-$drv${RST} ${PKG}nvidia-utils-$drv${RST}
    옵션 B — NVIDIA .run installer (배포판 무관, 가장 깔끔):
      https://www.nvidia.com/Download/index.aspx
        → product:  GPU 모델 ($GPU_GEN)
        → OS:       Linux 64-bit
        → branch:   $drv (production)
      → ${PKG}NVIDIA-Linux-x86_64-${drv}.xx.run${RST}

${SEC}[2] Docker CE + compose plugin${RST}
    Repository: https://download.docker.com/linux/ubuntu/dists/${UBU_CODENAME}/pool/stable/amd64/
    필요 .deb (최신 안정버전 5종):
      - ${PKG}containerd.io_*_amd64.deb${RST}
      - ${PKG}docker-ce_*_amd64.deb${RST}
      - ${PKG}docker-ce-cli_*_amd64.deb${RST}
      - ${PKG}docker-buildx-plugin_*_amd64.deb${RST}
      - ${PKG}docker-compose-plugin_*_amd64.deb${RST}
    또는 한 번에 (인터넷 머신):
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      echo "deb [signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${UBU_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list
      sudo apt update
      mkdir -p offline/docker && cd offline/docker
      apt download ${PKG}docker-ce${RST} ${PKG}docker-ce-cli${RST} ${PKG}containerd.io${RST} ${PKG}docker-buildx-plugin${RST} ${PKG}docker-compose-plugin${RST}

${SEC}[3] nvidia-container-toolkit${RST}
    Repository: https://nvidia.github.io/libnvidia-container/stable/deb/amd64/
    필요 .deb:
      - ${PKG}libnvidia-container1_*_amd64.deb${RST}
      - ${PKG}libnvidia-container-tools_*_amd64.deb${RST}
      - ${PKG}nvidia-container-toolkit-base_*_amd64.deb${RST}
      - ${PKG}nvidia-container-toolkit_*_amd64.deb${RST}
    또는 한 번에 (인터넷 머신):
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      sudo apt update
      mkdir -p offline/toolkit && cd offline/toolkit
      apt download ${PKG}nvidia-container-toolkit${RST} ${PKG}nvidia-container-toolkit-base${RST} \\
                   ${PKG}libnvidia-container1${RST} ${PKG}libnvidia-container-tools${RST}

${SEC}[4] (선택) GPU 동작 검증용 base 이미지 (~150MB)${RST}
      docker pull ${PKG}nvidia/cuda:12.8.1-base-ubuntu22.04${RST}
      docker save ${PKG}nvidia/cuda:12.8.1-base-ubuntu22.04${RST} -o offline/cuda-base.tar
EOF
}

print_offline_plan_rhel() {
    local drv="$1"
    cat <<EOF
${SEC}[1] NVIDIA driver $drv${RST}  (RHEL 9)
    Repository: https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/
    (정식 RHEL: ${PKG}epel-release-latest-9.noarch.rpm${RST} 도 별도 설치 필요)
    옵션 A — dnf download (인터넷 머신):
      sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
      sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
      mkdir -p offline/driver && cd offline/driver
      sudo dnf download --resolve --alldeps \\
        ${PKG}nvidia-driver-${drv}-dkms${RST} || sudo dnf download --resolve --alldeps ${PKG}nvidia-driver${RST}
    옵션 B — NVIDIA .run installer:
      https://www.nvidia.com/Download/index.aspx → ${drv} branch
      (${PKG}kernel-devel${RST}, ${PKG}kernel-headers${RST}, ${PKG}gcc${RST}, ${PKG}make${RST}, ${PKG}dkms${RST} 도 같이 받아둘 것)

${SEC}[2] Docker CE + compose plugin${RST}
    Repository: https://download.docker.com/linux/rhel/9/x86_64/stable/Packages/
    인터넷 머신:
      sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
      mkdir -p offline/docker && cd offline/docker
      sudo dnf download --resolve --alldeps \\
        ${PKG}docker-ce${RST} ${PKG}docker-ce-cli${RST} ${PKG}containerd.io${RST} ${PKG}docker-buildx-plugin${RST} ${PKG}docker-compose-plugin${RST}

${SEC}[3] nvidia-container-toolkit${RST}
    Repository: https://nvidia.github.io/libnvidia-container/stable/rpm/
    인터넷 머신:
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \\
        | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
      mkdir -p offline/toolkit && cd offline/toolkit
      sudo dnf download --resolve --alldeps ${PKG}nvidia-container-toolkit${RST}

${SEC}[4] (선택) GPU 동작 검증용 base 이미지 (~150MB)${RST}
      docker pull ${PKG}nvidia/cuda:12.8.1-base-ubuntu22.04${RST}
      docker save ${PKG}nvidia/cuda:12.8.1-base-ubuntu22.04${RST} -o offline/cuda-base.tar
EOF
}

# GPU 정보 1회 채움 (이후 detect_driver_version / recommend_cuda_tag / print_offline_plan 가 글로벌 참조)
populate_gpu_info

# ========== --info / 오프라인 감지 ==========
if [ "$INFO_MODE" = true ]; then
    log "--info 모드: 다운로드 가이드만 출력 (실제 설치 안 함)"
    print_offline_plan
    exit 0
fi

# 모든 설치 단계가 skip 이면 인터넷 검사 의미 없음
if [ "$SKIP_DRIVER" = false ] || [ "$SKIP_DOCKER" = false ] || [ "$SKIP_TOOLKIT" = false ]; then
    log "인터넷 연결 확인…"
    if ! check_internet; then
        OFFLINE_MODE=true
        warn "인터넷 미감지 (download.docker.com / nvidia.github.io 모두 도달 실패)"
        warn "오프라인 모드 — 자동 설치 불가. 다운로드 가이드 출력 후 종료."
        print_offline_plan
        exit 2
    fi
    ok "인터넷 OK"
fi

# ========== Plan ==========
log "설치 계획: (대상 CUDA 이미지: $CUDA_TAG)"
[ "$SKIP_DRIVER"  = false ] && echo "    - NVIDIA driver ${DRIVER_VERSION:-<GPU/CUDA 감지 후 자동>}"
[ "$SKIP_DOCKER"  = false ] && echo "    - Docker CE + docker compose plugin"
[ "$SKIP_TOOLKIT" = false ] && echo "    - nvidia-container-toolkit"
[ "$TARGET_USER" != "root" ] && echo "    - 사용자 '$TARGET_USER' 를 docker 그룹에 추가"
confirm "진행할까?" || { log "취소"; exit 0; }

# ========== (1) NVIDIA driver ==========
install_driver_ubuntu() {
    log "[1/4] NVIDIA driver $DRIVER_VERSION (apt + graphics-drivers PPA)"
    apt update
    apt install -y software-properties-common ca-certificates curl gnupg
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt update
    # 기존 nvidia 패키지 충돌 방지 — 깔린 게 없으면 no-op
    apt remove -y --purge '^nvidia-.*' '^libnvidia-.*' 2>/dev/null || true
    apt install -y "nvidia-driver-${DRIVER_VERSION}"
}

install_driver_rhel() {
    log "[1/4] NVIDIA driver $DRIVER_VERSION (dnf, NVIDIA cuda repo)"
    dnf install -y dnf-plugins-core
    # EPEL: CentOS Stream / Rocky / Alma 는 이름 그대로, 정식 RHEL 은 RPM URL 로.
    # 의존성 일부(kernel-devel 등)에서 필요. 실패해도 warn 만 (NVIDIA repo 만으로도 종종 OK).
    if ! rpm -q epel-release >/dev/null 2>&1; then
        if dnf install -y epel-release >/dev/null 2>&1; then
            ok "  epel-release 설치"
        elif dnf install -y "https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm" >/dev/null 2>&1; then
            ok "  epel-release 설치 (fedora URL)"
        else
            warn "  epel-release 설치 실패 — driver dependency 부족 시 수동 설치 필요"
        fi
    fi
    # 정식 RHEL 은 codeready-builder 레포가 kernel-devel 등 일부 dependency 제공
    # (subscription-manager 가 있을 때만 동작; 없으면 무시)
    if command -v subscription-manager >/dev/null 2>&1; then
        subscription-manager repos --enable "codeready-builder-for-rhel-9-$(uname -m)-rpms" >/dev/null 2>&1 || true
    fi
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    dnf clean all
    # nvidia-driver module: 특정 버전 stream 이 없으면 latest-dkms 로 fallback
    if dnf module info "nvidia-driver:${DRIVER_VERSION}-dkms" >/dev/null 2>&1; then
        dnf module install -y "nvidia-driver:${DRIVER_VERSION}-dkms"
    else
        warn "nvidia-driver:${DRIVER_VERSION}-dkms stream 없음 → latest-dkms 사용"
        dnf module install -y nvidia-driver:latest-dkms
    fi
}

if [ "$SKIP_DRIVER" = false ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        cur=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        cur_major=${cur%%.*}
        ok "기존 driver 감지 (version $cur) — 설치 건너뜀"
        # cu128 → 535+, cu130 → 575+ 미달이면 경고
        min_required=535
        [ "$CUDA_TAG" = "cu130" ] && min_required=575
        if [ "$cur_major" -lt "$min_required" ] 2>/dev/null; then
            warn "기존 driver $cur 이 $CUDA_TAG 이미지 최소요건($min_required+) 보다 낮음 — 업그레이드 권장"
            warn "  업그레이드: $0 (--skip-driver 빼고 재실행) 또는 수동 driver 교체"
        fi
    else
        if [ -z "$DRIVER_VERSION" ]; then
            DRIVER_VERSION=$(detect_driver_version)
            ok "자동 감지된 driver: $DRIVER_VERSION"
        else
            ok "지정된 driver: $DRIVER_VERSION (--driver-version)"
        fi
        case "$OS" in
            ubuntu) install_driver_ubuntu ;;
            rhel9)  install_driver_rhel  ;;
        esac
        cat <<EOF

============================================================
   ⚠️  드라이버 신규 설치됨 — REBOOT 필요
============================================================
   $ sudo reboot

   재부팅 후:
   $ sudo bash $0 --skip-driver -y
============================================================
EOF
        exit 0
    fi
fi

# ========== (2) Docker CE + compose plugin ==========
install_docker_ubuntu() {
    log "[2/4] Docker CE (apt, docker.com 공식 repo)"
    apt update
    apt install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${UBU_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list
    apt update
    apt install -y docker-ce docker-ce-cli containerd.io \
                   docker-buildx-plugin docker-compose-plugin
}

install_docker_rhel() {
    log "[2/4] Docker CE (dnf, docker.com 공식 repo)"
    dnf install -y dnf-plugins-core
    dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
    dnf install -y docker-ce docker-ce-cli containerd.io \
                   docker-buildx-plugin docker-compose-plugin
}

if [ "$SKIP_DOCKER" = false ]; then
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        ok "기존 docker 감지 ($(docker --version | awk '{print $3}' | tr -d ,)) — 설치 건너뜀"
    else
        case "$OS" in
            ubuntu) install_docker_ubuntu ;;
            rhel9)  install_docker_rhel  ;;
        esac
    fi
    systemctl enable --now docker

    if [ "$TARGET_USER" != "root" ]; then
        usermod -aG docker "$TARGET_USER" || true
        ok "사용자 '$TARGET_USER' docker 그룹에 추가 (재로그인 후 적용)"
    fi
fi

# ========== (3) nvidia-container-toolkit ==========
install_toolkit_ubuntu() {
    log "[3/4] nvidia-container-toolkit (apt)"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt update
    apt install -y nvidia-container-toolkit
}

install_toolkit_rhel() {
    log "[3/4] nvidia-container-toolkit (dnf)"
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        | tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
    dnf install -y nvidia-container-toolkit
}

if [ "$SKIP_TOOLKIT" = false ]; then
    if command -v nvidia-ctk >/dev/null 2>&1; then
        ok "기존 nvidia-container-toolkit 감지 — 설치 건너뜀"
    else
        case "$OS" in
            ubuntu) install_toolkit_ubuntu ;;
            rhel9)  install_toolkit_rhel  ;;
        esac
    fi
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# ========== (4) 검증 ==========
log "[4/4] 검증"
ok "$(docker --version)"
ok "$(docker compose version)"
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi

if [ "$SKIP_TEST" = false ]; then
    log "GPU 컨테이너 동작 테스트 (CUDA base 이미지 ~150MB pull)"
    if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi -L; then
        ok "GPU 컨테이너 OK"
    else
        warn "GPU 컨테이너 테스트 실패 — driver/toolkit 설정 확인 필요"
        warn "  $ sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        exit 1
    fi
fi

# ========== CUDA 추천 ==========
REC=$(recommend_cuda_tag)
REC_TAG="${REC%%|*}"
REC_REASON="${REC#*|}"
REC_VER=12.8.1; [ "$REC_TAG" = "cu130" ] && REC_VER=13.0.0

cat <<EOF

============================================================
   ✅ host 사전 요구 설치 완료
============================================================
   GPU         : ${GPU_LINE:-(미감지)}
   driver      : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "(미설치)")

   👉 추천 CUDA tag: ${REC_TAG}
      이유: ${REC_REASON}

   /service/ai-console/.env 에 다음 3줄을 일치시켜 사용:
      AI_CONSOLE_TAG=${REC_TAG}
      CUDA_VER=${REC_VER}
      TORCH_CUDA=${REC_TAG}

   다음 단계:
     1) (sudo 없이 docker 쓰려면) 재로그인 또는: newgrp docker
     2) cd /service/ai-console && sudo bash install_compose.sh
============================================================
EOF

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# ai-console 원샷 설치기
#
#   curl -fsSL https://raw.githubusercontent.com/adorahelen/ai-console-public/main/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- --instance acme --yes
#
#   ./install.sh                  # 대화형: HW 감지 → 티어 → 프리셋 선택
#   ./install.sh --instance acme  # 고객사별 인스턴스 (포트 자동 회피)
#   ./install.sh --preset llama31-8b-q4 --yes
#   ./install.sh --dry-run        # 계획만 출력 (아무것도 설치 안 함)
#   ./install.sh --no-model       # 모델 다운로드 생략 (나중에 수동)
#   ./install.sh --fresh          # 완료 표식 무시하고 무거운 단계 재실행
#   ./install.sh --preset X --config-only  # 기설치 환경에서 config.ini만 재생성 (§3~7 생략)
#   ./install.sh --no-service     # systemd 등록·기동 생략 (수동 ./run.sh 운영)
#
# 하는 일: HW 감지 → 모델 프리셋 결정(models.yaml) → 포트 배정 → 시스템 의존성
#   → 파이썬 venv → llama.cpp 빌드 → Qdrant 바이너리 → 모델 다운로드
#   → config.ini 생성(+자체서명 SSL) → aibotctl PATH 등록 → **systemd 등록·기동·준비대기**
#
# 배포 형태 C안(고객사별 인스턴스): 인스턴스 = 클론 1개 + 자기 config·포트·Qdrant.
#   운영 가이드는 docs/multi-instance.md.
# 재실행 안전: 무거운 단계는 .install-state/ 표식으로 건너뛴다(--fresh 로 무시).
# ═══════════════════════════════════════════════════════════════

# ── 부트스트랩 ────────────────────────────────────────────────
# 여기까지는 **POSIX sh 로도 돌아야 한다** — `curl … | sh` 는 shebang을 무시하고
# dash로 실행하므로 set -o pipefail·[[ ]]·<<< 를 아직 쓰면 안 된다.
_self="$0"
case "$_self" in */*) _here="${_self%/*}" ;; *) _here="." ;; esac
_boot=0

if [ ! -f "$_here/models.yaml" ]; then
  # 스크립트만 파이프로 들어왔고 리포가 없다 → 클론하고 그쪽으로 넘긴다.
  command -v git >/dev/null 2>&1 || { echo "git이 필요합니다 (apt install git)"; exit 1; }
  _inst=default; _prev=
  for _a in "$@"; do
    if [ "$_prev" = "--instance" ]; then _inst="$_a"; fi
    _prev="$_a"
  done
  if [ "$_inst" = default ]; then _sfx=""; else _sfx="-$_inst"; fi
  _here="${AI_CONSOLE_DIR:-$HOME/ai-console$_sfx}"
  _repo="${AI_CONSOLE_REPO:-https://github.com/adorahelen/ai-console-public.git}"
  if [ -d "$_here/.git" ]; then
    echo "▸ 기존 클론 재사용: $_here"
  else
    echo "▸ 클론: $_repo → $_here"
    git clone --depth 1 "$_repo" "$_here" || exit 1
  fi
  _boot=1
fi
# 클론했거나 bash가 아니면 bash로 재진입 (재진입 후엔 두 조건 모두 거짓 → 무한루프 없음)
if [ "$_boot" = 1 ] || [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$_here/install.sh" "$@"
fi

# ── 여기부터 bash 전용 ────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

# ── 인자 파싱 ─────────────────────────────────────────────────
PRESET="" ; TIER="" ; YES=0 ; DRY=0 ; NO_MODEL=0 ; CONFIG_ONLY=0 ; FRESH=0 ; NO_SERVICE=0
INSTANCE="${AI_CONSOLE_INSTANCE:-default}"
while [ $# -gt 0 ]; do
  case "$1" in
    --preset)  PRESET="$2"; shift 2 ;;
    --tier)    TIER="$2"; shift 2 ;;
    --instance) INSTANCE="$2"; shift 2 ;;
    --yes|-y)  YES=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --no-model) NO_MODEL=1; shift ;;
    --fresh)   FRESH=1; shift ;;
    --config-only) CONFIG_ONLY=1; NO_MODEL=1; shift ;;
    --no-service) NO_SERVICE=1; shift ;;
    -h|--help) grep '^#' "$0" | head -22; exit 0 ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

# curl|sh 등 비대화(stdin이 TTY 아님)면 프리셋 메뉴·confirm이 EOF로 중단된다 → 비대화 자동 진행
# (기본 프리셋·자동 확인). 대화형 터미널이면 그대로 메뉴/확인을 띄운다.
[ -t 0 ] || YES=1

say()  { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*"; exit 1; }
run()  { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }

# ── 진행 표시 ─────────────────────────────────────────────────
# HW 감지·프리셋 선택은 즉시 끝나는 사전점검이라 번호를 매기지 않는다.
# 번호가 붙는 건 실제 작업 10단계 — 총계는 프리셋이 정해진 뒤(api면 빌드·다운로드
# 생략) plan_steps 가 확정한다. 안 도는 단계를 총계에 넣으면 진행률이 거짓말이 된다.
TOTAL_STEPS=10
STEP_N=0
step() { STEP_N=$((STEP_N+1)); printf '\n\033[1;36m▸ [%d/%d] %s\033[0m\n' "$STEP_N" "$TOTAL_STEPS" "$*"; }
plan_steps() {  # 실제로 돌 단계만 센다
  TOTAL_STEPS=6                                   # 포트·config·SSL·API키·관리자키·PATH등록
  [ "$NO_SERVICE" = 0 ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))                        # systemd 등록·기동
  if [ "$CONFIG_ONLY" != 1 ]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 3))              # 의존성·venv·Qdrant
    [ "$P_runtime" != "api" ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))                    # llama.cpp 빌드
    [ "$NO_MODEL" = 0 ] && [ "$P_runtime" != "api" ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))  # 모델
  fi
  return 0
}

# ── 재실행 복구 표식 ──────────────────────────────────────────
# 무거운 단계(pip·빌드·모델 다운로드)는 완료 표식을 남긴다. 중간에 죽어도
# 재실행하면 끝난 단계를 건너뛰고 죽은 지점부터 이어간다.
# 표식만 믿지 않고 산출물 실재까지 함께 본다(표식은 있는데 .venv를 지운 경우).
STATE_DIR="$ROOT/.install-state"
stage_done() { [ "$FRESH" = 0 ] && [ -f "$STATE_DIR/$1.done" ]; }
stage_mark() { [ "$DRY" = 1 ] || { mkdir -p "$STATE_DIR"; : > "$STATE_DIR/$1.done"; }; }
[ "$FRESH" = 1 ] && [ "$DRY" = 0 ] && rm -rf "$STATE_DIR"

# 실패 시 어디서 죽었는지·어떻게 이어가는지 알려준다 (curl|sh 사용자는 로그가 전부다)
on_err() {
  local rc=$?
  printf '\n\033[1;31m✗ [%d/%d] 단계에서 실패 (exit %d)\033[0m\n' "$STEP_N" "$TOTAL_STEPS" "$rc"
  printf '  같은 명령을 다시 실행하면 완료된 단계는 건너뛰고 이어서 진행합니다:\n'
  printf '    cd %s && ./install.sh --instance %s\n' "$ROOT" "$INSTANCE"
  printf '  처음부터 다시 하려면 --fresh 를 붙이세요.\n'
  exit "$rc"
}
trap on_err ERR

confirm() {  # confirm "질문"  → yes면 0
  [ "$YES" = 1 ] && return 0
  read -rp "$1 [Y/n] " a; [ -z "$a" ] || [ "$a" = "y" ] || [ "$a" = "Y" ]
}

# ── 0. 기본 도구 확인 ─────────────────────────────────────────
command -v python3 >/dev/null || die "python3가 필요합니다"
PYV=$(python3 -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])')
[ "$PYV" -ge 310 ] || die "Python 3.10+ 필요 (현재 $(python3 -V))"

echo
printf '\033[1m ai-console 설치 — 인스턴스: %s / 위치: %s\033[0m\n' "$INSTANCE" "$ROOT"

# ── 1. HW 감지 ────────────────────────────────────────────────
say "하드웨어 감지"
VRAM_MB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  # nvidia-smi 는 드라이버/라이브러리 불일치 시 에러 문구를 **stdout 으로** 뱉고 exit 0 을
  # 낸다("Failed to initialize NVML: Driver/library version mismatch"). 그래서 2>/dev/null
  # 도 `|| echo 0` 도 걸리지 않고, 비숫자가 그대로 $(( )) 에 들어가 set -u 로 즉사했다.
  # 값이 숫자인지 직접 확인하는 것 말고는 방법이 없다.
  # 게다가 종료코드는 0이 아니라 18을 낸다 — set -o pipefail 이 이를 전파해 치환이
  # 실패하므로 `|| true` 로 파이프라인 전체를 감싼다.
  _v=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '\r' | sed 's/^ *//; s/ *$//' || true)
  case "${_v:-}" in
    ''|*[!0-9]*) warn "nvidia-smi 응답이 숫자가 아닙니다 — GPU 없음으로 진행${_v:+ ($_v)}" ;;
    *)           VRAM_MB="$_v" ;;
  esac
fi
# free -g 는 내림이라 16GB 머신이 15로 보고된다(OS 예약분). min_ram_gb 필터가
# 명목 용량 기준이므로 MB 로 읽어 올림한다 — 안 그러면 실제 16GB 가 16GB 요건에 떨어진다.
RAM_MB=$(free -m | awk '/^Mem/{print $2}')
RAM_GB=$(( (RAM_MB + 1023) / 1024 ))
CORES=$(nproc)
VRAM_GB=$(( VRAM_MB / 1024 ))
echo "  GPU VRAM : ${VRAM_GB}GB / 램: ${RAM_GB}GB / 코어: ${CORES}"

if [ -z "$TIER" ]; then
  if   [ "$VRAM_GB" -ge 24 ]; then TIER="gpu-24gb-plus"
  elif [ "$VRAM_GB" -ge 12 ]; then TIER="gpu-16gb"
  elif [ "$VRAM_GB" -ge 6  ]; then TIER="gpu-8gb"
  else TIER="cpu-only"; fi
fi
ok "티어 판정: $TIER"
# 램 요건은 models.yaml 의 min_ram_gb 로 옮겼다 — 여기서 경고로 때우지 않고
# parse_preset.py 가 후보에서 아예 제외한다(프리셋을 늘려도 이 파일을 안 고쳐도 된다).

# ── 2. 프리셋 선택 (models.yaml) ──────────────────────────────
# 파서는 scripts/parse_preset.py (stdlib 전용 — venv 前 시스템 python으로 실행 가능,
# server_overrides 중첩까지 올바로 처리).
say "모델 프리셋 선택"
# --vram/--ram 을 넘기면 min_vram_gb/min_ram_gb 미달 후보가 걸러진 목록이 온다.
PRESET_INFO=$(python3 scripts/parse_preset.py "$TIER" "$PRESET" --vram "$VRAM_GB" --ram "$RAM_GB")

if echo "$PRESET_INFO" | head -1 | grep -q CANDIDATES; then
  COUNT=$(echo "$PRESET_INFO" | tail -n +2 | wc -l)
  # 필터가 전부 걸러낸 경우. 막지 않으면 아래 번호 입력 루프가 무한히 돌고,
  # --yes 면 빈 PRESET 이 export 단계까지 내려가 죽는다.
  if [ "$COUNT" -eq 0 ]; then
    warn "이 하드웨어(VRAM ${VRAM_GB}GB / 램 ${RAM_GB}GB)에 맞는 '$TIER' 프리셋이 없습니다."
    echo "  models.yaml 의 min_vram_gb / min_ram_gb 요건을 모두 밑돕니다. 선택지:"
    echo "   1) 램/GPU 증설 후 재실행"
    echo "   2) 요건을 아는 상태에서 직접 지정: ./install.sh --preset <이름>"
    echo "   3) 외부 API 로 우회(HW 무관): ./install.sh --tier api"
    echo "  전체 프리셋 목록: models.yaml"
    die "적합한 프리셋 없음"
  fi
  echo "$PRESET_INFO" | tail -n +2 | while IFS='|' read -r i name mv mr vram note; do
    printf "  %s) %-24s %s %s\n" "$i" "$name" "$vram" "${note:+— $note}"
    printf "     %s\n" "요건: VRAM ${mv}GB+ / 램 ${mr}GB+"
  done
  DEFAULT=$(echo "$PRESET_INFO" | sed -n '2p' | cut -d'|' -f2)
  if [ "$YES" = 1 ]; then PRESET="$DEFAULT"
  else
    # 범위 밖·비숫자 입력이면 재질문 (빈 PRESET으로 진행하면 export 단계에서 사망)
    while :; do
      read -rp "번호 선택 [1]: " n; n=${n:-1}
      case "$n" in *[!0-9]*|'') warn "1~${COUNT} 사이 숫자를 입력하세요"; continue;; esac
      [ "$n" -ge 1 ] && [ "$n" -le "$COUNT" ] && break
      warn "1~${COUNT} 사이 숫자를 입력하세요"
    done
    PRESET=$(echo "$PRESET_INFO" | tail -n +2 | sed -n "${n}p" | cut -d'|' -f2)
  fi
  PRESET_INFO=$(python3 scripts/parse_preset.py "$TIER" "$PRESET")
fi

# 값에 공백이 있어도 안전하게 export (eval+word-split 회피 — extra_args 멀티토큰 대비)
while IFS='=' read -r _k _v; do
  [ -n "$_k" ] && export "P_$_k=$_v"
done <<< "$(echo "$PRESET_INFO" | tail -n +2)"
ok "선택: $PRESET (handler=$P_handler, runtime=$P_runtime)"
[ "$P_runtime" = "api" ] && warn "API 프리셋 — 모델 다운로드·빌드 생략, config에 API 키만 채우면 됩니다"

plan_steps   # 프리셋(runtime)이 정해졌으니 남은 단계 수 확정

# ── 2.5 포트 배정 (C안 — 인스턴스 N대 공존) ───────────────────
# server·Qdrant·llama-server가 인스턴스마다 겹치므로 레지스트리 기반으로 배정한다.
# 재실행 시 같은 포트를 돌려주고, 기존 설치본은 config.ini의 포트를 그대로 승계한다.
step "포트 배정 (인스턴스: $INSTANCE)"
PORT_INFO=$(python3 scripts/alloc_ports.py "$INSTANCE" "$ROOT") || die "포트 배정 실패"
while IFS='=' read -r _k _v; do
  [ -n "$_k" ] && export "$_k=$_v"
done <<< "$PORT_INFO"
echo "  콘솔 https://localhost:${PORT_server} · Qdrant ${PORT_qdrant}(gRPC ${PORT_qdrant_grpc}) · llama-server ${PORT_llama_server}/${PORT_llama_server_translation}/${PORT_llama_server_gemma}"
if [ "${INSTANCE_REUSED}" = 1 ]; then
  ok "기존 인스턴스 '$INSTANCE' — 포트 유지"
else
  ok "인스턴스 '$INSTANCE' 등록 (총 ${INSTANCE_COUNT}대) — ${INSTANCE_REGISTRY}"
fi

# --config-only: 기설치 환경 전제 — §3~7(의존성·venv·빌드·Qdrant·다운로드) 생략, config 생성 직행
if [ "$CONFIG_ONLY" = 1 ]; then
  [ -x .venv/bin/python ] || die "--config-only는 기설치 환경 전용 (.venv 없음 — 전체 install.sh 먼저 실행)"
fi
if [ "$CONFIG_ONLY" != 1 ]; then

# ── 3. 시스템 의존성 ──────────────────────────────────────────
step "시스템 의존성 확인 (git·cmake·컴파일러·openssl·python3-venv)"
MISSING=""
for c in git cmake g++ make curl openssl; do command -v "$c" >/dev/null || MISSING="$MISSING $c"; done
# Ubuntu 서버 기본 이미지엔 python3-venv(ensurepip)·python3-dev(Python.h)가 없다 —
# 전자는 venv 생성 즉사, 후자는 annoy/hnswlib C 확장 빌드 실패 (둘 다 T1 실측)
python3 -c 'import ensurepip' 2>/dev/null || MISSING="$MISSING python3-venv"
python3 -c 'import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"],"Python.h")) else 1)' \
  2>/dev/null || MISSING="$MISSING python3-dev"
# GPU 감지 시 CUDA 툴킷(nvcc) 필수 — llama.cpp/llama-cpp-python CUDA 빌드가 없으면
# 수십 분 진행 후 "CUDA Toolkit not found"로 사망 (T2 실측). apt로 못 채우므로 즉시 안내 후 중단.
if [ "$VRAM_GB" -gt 0 ] && ! command -v nvcc >/dev/null; then
  warn "GPU 감지(VRAM ${VRAM_GB}GB)됐지만 CUDA 툴킷(nvcc)이 없습니다 — CUDA 빌드 불가"
  warn "Ubuntu 기본 'nvidia-cuda-toolkit'(12.0)은 최신 GPU(sm_120) 미지원 — NVIDIA 공식 저장소로:"
  echo "    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "    sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install -y cuda-toolkit-12-9"
  echo "    echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc && source ~/.bashrc"
  die "CUDA 툴킷 설치 후 재실행하세요 (CPU로만 쓰려면 --tier cpu-only)"
fi
if [ -n "$MISSING" ]; then
  if command -v apt-get >/dev/null; then PKG="sudo apt-get install -y build-essential cmake git curl openssl python3-venv python3-dev"
  elif command -v dnf >/dev/null;   then PKG="sudo dnf install -y gcc-c++ cmake git curl openssl make python3-devel"
  else die "누락:$MISSING — 수동 설치 필요"; fi
  warn "누락:$MISSING"
  confirm "패키지 매니저로 설치할까요? ($PKG)" && run bash -c "$PKG" || die "의존성 미해결"
else ok "시스템 의존성 충족"; fi

# ── 4. 파이썬 venv ────────────────────────────────────────────
step "파이썬 환경 (.venv)"
run python3 -m venv .venv
PIP=".venv/bin/pip"
if stage_done pydeps && [ -x .venv/bin/python ]; then
  ok "파이썬 의존성 — 이전 실행에서 완료(건너뜀, --fresh 로 재설치)"
else
  run "$PIP" install -q --upgrade pip
  # CPU 전용 환경: torch를 CPU 인덱스로 먼저 설치 (기본 PyPI는 CUDA 빌드 ~5GB를 끌어옴).
  # GPU 환경은 requirements.deploy.txt의 기본 CUDA torch를 그대로 쓴다.
  # 대형 설치 3종은 -q 없이 — 진행바·Building wheel 표시가 없으면 어느 단계에서
  # 멈췄는지/죽었는지 구분 불가 (T1에서 무한대기로 오인)
  if [ "$VRAM_GB" -eq 0 ]; then
    say "torch CPU 빌드 설치 (GPU 없음 — CUDA 빌드 회피, 디스크 ~5GB 절약)"
    run "$PIP" install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  fi
  say "파이썬 의존성 설치 (requirements.deploy.txt — annoy/hnswlib 소스 빌드 포함, 수 분)"
  run "$PIP" install -r requirements.deploy.txt
  run "$PIP" install -q "huggingface_hub[cli]"
  if [ "$P_runtime" = "inprocess" ]; then
    say "llama-cpp-python 설치 (in-process 프리셋 — 소스 빌드, 수 분 소요)"
    if [ "$VRAM_GB" -gt 0 ]; then
      run bash -c "CMAKE_ARGS='-DGGML_CUDA=on' $PIP install llama-cpp-python"
    else
      run "$PIP" install llama-cpp-python
    fi
  fi
  stage_mark pydeps
fi
ok "venv 준비 완료"

# ── 5. llama.cpp 빌드 (server 런타임 + 임베딩/도구용) ─────────
if [ "$P_runtime" != "api" ]; then
  step "llama.cpp 빌드"
  if [ ! -d llama.cpp ]; then
    TAG=$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['tag_name'])" 2>/dev/null || true)
    run git clone --depth 1 ${TAG:+--branch "$TAG"} https://github.com/ggml-org/llama.cpp llama.cpp
    [ -n "$TAG" ] && ok "릴리스 태그 고정: $TAG"
  fi
  CUDA_FLAG=""; [ "$VRAM_GB" -gt 0 ] && CUDA_FLAG="-DGGML_CUDA=ON"
  run cmake -S llama.cpp -B llama.cpp/build $CUDA_FLAG -DCMAKE_BUILD_TYPE=Release
  run cmake --build llama.cpp/build --target llama-server -j "$CORES"
  ok "llama-server: llama.cpp/build/bin/llama-server"
fi

# ── 6. Qdrant ─────────────────────────────────────────────────
step "Qdrant 벡터DB"
if [ ! -x qdrant/qdrant ]; then
  QTAG=$(curl -fsSL https://api.github.com/repos/qdrant/qdrant/releases/latest 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['tag_name'])" 2>/dev/null || echo "v1.12.1")
  run mkdir -p qdrant
  run bash -c "curl -fsSL https://github.com/qdrant/qdrant/releases/download/${QTAG}/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz -C qdrant"
  ok "Qdrant $QTAG → ./qdrant/qdrant"
else ok "Qdrant 이미 존재"; fi

# ── 7. 모델 다운로드 ──────────────────────────────────────────
HF=".venv/bin/huggingface-cli"
if [ "$NO_MODEL" = 1 ] || [ "$P_runtime" = "api" ]; then
  warn "모델 다운로드 생략"
else
  step "모델 다운로드 (임베딩 2.3GB + LLM)"
  # huggingface-cli 는 이어받기·기존파일 스킵을 자체 처리하므로 표식은 '둘 다 끝남'만 표시
  if stage_done model && [ -d "$P_emb_dir" ] && [ -d "$P_local_dir" ]; then
    ok "모델 — 이전 실행에서 완료(건너뜀, --fresh 로 재확인)"
  else
    # onnx/ 는 받지 않는다 — 리포 전체를 받으면 4.3GB 인데 그중 2.2GB 가 onnx 런타임용이고,
    # 이 콘솔은 FlagEmbedding(BGEM3FlagModel=torch) 경로만 쓴다(코드 내 onnx 참조 0건).
    # imgs/·long.jpg 는 README 삽화. 제외 후 실사용분은 pytorch_model.bin + 토크나이저 +
    # colbert_linear.pt/sparse_linear.pt(2-way RRF 에 필수) ≈ 2.3GB. [2026-07-28 실측]
    say "임베딩 모델 (BGE-M3, 2.3GB — onnx 제외)"
    run "$HF" download "$P_emb_repo" --local-dir "$P_emb_dir" --exclude "onnx/*" "imgs/*" "*.jpg"
    say "LLM 다운로드: $P_repo (${P_include:-전체})"
    run "$HF" download "$P_repo" ${P_include:+--include "$P_include"} --local-dir "$P_local_dir"
    stage_mark model
  fi
fi

fi  # CONFIG_ONLY

# ── 8. config.ini 생성 ────────────────────────────────────────
# 티어를 config 생성 블록에 전달 (cpu-only 에서 n_gpu_layers 를 눌러야 한다)
export INSTALL_TIER="$TIER"
step "config.ini 생성"
if [ -f config.ini ] && ! confirm "config.ini가 이미 있습니다. 덮어쓸까요?"; then
  warn "config.ini 유지"
else
  run .venv/bin/python - "$PRESET" << 'PYEOF'
import configparser, glob, os, sys

preset = sys.argv[1]
# install.sh가 export한 P_* 변수 재사용
env = {k[2:]: v for k, v in os.environ.items() if k.startswith('P_')}

cfg = configparser.ConfigParser()
cfg.read('config.ini.template', encoding='utf-8')

# 기존 config.ini의 수동 설정(API 키 등)은 보존 — 프리셋 전환이 키를 지우면 안 됨 (Gate2 리뷰)
old = configparser.ConfigParser()
if os.path.exists('config.ini'):
    old.read('config.ini', encoding='utf-8')
    for _sec in ('openai', 'bedrock'):
        if old.has_section(_sec):
            if not cfg.has_section(_sec):
                cfg.add_section(_sec)
            for _k, _v in old[_sec].items():
                if _v and not _v.upper().startswith('YOUR'):
                    cfg[_sec][_k] = _v

cfg['model']['model'] = env['handler']
cfg['database']['use_db_mode'] = 'False'
cfg['embedding']['use_bge_mode'] = 'True'
cfg['embedding']['bge_model_path'] = os.path.abspath(env.get('emb_dir', 'models/bge-m3'))
cfg['server']['auth_mode'] = 'file'   # DB 없는 로컬 설치 — api_keys/*.key 파일 인증

# 인스턴스별 포트 (scripts/alloc_ports.py 가 배정 → install.sh 가 PORT_* 로 export).
# C안에서 한 호스트에 콘솔 N대가 공존하므로 고정 포트를 쓰면 두 번째부터 기동 실패한다.
# qdrant_grpc 는 대응 섹션이 없어 제외 — Qdrant 기동 시 환경변수로만 쓴다.
for _role in ('server', 'qdrant', 'llama_server',
              'llama_server_translation', 'llama_server_gemma'):
    _p = os.environ.get('PORT_' + _role)
    if _p:
        if not cfg.has_section(_role):
            cfg.add_section(_role)
        cfg[_role]['port'] = _p

# 모델 경로 (GGUF 글롭 해석 — 가장 큰 파일)
if env.get('local_dir') and env.get('path_key'):
    ggufs = sorted(glob.glob(os.path.join(env['local_dir'], '**/*.gguf'), recursive=True),
                   key=os.path.getsize, reverse=True)
    if ggufs:
        cfg['paths'][env['path_key']] = os.path.abspath(ggufs[0])

# llama-server 섹션 (섹션이 템플릿에 없을 수 있으므로 setdefault 로 안전 접근)
for _sec in ['llama_server', 'llama_server_translation', 'llama_server_gemma']:
    if not cfg.has_section(_sec):
        cfg.add_section(_sec)
if env.get('runtime') == 'server' and env.get('server_section'):
    s = env['server_section']
    cfg[s]['use_server_mode'] = 'True'
    cfg[s]['binary_path'] = os.path.abspath('llama.cpp/build/bin/llama-server')
    for pair in filter(None, env.get('overrides', '').split(',')):
        k, _, v = pair.partition(':')
        cfg[s][k.strip()] = v.strip()
    if env.get('extra_args'):
        cfg[s]['extra_args'] = env['extra_args']
    # cpu-only 는 GPU 오프로딩을 끈다. config.ini.template 의 n_gpu_layers=99 가 그대로
    # 나가면 CUDA 빌드된 llama-server + GPU 부재 조합에서 기동이 깨진다.
    # (CPU 전용 빌드에서는 무시되지만, 티어와 모순된 값을 config 에 남기지 않는다.)
    # [2026-08-19 VM 스모크 실측 — cpu-only 설치가 -ngl 99 로 spawn 되던 것]
    if os.environ.get('INSTALL_TIER') == 'cpu-only':
        cfg[s]['n_gpu_layers'] = '0'
    # 다른 서버 섹션은 비활성 (한 번에 하나)
    for other in ['llama_server', 'llama_server_translation', 'llama_server_gemma']:
        if other != s:
            cfg[other]['use_server_mode'] = 'False'
else:
    for other in ['llama_server', 'llama_server_translation', 'llama_server_gemma']:
        cfg[other]['use_server_mode'] = 'False'

with open('config.ini', 'w', encoding='utf-8') as f:
    cfg.write(f)
print("config.ini 생성 완료")

# 장착 상태 재적용 — 재설치가 카트리지 mount를 깨지 않도록 (cartridge_mount 공용 함수)
import json as _json, os as _os
if _os.path.isfile('cartridges/.mounted.json'):
    try:
        import cartridge_mount
        _st = _json.load(open('cartridges/.mounted.json', encoding='utf-8'))
        # 프리셋이 바뀌면 핸들러도 바뀐다 — 저장된 배선을 그대로 쓰면 새 핸들러가 읽는
        # 키는 순정인 채 status 만 "장착됨"이 된다. 현재 핸들러 기준으로 재계산.
        _st2 = cartridge_mount.rewire_state_for_handler(_st)
        cartridge_mount.apply_state_to_config(_st2)
        _warns = list(_st2.get('warnings') or [])
        if _st2.get('prompts') != _st.get('prompts') or _st2.get('handler') != _st.get('handler'):
            _persist = {k: v for k, v in _st2.items()
                        if k not in ('_stale_prompt_keys', 'warnings')}
            _json.dump(_persist, open('cartridges/.mounted.json', 'w', encoding='utf-8'),
                       ensure_ascii=False, indent=2)
        for _w in _warns:
            print(f"  ⚠ {_w}")
        print(f"장착 재적용: {_st2.get('cartridge')} ({len(_st2.get('prompts', {}))} 슬롯)")
    except Exception as _e:
        print(f"⚠ 장착 재적용 실패({_e}) — ./aibotctl cartridge status 확인 후 재mount")
PYEOF
fi

# ── 9. 자체서명 SSL ───────────────────────────────────────────
step "자체서명 SSL 인증서"
if [ ! -f ssl/selfsigned.key ]; then
  run mkdir -p ssl
  run openssl req -x509 -newkey rsa:2048 -nodes -keyout ssl/selfsigned.key \
      -out ssl/selfsigned.crt -days 3650 -subj "/CN=localhost" 2>/dev/null
  ok "ssl/selfsigned.crt 생성"
else ok "인증서 이미 존재"; fi

# ── 10. 기본 API 키 발급 (file 인증 모드) ────────────────────
step "기본 API 키 (api_keys/default.key)"
if [ ! -f api_keys/default.key ]; then
  run mkdir -p api_keys
  if [ "$DRY" = 1 ]; then echo "  [dry-run] openssl rand -hex 32 > api_keys/default.key"
  else openssl rand -hex 32 > api_keys/default.key; chmod 600 api_keys/default.key; fi
  ok "발급 완료"
else ok "키 이미 존재"; fi

# ── 10-b. 관리자 키 발급 (관리 평면 — security-review.md S-7) ─
# 사용자 Bearer 키(api_keys/*.key)와 **다른 평면**이다. 관리 API(키 발급·구독 목록)만 쓴다.
# 예전에는 run.sh 가 '0'32자 더미를 넘기고 서버가 검증을 통째로 끄고 있었다(S-2).
# 검증을 되살리는 이상 기본값이 있으면 안 된다 — 설치 때 랜덤 생성하고 fallback 을 없앤다.
step "관리자 키 (api_keys/admin.key)"
if [ ! -f api_keys/admin.key ]; then
  run mkdir -p api_keys
  if [ "$DRY" = 1 ]; then echo "  [dry-run] openssl rand -hex 32 > api_keys/admin.key"
  else openssl rand -hex 32 > api_keys/admin.key; chmod 600 api_keys/admin.key; fi
  ok "발급 완료 — 관리 API 전용(사용자 Bearer 키로는 쓰이지 않음)"
else ok "키 이미 존재"; fi

# ── 11. aibotctl PATH 등록 ────────────────────────────────────
# 지금까지 리포 루트에서 ./aibotctl 로만 불렀다(T2에서 command not found를 겪은 지점).
# 인스턴스가 여럿이면 이름이 겹치므로 default 만 'aibotctl', 나머지는 'aibotctl-<인스턴스>'.
step "aibotctl PATH 등록"
if [ "$INSTANCE" = "default" ]; then BIN_NAME="aibotctl"; else BIN_NAME="aibotctl-$INSTANCE"; fi
BIN_DIR=""
for d in "$HOME/.local/bin" "/usr/local/bin"; do
  if [ -d "$d" ] && [ -w "$d" ]; then BIN_DIR="$d"; break; fi
done
# ~/.local/bin 은 없으면 만든다 (sudo 불필요 — 우선 경로)
[ -z "$BIN_DIR" ] && { run mkdir -p "$HOME/.local/bin"; BIN_DIR="$HOME/.local/bin"; }
if [ "$DRY" = 1 ]; then
  echo "  [dry-run] ln -sfn $ROOT/aibotctl $BIN_DIR/$BIN_NAME"
else
  ln -sfn "$ROOT/aibotctl" "$BIN_DIR/$BIN_NAME"
  ok "$BIN_DIR/$BIN_NAME → $ROOT/aibotctl"
  case ":$PATH:" in
    *":$BIN_DIR:"*) : ;;
    *) warn "$BIN_DIR 가 PATH에 없습니다 — 셸 설정에 추가하세요:"
       echo "    echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc && source ~/.bashrc" ;;
  esac
fi

# ── 12. systemd 서비스 등록·기동 ──────────────────────────────
# 이 단계 이전까지 install.sh 는 "다음 단계: Qdrant 기동 → 콘솔 기동" 안내로 끝났다.
# 목표가 ".sh 한 번이면 그 뒤로 엔드포인트로 부른다" 이므로 여기서 실제로 띄운다.
#
# 유닛 2개 (인스턴스별 이름 — C안에서 인스턴스는 서로 다른 클론이라 systemd 템플릿(@)보다
# 평범한 이름이 안전하다):
#   ai-console-qdrant-<인스턴스>.service   지식 저장소
#   ai-console-<인스턴스>.service          콘솔 (Qdrant 뒤에 기동)
#
# 루트 ai-agent.service 는 원본 운영 서버 유물이다 — 전용 계정·고정 경로 ·
# conda 절대경로 · Requires=mariadb.service(우린 DB-off). 그걸 쓰지 않고 여기서 생성한다.
step "systemd 서비스 등록·기동"
SVC_CONSOLE="ai-console-${INSTANCE}"
SVC_QDRANT="ai-console-qdrant-${INSTANCE}"
SVC_MODE=""
SVC_CTL=""
SVC_JOURNAL_FLAG=""
if [ "$NO_SERVICE" = 1 ]; then
  warn "--no-service 지정 — 서비스 등록을 건너뜁니다"
elif [ "$DRY" = 1 ]; then
  echo "  [dry-run] systemd 유닛 생성: ${SVC_QDRANT}.service · ${SVC_CONSOLE}.service"
elif ! command -v systemctl >/dev/null 2>&1; then
  warn "systemctl 없음 — 서비스 등록 생략 (수동 기동: ./run.sh start)"
else
  # 시스템 유닛(부팅 시 자동 기동)을 우선한다. 권한이 없으면 사용자 유닛으로 물러선다.
  if [ "$(id -u)" = "0" ]; then          SVC_MODE=system; SUDO=""
  elif sudo -n true 2>/dev/null; then    SVC_MODE=system; SUDO="sudo"
  else                                   SVC_MODE=user;   SUDO=""
  fi

  if [ "$SVC_MODE" = system ]; then
    SVC_WANTED_BY=multi-user.target
    UNIT_DIR=/etc/systemd/system
    SCTL="$SUDO systemctl"
    UNIT_USER="User=$(id -un)
Group=$(id -gn)"
  else
    SVC_WANTED_BY=default.target
    UNIT_DIR="$HOME/.config/systemd/user"
    SCTL="systemctl --user"
    UNIT_USER=""
    mkdir -p "$UNIT_DIR"
  fi

  _write_unit() {   # $1=경로  (stdin=내용)
    if [ "$SVC_MODE" = system ]; then $SUDO tee "$1" >/dev/null; else cat > "$1"; fi
  }

  _write_unit "$UNIT_DIR/${SVC_QDRANT}.service" <<UNIT
[Unit]
Description=ai-console Qdrant (instance: ${INSTANCE})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
${UNIT_USER}
WorkingDirectory=${ROOT}
Environment=QDRANT__SERVICE__HTTP_PORT=${PORT_qdrant}
Environment=QDRANT__SERVICE__GRPC_PORT=${PORT_qdrant_grpc}
ExecStart=${ROOT}/qdrant/qdrant
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=${SVC_WANTED_BY}
UNIT

  # 콘솔은 Qdrant 뒤에. Requires 가 아니라 Wants — Qdrant 가 잠깐 죽어도 콘솔까지
  # 같이 내리지 않는다(질의는 실패해도 프로세스는 살아 재연결하는 편이 운영에 낫다).
  # MariaDB 의존은 넣지 않는다 — 이 배포는 use_db_mode=False 로 돈다.
  _write_unit "$UNIT_DIR/${SVC_CONSOLE}.service" <<UNIT
[Unit]
Description=ai-console (instance: ${INSTANCE}, preset: ${PRESET})
After=network-online.target ${SVC_QDRANT}.service
Wants=network-online.target ${SVC_QDRANT}.service

[Service]
Type=simple
${UNIT_USER}
WorkingDirectory=${ROOT}
ExecStart=${ROOT}/.venv/bin/python ${ROOT}/qa_llm.py
Restart=always
RestartSec=5
# 모델 로딩이 길다(26B MoE 는 수 분) — 기동 판정을 넉넉히
TimeoutStartSec=900
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=${SVC_WANTED_BY}
UNIT

  $SCTL daemon-reload
  # enable 은 부팅 기동만 설정하고, 기동은 **항상 restart** 로 한다.
  # `enable --now` 는 이미 돌고 있으면 아무것도 안 한다 — install.sh 를 다시 돌려
  # config.ini(프리셋·포트)를 바꿔도 낡은 프로세스가 그대로 살아 있게 되고,
  # 바로 뒤 준비 폴링이 그 낡은 프로세스에 "0초 만에 준비 완료" 를 찍는다(실제 발생).
  $SCTL enable "${SVC_QDRANT}.service"  >/dev/null 2>&1 || true
  $SCTL enable "${SVC_CONSOLE}.service" >/dev/null 2>&1 || true
  $SCTL restart "${SVC_QDRANT}.service"
  $SCTL restart "${SVC_CONSOLE}.service"
  SVC_CTL="$SCTL"
  [ "$SVC_MODE" = user ] && SVC_JOURNAL_FLAG="--user"
  ok "유닛 등록·기동: ${SVC_QDRANT} · ${SVC_CONSOLE} (${SVC_MODE} 모드)"
  if [ "$SVC_MODE" = user ]; then
    warn "사용자 유닛이라 로그아웃하면 멈춥니다 — 부팅 지속을 원하면:"
    echo "    sudo loginctl enable-linger $(id -un)"
    echo "    (또는 sudo 로 install.sh 재실행하면 시스템 유닛으로 등록됩니다)"
  fi

  # 준비 대기 — uvicorn 은 모델 로딩이 끝나야 리스닝을 시작한다.
  # 즉 포트가 열렸다 = 핸들러 로드 완료. GET / 로 한 번 더 확인한다(무인증).
  printf '  모델 로딩 대기'
  _READY=0
  _i=0
  while [ $_i -lt 180 ]; do
    if curl -sk --max-time 3 "https://127.0.0.1:${PORT_server}/" >/dev/null 2>&1; then _READY=1; break; fi
    printf '.'; sleep 5; _i=$((_i+5))
  done
  echo
  if [ "$_READY" = 1 ]; then
    ok "엔드포인트 준비 완료 (${_i}초)"
  else
    warn "180초 안에 응답 없음 — 로딩이 더 길 수 있습니다. 확인:"
    echo "    $SCTL status ${SVC_CONSOLE} --no-pager"
    echo "    journalctl $([ "$SVC_MODE" = user ] && echo --user) -u ${SVC_CONSOLE} -f"
  fi
fi

# 운영 안내는 실제 등록 결과에 맞춰 구성한다 (미등록/dry-run 에 systemd 명령을 안내하면 거짓말이 된다)
if [ -n "$SVC_MODE" ]; then
  SVC_OPS="서비스 운영 (systemd · ${SVC_MODE} 모드):
  상태  : ${SVC_CTL} status ${SVC_CONSOLE} --no-pager
  로그  : journalctl ${SVC_JOURNAL_FLAG} -u ${SVC_CONSOLE} -f
  재시작: ${SVC_CTL} restart ${SVC_CONSOLE}
  ※ systemd 로 도는 중에는 ./run.sh 로 중복 기동하지 마세요(포트 충돌)."
else
  SVC_OPS="수동 운영 (systemd 미등록):
  1) Qdrant : QDRANT__SERVICE__HTTP_PORT=${PORT_qdrant} QDRANT__SERVICE__GRPC_PORT=${PORT_qdrant_grpc} ./qdrant/qdrant &
  2) 콘솔   : ./run.sh start        (상태 ./run.sh status · 재시작 ./run.sh restart)
  ※ 부팅 자동 기동을 원하면 --no-service 없이 다시 실행하세요."
fi
if [ -n "$SVC_MODE" ]; then SVC_RESTART_CMD="${SVC_CTL} restart ${SVC_CONSOLE}"; else SVC_RESTART_CMD="./run.sh restart"; fi

# ── 완료 ──────────────────────────────────────────────────────
echo
ok "설치 완료 — 인스턴스: $INSTANCE / 티어: $TIER / 프리셋: $PRESET"
cat << NEXT

엔드포인트:
  URL   : https://<이 호스트>:${PORT_server}/agent/chat/completions   ← OpenAI 호환(카트리지 적용)
  토큰  : ${ROOT}/api_keys/default.key   (값 확인: cat 그 파일)
  확인  : curl -sk -X POST https://localhost:${PORT_server}/agent/chat/completions \\
            -H "Authorization: Bearer \$(cat ${ROOT}/api_keys/default.key)" \\
            -H 'Content-Type: application/json' \\
            -d '{"model":"any","messages":[{"role":"user","content":"안녕"}]}'
  스키마: https://localhost:${PORT_server}/docs      상태: https://localhost:${PORT_server}/

${SVC_OPS}

카트리지:
  ${BIN_NAME} cartridge validate cartridges/<이름>    ← 검증
  ${BIN_NAME} cartridge mount    cartridges/<이름>    ← 장착(배선+지식적재+런타임 반영)
  ${BIN_NAME} cartridge status                       ← 확인 (반영 실패 시에만 ${SVC_RESTART_CMD})
  되돌리기: ${BIN_NAME} cartridge purge

  * 외부 서비스 연동(openai-full 등)은 docs/api-integration.md §1-b 참조.
    [server] agent_rag=true 여야 카트리지 지식이 적용됩니다(신규 설치 기본값).
  * Qdrant는 이 인스턴스 포트(${PORT_qdrant}) 전용입니다. 컬렉션명은 [qdrant] collection
    (기본 bge) — 포트를 공유하면서 이름까지 같으면 지식을 덮어씁니다. docs/multi-instance.md
  * API 프리셋이면 config.ini의 [openai]/[bedrock] 키를 채우세요.
NEXT

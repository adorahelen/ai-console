# install.sh 경로 해설 — 옵션별로 정확히 무슨 일이 일어나나

> 🇬🇧 English: [install-paths.en.md](install-paths.en.md)

> `./install.sh`는 **① GPU 유무(VRAM 감지)** 와 **② 프리셋의 runtime**(inprocess / server / api)
> 두 축으로 갈라진다. 이 문서는 각 조합에서 어떤 단계가 실행/생략되는지, 시간이 어디에 쓰이는지 정리한다.
> 시간 수치는 [실측] = 2026-07-21 T1(VM 4vCPU)·T2(7500F+5070 Ti) 기록, [추정] = 미실측.

## 결정 트리

```
nvidia-smi로 VRAM 감지
├─ VRAM = 0  →  CPU 경로   (torch CPU 휠 · CUDA 컴파일 전부 생략 · nvcc 불필요)
└─ VRAM > 0  →  GPU 경로   (nvcc 필수 · CUDA 커널 컴파일 2회 · torch 기본(CUDA) 휠)
프리셋 runtime
├─ inprocess (llama31-8b)      → llama-cpp-python 빌드 + llama.cpp 빌드 + 모델 다운로드
├─ server    (gpt-oss, gemma)  → llama.cpp 빌드만 + 모델 다운로드 (llama-cpp-python 생략)
└─ api       (openai, claude)  → 빌드·모델 다운로드 전부 생략 (venv + config + 키만)
```

## A. CPU 경로 — `--tier cpu-only` (VM/GPU 없는 서버)

| 단계 | 하는 일 | 시간 |
|---|---|---|
| 의존성 검사 | git·cmake·g++·openssl + **python3-venv·python3-dev** (없으면 apt 안내) | 수 초 |
| venv + torch **CPU** 휠 | 기본 PyPI CUDA 빌드(~5GB) 대신 CPU 인덱스(~수백 MB) | [실측] 수 분 |
| requirements | annoy·hnswlib **C 확장 소스 빌드** 포함 | [실측] 수 분 |
| llama-cpp-python | C++ 빌드 — **CPU 백엔드만, CUDA 커널(.cu) 0개** | [실측] 5~10분 |
| llama.cpp (llama-server) | cmake CPU 빌드 (server 런타임 아니어도 도구용으로 빌드) | [추정] 5~10분 |
| Qdrant | 바이너리 다운로드 (컴파일 없음) | 수 초 |
| 모델 다운로드 | BGE-M3 ~4.5GB + llama31-8b Q4 ~5GB | [실측] 회선 1~2MB/s 기준 1시간± |
| config·SSL·키 | 템플릿→config.ini, 자체서명, api_keys/default.key | 수 초 |

**총 소요**: 빌드 ~15분 + 다운로드(회선 의존). CPU 온도: 빌드 중 전 코어 100%가 정상.

## B. GPU 경로 — GPU 자동 감지 (데스크톱/GPU 서버)

**전제: CUDA 툴킷(nvcc)** — 없으면 시작 시점에 감지·중단하고 NVIDIA 공식 저장소 설치 명령을 안내한다
(Ubuntu 기본 `nvidia-cuda-toolkit`=12.0은 최신 GPU sm_120 미지원 — `cuda-toolkit-12-9` 사용).

A와 다른 점만:

| 단계 | 차이 | 시간 |
|---|---|---|
| torch | CPU 인덱스 생략 → requirements가 기본(CUDA) 휠 설치 (~3GB) | [추정] 다운로드 수 분 |
| llama-cpp-python (inprocess만) | `GGML_CUDA=on` — **CUDA 커널 수백 개 컴파일** (fattn·mmq 템플릿 인스턴스 전 양자화 포맷) | [실측 진행 중] 10~30분, CPU 93°C까지 관측(스로틀 정상) |
| llama.cpp (llama-server) | 역시 `GGML_CUDA=ON` — **두 번째 CUDA 컴파일** | [추정] 10~20분 |
| 기동 시 | 모델이 GPU 탑재 (llama31-8b: "풀 GPU 모드·35레이어" 로그) | — |

**총 소요**: 빌드 ~30~60분 + 다운로드. 온도가 걱정되면 `CMAKE_BUILD_PARALLEL_LEVEL=3 ./install.sh …`로
병렬도를 줄일 수 있다 (시간 ~2배, 해당 휠 빌드는 처음부터 다시).

[실측 T2 · 2026-07-21, 7500F+5070 Ti] llama31-8b 프리셋 GPU 경로 **설치 완주 확인** —
CUDA 빌드 2회 포함 전 단계 통과. 다운로드는 12~14MB/s 회선에서 BGE-M3 3분 + 8B GGUF(4.92GB) 5분41초.

> 참고: inprocess 프리셋인데도 llama.cpp를 빌드하는 것은 도구용(주석 기준)이다. CUDA 컴파일이
> 2회가 되는 비용이 있어, inprocess 경로에서 생략 가능한지는 검증 후 개선 후보.

## C. API 경로 — `--tier api` (openai-api / claude-api)

빌드 0회, 모델 다운로드 0바이트. venv(+requirements) → config.ini → SSL·키 생성이 전부.
[실측] dry-run 기준 수 분. 설치 후 config.ini의 `[openai] api_key` 또는 `[bedrock]` 자격만 채우면 된다.

## 프리셋별 다운로드 용량

임베딩(BGE-M3)은 전 프리셋 공통 **2.3GB**다 — `install.sh`가 `onnx/`·삽화를 제외하고 받는다
(리포 전체는 4.3GB지만 그중 2.2GB가 onnx 런타임용이고 이 콘솔은 torch 경로만 쓴다). [2026-07-28 실측]
기설치 상태면 재사용되므로 두 번째 프리셋부터는 LLM 몫만 늘어난다.

아래 LLM 용량은 전부 HF `content-length` 실측이다. [2026-07-31 확인]

| 프리셋 | LLM | +임베딩 2.3GB | 비고 |
|---|---|---|---|
| gemma4-e2b-q4 | 2.62GB (UD-Q4_K_XL) | **~4.9GB** | cpu-only 최저사양 폴백 |
| gemma4-e4b-q4 | 4.22GB (UD-Q4_K_XL) | **~6.5GB** | cpu-only · gpu-8gb 기본 |
| gemma4-12b-q4 | 6.72GB (UD-Q4_K_XL) | **~9.0GB** | gpu-16gb 기본 |
| llama31-8b-q4 | 4.92GB (Q4_K_M) | **~7.2GB** | `--preset` 전용 |
| gpt-oss-20b | 13.79GB (F16) | **~16.1GB** | `--preset` 전용 |
| gemma4-26b-moe-offload / -full | 16.95GB (UD-Q4_K_M) | **~19.3GB** | 두 프리셋이 같은 파일을 공유 |
| openai-api / claude-api | 0 | **0** | 다운로드·빌드 전부 생략 |

## 모델 교체 방법 + gemma4 계열 가이드 (2026-07-21)

**교체 방법 3가지** — 어느 경우든 핸들러가 레지스트리에 있는 계열만 가능
(gpt/gpt-oss/gemma/llama/claude/qwen — mistral 등 타계열은 핸들러 부재로 불가):

1. `./install.sh --preset <이름> --yes` — 프리셋 전환. 빌드는 재사용, 모델만 추가 다운로드, config 재생성
2. models.yaml에 프리셋 추가 후 ① — 새 GGUF 변형 테스트 경로
3. 수동: config.ini `[model] model=` + `[paths]` 모델 경로 + 해당 `llama_server_*` 섹션 편집

**Gemma 4 공식 라인업** (2026-03-31 출시 · Apache 2.0 · 전 모델 공식 QAT GGUF → llama.cpp 호환.
출처: ai.google.dev/gemma/docs/core, blog.google, unsloth):

| 모델 | Q4 메모리(공식) | 16GB GPU 판정 | 프리셋 (`--preset` 전용, 미실측) |
|---|---|---|---|
| E2B (활성 2B) | 2.9GB | 되지만 콘솔 품질 미지수 | `gemma4-e2b-q4` |
| E4B (활성 4B) | 4.5GB | 풀GPU, 가볍게 시작 | `gemma4-e4b-q4` |
| 12B (Unified 멀티모달) | 6.7GB | 풀GPU+KV 여유 — 스윗스팟 후보 | `gemma4-12b-q4` |
| 26B A4B (MoE) | 14.4GB | 오프로드 필요 — 원본 검증 모델 | `gemma4-26b-moe-offload` (티어 제안) |
| 31B (Dense) | 17.5GB | **비추** — VRAM 초과, dense라 오프로드 시 급감. 24GB+ 전용 | 미등록 |

권장 스윕 순서(16GB): `e4b` → `12b` → `26b-moe-offload`. 각 단계에서 실측할 것:
VRAM(`nvidia-smi`)·tok/s·**chat template 토큰 누출 여부** (새 변형이 handler_gemma 템플릿과
완전 호환인지는 실측 전 단정 금지 — 누출 시 `handler_gemma.build_agent_prompt` 검토).
실측치 확보 → 프리셋 note의 [미실측] 해소 → 티어 목록 승격 판단.

**비추천**: 31B(16GB에선), qwen(크래시 분기는 3차 리뷰에서 제거됐으나 응답 품질 미검증 —
티어 비제안 유지), 핸들러 없는 타계열.

## 설치된 모델 인벤토리 확인

```bash
# GGUF 전체 검색 + 용량 (100MB 초과만 — llama.cpp의 ggml-vocab-* 테스트 파일 제외)
find ~ -name "*.gguf" -size +100M -exec du -h {} + 2>/dev/null

# HF 캐시 (BGE-M3 임베딩, safetensors 등 GGUF 외 모델·데이터셋)
hf cache scan
```

프리셋별 모델은 `models/<프리셋>/` 디렉토리로 분리되므로, 위 결과에서 어떤 프리셋이 받아져 있는지 바로 보인다.

## 재실행·전환·청소

- **재실행 안전**: pip는 설치분 스킵, 모델은 프리셋별 디렉토리 분리, config.ini는 `--yes`면 덮어씀.
  중간에 죽었으면 원인(의존성 등) 해소 후 **같은 명령 재실행**이 정답.
- **프리셋 전환**: `./install.sh --preset <다른것> --yes` — 빌드는 재사용, 새 모델만 받는다.
- **청소**: 시스템 전역 잔재는 apt 패키지(+CUDA 툴킷)뿐. 나머지는 `rm -rf ~/ai-console ~/.cache/huggingface ~/.cache/pip`.

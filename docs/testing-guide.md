# 동적 검증 가이드 — T1: VM 스모크 테스트

> 🇬🇧 English: [testing-guide.en.md](testing-guide.en.md)

> 정적 분석(컴파일·AST·모듈실행 + 코드리뷰 5차, 총 20여건 수정)으로 못 잡는 것을 실행으로 검증한다.
> 대상 환경: **Ubuntu 24.04 VM · 4vCPU · RAM 16GB · 디스크 50GB · GPU 없음**
> 소요: 설치 ~20~30분(회선 의존) + 테스트 ~30분

## 개념 — 무엇을 검증하나

정적으로 검증 불가능해서 여기서 처음 확인되는 6가지:

| # | 영역 | 정적에서 못 본 이유 |
|:--:|---|---|
| V1 | install.sh 실빌드 (pip·llama.cpp·Qdrant·HF 다운로드) | dry-run까지만 했음 |
| V2 | 모델 실로딩 (llama-cpp-python + Llama-3.1-8B Q4 GGUF) | 모델 파일 없었음 |
| V3 | file 인증 통합 (uvicorn 경유 전체 스택) | 함수 단위만 실행했음 |
| V4 | 위저드 LLM 3종 실응답 (초안·테스트챗·qna변환 **품질**) | LLM 필요 |
| V5 | llama chat template 정확성 (agent 경로 신규 작성분) | 응답을 봐야 판단 |
| V6 | RAG 전체 파이프라인 (임베딩→Qdrant 적재→검색 히트) | Qdrant·BGE 필요 |

T2(GPU·성능)는 5070 Ti 데스크톱에서 별도 — [맨 아래](#t2--gpu-검증-데스크톱-예고) 참조.

---

## T1-0. 준비

```bash
# VM에서 (Ubuntu 24.04 클라우드 이미지는 git·빌드도구가 이미 들어 있다)
git clone https://github.com/adorahelen/ai-console-public.git ai-console && cd ai-console
```

> 리포가 private 인 동안에는 clone 이 안 된다. multipass VM 이면 호스트에서 반입한다:
> ```bash
> git archive --format=tar HEAD | gzip > ~/ai-console.tar.gz   # 워킹트리 오염 없이 HEAD 만
> multipass transfer ~/ai-console.tar.gz <vm>:/home/ubuntu/ai-console.tar.gz
> multipass exec <vm> -- bash -lc 'mkdir -p ~/ai-console && tar xzf ~/ai-console.tar.gz -C ~/ai-console'
> ```
> ⚠️ snap 격리 때문에 multipass 는 `/tmp` 하위 임의 경로를 읽지 못한다 — 반입 파일은 홈 디렉토리를 경유할 것.

기대: `ai-console` 디렉토리에 `install.sh`, `models.yaml`, `qa_llm.py` 존재.

## T1-1. 설치 (V1)

```bash
./install.sh --preset llama31-8b-q4 --yes     # 아래 기대 출력을 낸 조합
```

> ⚠️ **cpu-only 티어의 기본값은 이제 `gemma4-e4b-q4`다**(Gemma-first 전환). 아래 블록은
> `llama31-8b-q4`(`runtime=inprocess`)를 실측한 기록이라 프리셋을 고정해 재현한다.
> **기본 경로(`./install.sh --tier cpu-only --yes`)는 아직 한 번도 돌려본 적이 없고**,
> `runtime=server`라 `llama-cpp-python` 설치 줄이 없고 다운로드 리포도 다르다 —
> 그 차이를 확인하는 것이 V1에서 새로 해야 할 일이다.

단계별 기대 출력 (2026-07-21 VM 실측 반영):

```
✓ 티어 판정: cpu-only
✓ 선택: llama31-8b-q4 (handler=llama, runtime=inprocess)
▸ 시스템 의존성 확인 (git·cmake·컴파일러·openssl·python3-venv)
  (multipass Ubuntu 24.04 클라우드 이미지는 그냥 통과한다 — 2026-08-19 실측.
   최소 서버 이미지면 python3-venv·python3-dev 누락 감지 → sudo apt install 안내가 정상)
▸ 파이썬 환경 (.venv)
▸ torch CPU 빌드 설치 (GPU 없음 — CUDA 빌드 회피, 디스크 ~5GB 절약)   ← ★ 이 줄 필수
▸ 파이썬 의존성 설치 (requirements.deploy.txt — annoy/hnswlib 소스 빌드 포함, 수 분)
▸ llama-cpp-python 설치 (in-process 프리셋 — 소스 빌드, 5~10분: CPU 전 코어 100%가 정상)
✓ 릴리스 태그 고정: bXXXXX
✓ Qdrant vX.Y.Z → ./qdrant/qdrant
▸ LLM 다운로드: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF (*Q4_K_M*)   ← ~5GB
  (+ BGE-M3 ~2.3GB — onnx 제외(`24b01e6`). 디스크 실측 2.2G. 회선 1~2MB/s면 30분 안팎, 네트워크 대기가 정상)
✓ 설치 완료 — 티어: cpu-only / 프리셋: llama31-8b-q4
```

⚠️ pip 대형 설치 3종(torch·requirements·llama-cpp-python)은 진행바가 실시간으로 보여야 한다
(74e284f에서 `-q` 제거). 진행바 없이 조용하면 구버전 스크립트 — `git pull` 후 재실행.
진행 확인은 별도 터미널에서 `watch -n2 'du -sh ~/ai-console/.venv'`.

**성공 기준 (전부 충족):**

```bash
ls config.ini api_keys/default.key ssl/selfsigned.key        # 3개 존재
.venv/bin/python -c "import llama_cpp; print('ok')"          # ok
ls models/llama31-8b/*.gguf models/bge-m3/pytorch_model.bin  # 모델 존재
grep "auth_mode" config.ini                                  # auth_mode = file
df -h . | tail -1                                            # 사용량 ~15~18GB 내외 (8B Q4 ~5GB)
```

## T1-2. 기동 (V2)

> ⚠️ **2026-08-19 정정 — 기본 경로에서는 이 절의 수동 기동이 필요 없다.**
> `runtime=server` 프리셋(Gemma-first 전환 이후 전 티어 기본)에서는 `install.sh`가
> systemd 유닛 두 개(`ai-console-<인스턴스>`·`ai-console-qdrant-<인스턴스>`)를 등록·기동하고
> 엔드포인트 준비 확인까지 마친 뒤 종료한다. 아래 수동 절차는 `--no-service` 설치이거나
> in-process 프리셋(`llama31-8b-q4` 등)일 때만 쓴다.

설치 직후의 정상 상태 — 이걸 확인하는 것이 V2다:

```bash
systemctl is-active ai-console-default ai-console-qdrant-default  # active / active
pgrep -af "qdrant|llama-server|qa_llm"                            # 3개 프로세스
curl -s http://127.0.0.1:8183/health                              # {"status":"ok"}  ← llama-server
curl -s http://127.0.0.1:6333/collections                         # {"result":...,"status":"ok"}
```

수동 기동(in-process 프리셋 · `--no-service` 설치):

```bash
./qdrant/qdrant > qdrant.log 2>&1 &
sleep 3 && curl -s http://127.0.0.1:6333/collections
.venv/bin/python qa_llm.py 2>&1 | tee console.log
```

**성공 기준:**
```bash
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/ \
  -H "Authorization: Bearer $(cat api_keys/default.key)"
# → 200 (API 상태 JSON). 401이면 인증(V3) 문제, 연결 거부면 기동 실패
```

> ⚠️ BGE-M3 로딩이 CPU에서 1~2분 걸린다 — 그 사이의 연결 거부는 실패가 아니다.



```bash
KEY=$(cat api_keys/default.key)
# 유효 키 → 200
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/api/wizard/prompt-draft \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"role":"테스트"}'
# 무효 키 → 401
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/api/wizard/prompt-draft \
  -H "Authorization: Bearer WRONG" -H "Content-Type: application/json" -d '{"role":"x"}'
```

성공 기준: 첫 호출 **200** (또는 503=핸들러 미로드, 이땐 V2 실패), 둘째 **401**.

## T1-4. 위저드 LLM 3종 (V4·V5)

브라우저 `https://localhost:8443/wizard` → 자체서명 경고 통과 → API 키 붙여넣기.
또는 CLI로:

```bash
KEY=$(cat api_keys/default.key)
# ① 프롬프트 초안
curl -sk https://localhost:8443/api/wizard/prompt-draft -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"role":"사내 인사규정 안내","audience":"전 직원","tone":"정중","rules":"규정에 없으면 모른다고 답한다","needs_action":false}' | python3 -m json.tool
# ③ 문서 → qna 변환
curl -sk https://localhost:8443/api/wizard/knowledge-convert -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"연차는 입사 1년차 15일이다. 재택근무는 주 2회까지 허용된다.","count":3}' | python3 -m json.tool
```

**성공 기준 (품질 판단 — 눈으로):**
- ① `draft`에 역할·규칙이 반영된 한국어 시스템 프롬프트. 깨진 토큰(`<|start_header_id|>`·`<|eot_id|>` 등)이 응답에 새어나오면 **V5(chat template) 실패**.
- ③ `items`가 유효한 배열, question/answer가 문서 내용 기반. 빈 배열이면 LLM이 JSON 형식을 못 지킨 것 → 메타 프롬프트 조정 필요.

> ⚠️ **미검증 위험 2순위**: llama chat template(`handler_llama.build_agent_prompt`)의 agent 경로는 이번 리팩토링에서 작성 — 응답 품질/토큰 누출은 여기서 처음 드러난다. CPU 8B라 초안 1건에 **2~5분** 걸릴 수 있으니 타임아웃 아님.

## T1-5. 카트리지 장착 + RAG 적재 (V6)

> ⚠️ **2026-08-19 정정 — `/api/wizard/cartridge-save` 는 존재하지 않는다.**
> 실제 위저드 API 는 `prompt-draft` · `knowledge-convert` · `cartridge-mount` 세 개뿐이다.
> 이전 판의 `cartridge-save` 절차를 그대로 따라 하면 404 가 난다.
> 카트리지 저장·적재의 정본 경로는 **`aibotctl cartridge`** CLI 다.

동봉된 예시 카트리지(`cartridges/console-guide` — 콘솔 자신의 사용법, 지식 8건)로 검증한다:

```bash
./aibotctl cartridge validate cartridges/console-guide   # 스키마 검증 (읽기 전용)
./aibotctl cartridge mount    cartridges/console-guide   # 배선 + 임베딩 → Qdrant 적재
./aibotctl cartridge status                              # 장착 상태
```

**성공 기준:**

```bash
# 적재 전에는 컬렉션 자체가 없다 → mount 후 points 가 지식 건수와 일치
curl -s http://127.0.0.1:6333/collections/bge \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])"   # → 8
./aibotctl cartridge status    # 장착: console-guide · 지식 8건
```

> mount 는 **재시작 없이 런타임 반영**된다(핸들러 리로드). 되돌리기는 `./aibotctl cartridge purge`.
> ⚠️ 카트리지의 `model.recommended` 가 현재 핸들러와 다르면 경고가 뜬다 — mount 는 `[model]` 을 바꾸지 않는다.

자기 문서로 카트리지를 만들려면 `knowledge-convert` 로 QnA 를 뽑아 `cartridges/<이름>/knowledge/*.yaml`
로 저장한 뒤 같은 CLI 를 태운다. 위저드 UI(3단계)도 내부적으로 `cartridge-mount` 를 호출한다.



RAG 검색만 순수하게 확인하는 경로는 `/api/search`(요청 `SearchRequest{query, search_type, top_k}`).
생성·권한게이트 없이 **적재한 지식이 실제로 검색되는지**를 본다.

```bash
KEY=$(cat api_keys/default.key)
curl -sk https://localhost:8443/api/search -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"연차 며칠이야?","search_type":"QNA","top_k":5}' | python3 -m json.tool
```

성공 기준: 결과 context/source에 **T1-5에서 넣은 "연차 15일" 문서가 등장** → 임베딩→적재→검색이 한 줄로 동작.

> ✅ 이 경로는 재검증에서 **활성 모델(DEFAULT_MODEL) 라우팅으로 수정됨** — 원래 코드는 `gpt-oss` 핸들러를 하드코딩해 qwen 설치에서 503이었다(핸들러-키 결합, Phase 1 문자열 탈색이 못 잡은 것).
> ⚠️ 생성까지 포함한 종단(`/api/query/stream`, `/api/ai/chats`)은 요청 스키마·권한 모델이 복잡 — 스모크에선 검색(retrieval)까지만 확인하고, 생성 품질은 **T1-4 위저드 test-chat**으로 이미 봤다. 종단 대화는 `/docs`에서 스키마 확인 후 별도.

---

## 증상 → 원인 → 조치

| 증상 | 유력 원인 | 조치 |
|---|---|---|
| `ensurepip is not available` (venv 생성 실패) | python3-venv 미설치 — **T1 실측** (신버전은 의존성 검사가 사전 감지) | `sudo apt install -y python3.12-venv && rm -rf .venv` 후 재실행 |
| `fatal error: Python.h` (annoy/hnswlib 빌드 실패) | python3-dev 미설치 — **T1 실측** (동일) | `sudo apt install -y python3.12-dev` 후 재실행 (pip는 설치분 스킵) |
| `Could not find nvcc … CUDA Toolkit not found` (llama-cpp-python/llama.cpp 빌드) | GPU 머신인데 CUDA 툴킷 미설치 — **T2 실측** (신버전은 시작 시 감지·안내) | NVIDIA 공식 repo로 `cuda-toolkit-12-9` 설치 (Ubuntu 기본 `nvidia-cuda-toolkit`=12.0은 sm_120 미지원) 후 재실행 |
| `import llama_cpp` 실패 | 소스 빌드 실패(컴파일러·메모리) | `pip install llama-cpp-python -v`로 로그 확인. RAM 부족이면 swap 추가 |
| 기동 시 `[paths]/[prompts] 필수키 누락` | config.ini.template 구버전 | `git pull` 후 `cp config.ini.template config.ini` 재생성 |
| 전 API 401 | `auth_mode` 미설정 or 키 불일치 | `grep auth_mode config.ini` = file, `cat api_keys/default.key` 대조 |
| 위저드 503 | llm_handler 미로드(모델 로딩 실패) | console.log에서 RAG/모델 초기화 에러 확인 |
| 초안에 `<|start_header_id|>`류 누출 | llama chat template 오류(V5) | `handler_llama.build_agent_prompt` 수정 — 이 가이드로 회귀 |
| qna 변환 502 `qna 파싱 실패` | 소형 LLM이 JSON 문법 위반(문자열 내 따옴표 등) — **T2 실측** (8B, 긴 영어 문서에서 재현) | 신버전은 JSONL+줄단위 관대 파싱(깨진 줄만 폐기). 구버전이면 `git pull`. 여전히 0건이면 count 낮추고 문서 분할 |
| Qdrant 연결 거부 | 미기동 | `./qdrant/qdrant &` 후 `curl :6333/collections` |
| BGE 로딩에서 멈춘 듯 | CPU라 정상(1~2분) | `top`으로 python CPU 사용 확인, 기다림 |
| `/api/search` 503 | 구버전(gpt-oss 하드코딩) | `git pull` — DEFAULT_MODEL 라우팅 반영본 필요 |
| `/api/query/stream` 403 | query_type이 permissions에 없음 | file-auth는 general/qwen/gemma 등 허용(신버전). 구버전이면 `git pull` |

> 🔎 (5차 리뷰 반영) gpt-oss 하드코딩은 전부 해소 — v1 `/agent/chat/completions`·ticket_memo 모두 활성 모델 라우팅. **v1도 이제 Bearer 인증 필수**(completions2와 동일). 검증 플러그인은 template 기본 비활성(`plugin_module` 빈값) — secops 장착 시에만 활성.

---

## 완료 체크리스트

- [ ] V1 install.sh 완주 (config·키·SSL·모델 생성)
- [ ] V2 콘솔 기동, 포트 8443 응답
- [ ] V3 file 인증 200/401
- [ ] V4 프롬프트 초안·qna 변환 유효 응답
- [ ] V5 chat template 토큰 누출 없음
- [ ] V6 카트리지 저장 → 적재 → 컬렉션 벡터 증가
- [ ] 종단 대화에서 RAG 지식 등장

**하나라도 실패하면**: 증상표로 1차 대응 → 안 되면 `console.log`·`qdrant.log` 첨부해 이슈 기록. 이 가이드의 ⚠️ 항목이 실패 후보 1순위다.

---

## T1 실행 결과 — cpu-only 기본 경로 ✅ V1~V6 완주 (2026-08-19 실측)

`./install.sh --tier cpu-only --yes` 로 **처음으로 기본 경로를 완주**했다.
multipass Ubuntu 24.04 VM · 4vCPU · RAM 16GB · 디스크 50GB · GPU 없음.
1차(2026-07-21)는 `--preset llama31-8b-q4`(in-process)였으므로, 프리셋·런타임·핸들러가 전부 다른 경로다.

| 축 | 결과 |
|---|---|
| V1 설치 | 완주. 프리셋 `gemma4-e4b-q4` 자동 선택 · GGUF 4.0G · BGE-M3 2.2G · 디스크 18G · `auth_mode=file` |
| V2 기동 | systemd 2유닛 active, 3프로세스 기동, `/health` ok, 콘솔 200 |
| V3 인증 | 유효키 200 / 무효키 401 |
| V4 위저드 | `prompt-draft` 규칙 반영 · `knowledge-convert` 3건 전부 문서 근거 |
| V5 템플릿 | **특수토큰 누출 0건** (응답 3건 전수 검사). `handler_gemma` 경로 정상 |
| V6 RAG | validate → mount 6.5초 · Qdrant `points=8` · 검색 인용 5건, 1순위가 정답 문서 |

**응답 시간 (4vCPU·GPU 없음)**: 짧은 챗 63초 · 초안 68초 · qna 변환 78초 · RAG 질의 125초.
느린 것이지 실패가 아니다 — 이 티어에서는 `--max-time` 을 넉넉히 줄 것.

**이 실행에서 고친 것**
- `install.sh` — cpu-only 인데 llama-server 가 `-ngl 99` 로 spawn 되던 것(템플릿 기본값 유출) → 티어가 cpu-only 면 `n_gpu_layers=0` 강제
- `cartridges/console-guide` — `model.recommended` 가 `llama31-8b-q4` 라 기본 설치와 어긋나 있던 것 → `gemma4-e4b-q4`
- `cartridge_mount.py` — `recommended` 가 **있기만 하면** 무조건 경고하던 것(프리셋명↔핸들러명을 비교하라고 사용자에게 떠넘김) → models.yaml 로 프리셋→핸들러를 풀어 실제로 다를 때만 경고. 없는 프리셋이면 별도 문구
- 이 문서 — T1-2 수동 기동 전제, T1-5 의 없는 엔드포인트, BGE 용량, 의존성 안내, clone URL

**남은 미검증**: 이 티어의 tok/s 와 런타임 메모리는 여전히 미측정(`models.yaml` 의 [미실측] 라벨 유지).

---

## T2 — GPU 검증 (데스크톱)

### 1차 — llama31-8b 경량 GPU 검증: ✅ V1~V6 완주 (2026-07-21 실측)

7500F + 5070 Ti 16GB + 램 30Gi, Ubuntu 24.04, CUDA 툴킷 12.9(NVIDIA repo):

- **V1**: CUDA 빌드 2회(llama-cpp-python + llama-server) 포함 설치 완주. 빌드 중 CPU 93°C 관측(스로틀 정상 범위)
- **V2**: 풀 GPU 모드 로드 — **GPU 레이어 35개, VRAM 7,770MiB** (8B Q4 + ctx 16384 KV + BGE-M3)
- **V3**: file 인증 200/401 정상
- **V4·V5**: 위저드 종단(초안→테스트챗→qna 5건→저장→적재) 성공, chat template 토큰 누출 없음.
  실측 결함 2건 발견·수정: qna 추출 JSON 전면 파싱 실패(따옴표 인용 문서) → **Q:/A:/AL: 라인 태그 포맷 + 3단 폴백 파서**로 교체
- **V6**: `/api/search`로 적재 지식 5건 전부 검색, 정답 문서 유사도 1.0 상위 랭크 — **BGE-M3 GPU(sm_120) 실연산 정상**

### 2차 — 주력 스펙 실측 (남은 본론)

- `./install.sh --preset gpt-oss-20b --yes` → **llama-server 별도 프로세스 런타임** (1차의 in-process와 다른 경로)
- 실측 대상: VRAM 사용량, **tok/s**, 동시 2슬롯(n_parallel) 동작
- **KV 캐시 재사용 A/B**: gpt-oss spawn엔 `--cache-reuse`가 없음(gemma만 승계) — 0 vs 256으로 같은 RAG 질의 반복해 **TTFT 비교** → 기본값 결정
- 이후 여유 시 gemma-26b-moe-offload — `--cpu-moe` 실측으로 models.yaml의 [미실측] 라벨 해소 (램 30Gi 경계 주의)

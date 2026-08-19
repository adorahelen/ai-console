# 다중 인스턴스 운영 — 배포 형태 C안 (2026-07-27)

> 🇬🇧 English: [multi-instance.en.md](multi-instance.en.md)

> **결정: C안 — 고객사(도메인)별 인스턴스.** 한 호스트에 콘솔 N대를 세우고,
> 각 인스턴스가 자기 클론·자기 config·자기 포트·자기 Qdrant·자기 카트리지를 갖는다.
> 장착 설계의 **A안(도메인당 콘솔 1대·전역 장착)** 을 여러 대로 확장한 것 — 카트리지 모델은 그대로다.
> **B안(docker) 기각**: GPU 패스스루·모델 볼륨·llama.cpp CUDA 빌드를 다시 검증해야 하는 비용 대비
> 이득이 얇다(설치기가 이미 베어메탈 원샷이다). `docker/`는 자산으로 남겨둔다.
> **관련**: [install-paths.md](install-paths.md)

---

## 인스턴스의 정의

**인스턴스 = 클론 디렉토리 1개.** 이게 전제이고, 나머지는 전부 여기서 따라온다.

| 자산 | 격리 방식 |
|---|---|
| 코드·config.ini | 클론 디렉토리별 |
| 지식(Qdrant) | **인스턴스마다 자기 Qdrant 프로세스** (아래 참조) |
| 카트리지 장착 상태 | `cartridges/.mounted.json` (클론별) |
| API 키 | `api_keys/*.key` (클론별) |
| 모델 가중치 | 클론별 `models/` — 공유하려면 심링크 |

한 디렉토리를 두 인스턴스 이름으로 등록하는 것은 **거부된다**
([scripts/alloc_ports.py](../scripts/alloc_ports.py)) — 포트·카트리지·Qdrant가 뒤엉킨다.

### Qdrant 공유 — 컬렉션명 파라미터화 (2026-07-28)

지식 컬렉션명은 **`[qdrant] collection`** (기본 `bge`)이 단일 소스다 — `config_utils.qdrant_collection()`.
코드 리터럴 21곳(임베딩·RAG 2종·프롬프트 API·동기화·mount/purge·utils)을 전부 이 헬퍼로 외부화했다.

- **기본값 유지 시 동작 불변** — 키가 없으면 종전 `bge` 그대로(기존 컬렉션과 호환, T2 실기동 확인).
- **Qdrant 1대 공유**: 인스턴스마다 `collection` 을 달리 주면 지식이 컬렉션 단위로 격리된다.
  단, 컬렉션명이 같으면 종전처럼 **서로의 지식을 덮어쓴다** — 공유 시 이름 배정은 운영자 책임.
- 인스턴스별 Qdrant 분리(포트 배정)는 여전히 기본 권장 — 장애 격리까지 얻는다.

---

## 포트 배정

`install.sh`가 [scripts/alloc_ports.py](../scripts/alloc_ports.py)로 6종을 배정한다.

| 역할 | 기본 | 비고 |
|---|---|---|
| `server` | 8443 | 콘솔 HTTPS |
| `qdrant` | 6333 | Qdrant HTTP |
| `qdrant_grpc` | 6334 | **config에 섹션 없음** — 기동 환경변수로만 사용 |
| `llama_server` | 8181 | gpt-oss 계열 |
| `llama_server_translation` | 8182 | 번역 전용 |
| `llama_server_gemma` | 8183 | gemma 프리셋 |

**레지스트리**: `~/.ai-console/instances.tsv` (`AI_CONSOLE_HOME`으로 이동 가능)

```
acme	/srv/console-acme	server=8444,qdrant=6335,qdrant_grpc=6336,llama_server=8184,...
```

레지스트리를 두는 이유는 LISTEN 검사만으로는 **중지된** 형제 인스턴스의 포트를
비어 있다고 오판하기 때문이다. 배정 규칙:

1. 이미 등록된 인스턴스면 **기록된 포트를 그대로** 돌려준다(재설치가 포트를 흔들지 않음).
2. 처음이지만 `config.ini`가 있으면 **그 포트를 승계**한다(레지스트리 도입 前 설치본).
   `qdrant_grpc`는 섹션이 없으므로 Qdrant 규약대로 HTTP+1로 승계.
3. 그 외에는 기본값부터 올라가며 형제 예약분·실제 LISTEN을 모두 피한다.

인스턴스를 없앨 때는 **레지스트리에서 해당 줄을 지운다** — 포트가 회수된다.

---

## 설치

```bash
# 첫 인스턴스 (default)
curl -fsSL https://raw.githubusercontent.com/adorahelen/ai-console-public/main/install.sh | sh

# 고객사별 인스턴스 — ~/ai-console-acme 로 클론되고 포트가 자동 회피된다
curl -fsSL .../install.sh | sh -s -- --instance acme --yes

# 클론 위치를 직접 정하려면
AI_CONSOLE_DIR=/srv/console-acme  curl -fsSL .../install.sh | sh -s -- --instance acme
```

중간에 실패하면 **같은 명령을 다시 실행**한다. 무거운 단계(pip·모델 다운로드)는
`.install-state/` 표식으로 건너뛴다. 처음부터 다시 하려면 `--fresh`.

---

## 기동

인스턴스 디렉토리에서, **자기 포트로** 띄운다.

```bash
cd /srv/console-acme

# 1) Qdrant — HTTP·gRPC 둘 다 지정할 것
QDRANT__SERVICE__HTTP_PORT=6335 QDRANT__SERVICE__GRPC_PORT=6336 ./qdrant/qdrant &

# 2) 콘솔
.venv/bin/python qa_llm.py        # 또는 ./run.sh
```

- Qdrant의 `storage/`·`snapshots/`는 **cwd 기준**으로 생기므로 클론 디렉토리에서
  띄우기만 하면 데이터가 자연히 분리된다. (실측 확인)
- `QDRANT__SERVICE__GRPC_PORT`를 빼먹으면 두 번째 인스턴스가
  `Error while starting gRPC server: transport error`를 남긴다. 이 콘솔은 gRPC를 쓰지
  않아 죽지는 않지만, 로그가 더러워지므로 지정한다. (실측 확인)

---

## 카트리지 관리 (인스턴스별)

`install.sh`가 `~/.local/bin`에 심링크를 건다 — `default`는 `aibotctl`,
그 외는 **`aibotctl-<인스턴스>`**.

```bash
aibotctl-acme cartridge list                       # 목록 + 장착 표시
aibotctl-acme cartridge validate cartridges/acme   # 장착 전 검증
aibotctl-acme cartridge mount    cartridges/acme   # 배선 + 지식 적재 + 런타임 반영(재시작 불필요)
# 반영이 확인 안 되면 폴백: ./run.sh restart
aibotctl-acme cartridge status

aibotctl-acme cartridge unmount                    # 이 카트리지가 올린 지식만 제거
aibotctl-acme cartridge purge                      # 컬렉션째 삭제 → 클린 콘솔
```

`aibotctl-*`은 어디서 호출해도 **자기 리포 루트로 cd 한 뒤** 실행된다.
config `[prompts]` 배선이 cwd 기준 상대경로이기 때문이다
([cartridge_mount.py:99](../cartridge_mount.py) `relpath(cart_abs, os.getcwd())`).
그래서 카트리지 경로는 **리포 루트 기준**으로 넘기거나 절대경로를 쓴다.

### unmount vs purge

| | 지우는 범위 | 콘솔 필요 |
|---|---|---|
| `unmount` | `.mounted.json`에 기록된 guid만 (REST 경유) | **필요** |
| `purge` | 컬렉션 `bge` 통째 (Qdrant 직접) | 불필요 |

새 도메인 provisioning·재현 테스트의 출발점은 `purge`다. 추적 밖 잔여물(수동 업로드,
중단된 mount, 이전 카트리지 흔적)까지 사라진다. 되돌릴 수 없다.

---

## 네트워크 노출

**콘솔은 추가 도구 없이 단독으로 HTTPS를 연다.** `install.sh` 가 자체서명 인증서를 만들고
(`ssl/selfsigned.crt`), 콘솔이 자기 포트(기본 8443)에서 직접 TLS 를 종단한다.
VPN·오버레이 네트워크·프록시는 **선택**이며, 어느 것도 설치·기동의 전제가 아니다.

| 접근 방식 | 필요한 것 | 비고 |
|---|---|---|
| 같은 호스트 | 없음 | `https://localhost:8443/` (자체서명이라 `curl -k`) |
| 사내망 직접 | 방화벽 규칙 | 아래 바인드 주의 참조 |
| 리버스 프록시 | nginx·Caddy 등 | 실인증서로 바깥쪽 TLS 종단. 안쪽은 자체서명이므로 업스트림 인증서 검증을 끈다 |
| VPN·오버레이 네트워크 | 해당 제품 | 콘솔은 관여하지 않는다 — 사설 IP 로 위 "사내망 직접"과 동일하게 붙는다 |

인스턴스가 여럿이면 노출도 **인스턴스 포트별로** 나눈다(8443·8444…). 프록시를 쓴다면
경로가 아니라 **포트/호스트로 가르는 편이 안전**하다 — 콘솔은 자기 경로가 루트라고 가정한다.

> ⚠️ **바인드 주의 (실측 2026-07-28)**: 기본 바인드가 **`0.0.0.0`** 이다(uvicorn 기본).
> 즉 콘솔은 **호스트가 가진 모든 인터페이스에서 응답한다** — 사설 IP 로 붙을 수 있는 이유이자,
> 신뢰할 수 없는 망에 물린 호스트에서는 그대로 외부 노출이 된다는 뜻이다.
> 인증은 `api_keys/*.key` 파일 기반이 전부이므로, **폐쇄망이 아니라면 방화벽으로 포트를 막고
> 리버스 프록시만 통과시킬 것.** 바인드 주소를 `127.0.0.1` 로 좁히는 config 항목은 아직 없다
> (동작 변경이라 별도 과제 — [security-review.md](../security-review.md) S-8 폐쇄망 전제와 같은 뿌리).

---

## GPU VRAM 예산 (실측 2026-07-27 · T2 RTX 5070 Ti 16,303MiB)

C안에서 가장 먼저 막히는 자원은 포트도 디스크도 아니고 **VRAM**이다. 여기 수치는 전부 이 머신에서 잰 값이다.

### 핵심 성질 — VRAM은 **로드 시점에 고정**된다

인스턴스 2대를 동시에 부하(양쪽에 생성 요청)한 6회 표집에서 총 사용량이 **14,716MiB로 완전히 평탄**했다.
가중치와 KV 캐시를 기동 시 선점하므로, **예산 계산이 결정론적이다** — 피크를 걱정할 게 아니라 기동 시 들어가느냐만 보면 된다.

### 인스턴스 1대의 실측 점유

| 구성 | 프리셋 | 실측 VRAM | 비고 |
| :-- | :-- | --: | :-- |
| 콘솔 전체(모델 + BGE-M3) | `gpt-oss-20b` (F16, `-ngl 99`, ctx 32768) | **13,238 MiB** | 별도 `llama-server` 프로세스 spawn |
| 콘솔 전체(모델 + BGE-M3) | `llama31-8b-q4` (35층, ctx 16384) | **7,884 MiB** | llama-cpp-python **인프로세스** |
| 순수 `llama-server` 단독 | 위와 같은 모델·층수·ctx | **6,826 MiB** | 콘솔 없이 서버만 |
| **BGE-M3 몫** | fp16, dense+colbert | **≈ 1.0 – 1.4 GB** | 아래 참조 |

> **BGE-M3 몫을 두 경로로 교차 확인**: ① 위 표의 차이(7,884 − 6,826 = **1,014 MiB**) ② 별도 프로세스에서 BGE-M3만
> 로드·인코드해 직접 측정(**+1,386 MiB**). 후자가 큰 것은 **자기 CUDA 컨텍스트(약 300MiB)를 따로 잡기 때문**이며,
> 콘솔 안에서는 컨텍스트를 공유하므로 **인스턴스당 한계 비용은 ≈ 1.0GB**로 보면 된다.
> (README 의 종전 표기 "1.8GB"는 이 실측에 맞춰 갱신했다.)

### 예산 공식

```
인스턴스 1대 ≈ (모델 가중치 + KV 캐시)  +  BGE-M3 ≈ 1.0GB  +  CUDA 컨텍스트 ≈ 0.3GB
GPU 총량 ≥ Σ(인스턴스별 위 값) + 여유 1GB
```

`n_ctx` · `n_parallel` 을 줄이면 KV가 줄어 가장 크게 절약된다 — KV는 컨텍스트 길이 × 슬롯 수에 비례해 유휴 시에도 전량 예약된다.

### 16GB 한 장에 몇 대가 올라가나 (실측·산출)

| 조합 | 합계 | 16,303MiB에서 | 근거 |
| :-- | --: | :-- | :-- |
| `llama31-8b-q4` × 2 | **14,716 MiB** | ✅ **실측으로 올라감** (여유 1,587MiB) | 콘솔 7,840 + 별도 llama-server 6,826, 동시 부하에서도 평탄. 2번째 서버 생성 **152.9 tok/s** |
| `gpt-oss-20b` × 2 | ≈ 26,476 MiB | ❌ 불가 | 1대 실측 13,238 × 2 |
| `gpt-oss-20b` + `llama31-8b-q4` | ≈ 20,064 MiB | ❌ 불가 | 실측 합 |
| `llama31-8b-q4` × 3 | ≈ 21,500 MiB | ❌ 불가 | 6,826×2 + 7,840 |

**결론**: 16GB 한 장에서 C안 다중 인스턴스는 **8B급 2대가 상한**이다. 20B급은 인스턴스당 GPU 한 장을 전제해야 한다.
고객사 3곳 이상을 한 호스트에 올리려면 ① 24GB+ GPU, ② 인스턴스별 GPU 분리(`CUDA_VISIBLE_DEVICES`),
③ 일부 인스턴스를 `api` 티어(OpenAI·Claude 핸들러)나 `cpu-only` 로 돌리는 세 갈래 중 하나를 골라야 한다.

> **미검증**: 인스턴스별 `CUDA_VISIBLE_DEVICES` 분리는 이 머신이 GPU 1장이라 확인하지 못했다.
> install.sh 는 현재 GPU 배정을 하지 않는다 — 다GPU 호스트에서는 수동 설정이 필요하다.

---

## 알려진 제약

- **모델 가중치가 인스턴스마다 중복**된다. 같은 프리셋을 여러 인스턴스가 쓰면
  `models/`를 공용 경로로 심링크하는 편이 낫다(읽기 전용이라 안전).
- **GPU는 공유 자원**이다. 인스턴스 N대가 각자 모델을 띄우면 VRAM이 N배 든다.
  같은 호스트에서 GPU 프리셋 다중 인스턴스는 **아래 §GPU VRAM 예산**으로 먼저 계산할 것.
- **컬렉션명은 `[qdrant] collection` 으로 파라미터화됨(2026-07-28)** — 위 참조. 이름을 달리 주면 Qdrant 1대 공유 가능.
- 인스턴스 제거는 수동이다: 프로세스 종료 → 클론 삭제 → 레지스트리 줄 삭제 →
  `~/.local/bin/aibotctl-<이름>` 심링크 삭제.

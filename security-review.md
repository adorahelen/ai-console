# Security Review — ai-console


> ℹ️ 이 점검·수정은 원본 리포(agent-console, private)에서 2026-07-27 수행된 것을 2026-07-28 이식한 것이다.
> 인용된 검증 실측은 원본 리포에서 수행됐다.
> 표준 5축 문서 ④ 보안 · **이 축의 정본은 이 문서다** · 1차 점검 **2026-07-27 수행**
> 범위: 저장소 정적 점검(인증/인가 · 시크릿 · 인젝션 · 의존성 · PII/로그 · 전송/저장).
> 동적 침투 시험(DAST)·인증 우회 실증은 **미수행** — 아래 발견은 코드 근거 기반이다.

> ✅ **출시 블로커 3건(S-1·S-2·S-3) 해소 — 2026-07-27.** 관리 평면 무인증 상태를 걷어냈다.
> 관리자 키는 `install.sh` 가 `api_keys/admin.key` 로 랜덤 발급(0600)하고, `verify_admin_key` 가
> 타이밍 안전 비교로 검증하며, **키 파일이 없으면 fail-closed**(관리 API 전면 거부)다.
> 사용자 Bearer 평면과 분리 — Bearer 키 스캔은 `admin.key` 를 건너뛴다.
>
> **실측 검증(콘솔 기동 상태)**: 관리자 키 누락 422 · 틀린 키 401 · 예전 더미 `0`×32 401 ·
> 사용자 Bearer 키를 관리자 키로 제시 401 · 올바른 키 통과(`/api/embedding-status-all` 200).
> 일반 사용자 경로 회귀 없음(`/api/search` Bearer 200), `admin.key` 를 Bearer 로 쓰면 401.
> DB 를 타는 `/api/list`·`/api/generate` 는 인증 통과 후 DB 부재로 500 — 기본 설치의 fail-closed 그대로.
>
> **2026-07-31 갱신**: S-4·S-6 수정 완료. 같은 날 공개 전 재검수에서 무인증 엔드포인트 4건(S-10~S-13)을
> 발견·수정했다 — 다만 **전부 중간 이하 등급**이다. 이 제품은 개인이 clone 해서 자기 PC·내부망에 설치하는
> 형태라, Qdrant(6333) 가 무인증으로 열려 있는 이상 콘솔 API 를 잠가도 공격자의 능력은 바뀌지 않는다.
> **이 위협 모델에서 남은 실질 통제는 S-8(네트워크 노출·바인드 주소) 하나다.** S-9(의존성 CVE 자동 대조)도 미결.

## 점검 체크리스트

| 항목 | 상태 | 결과 |
| :-- | :-- | :-- |
| 인증/인가 | ✅ 점검 | ✅ **결함 3건 전부 수정** (S-1·S-2·S-3, 2026-07-27) |
| 시크릿·자격증명 관리 | ✅ 점검 | ✅ **S-7 수정**(관리자 키 랜덤 발급) · 커밋된 시크릿 **없음** |
| 입력 검증 / 인젝션 | ✅ 점검 | ✅ **S-5 수정**(`eval`) · SQL은 파라미터 바인딩 준수 · YAML은 전량 `safe_load` |
| 의존성 취약점 | ✅ 점검 | 🟢 주요 패키지 현행 유지 (S-9 참고) |
| 로그·개인정보(PII) 노출 | ✅ 점검 | ✅ **S-6 수정**(게이트를 handler_base 공통화 + 외부 API 경로 배선, 2026-07-31) |
| 전송/저장 암호화 | ✅ 점검 | 🟡 자체서명 TLS · Qdrant 무인증 (S-8) |

---

## 발견 사항

### ✅ S-1. `/api/list` 무인증 + 평문 API 키 반환 — **수정 완료(2026-07-27)**

`aibot_restapi_auth.py:877` — 관리자 검증 블록 전체가 `'''` 로 **주석 처리**되어 있다.
그대로 이어지는 쿼리는 `SELECT s.id, s.guid, s.name, ..., s.api_key, s.acl, ...` 로 **`api_key` 평문을 그대로** 담아 반환하며,
응답에도 `"api_key_masked": False` 가 명시돼 있다.

**영향**: 포트에 닿는 누구나 `POST /api/list` 한 번으로 모든 사용자의 API 키를 획득 → 전 구독 사칭.
**조건**: DB 모드(`[database] use_db_mode=True`). DB 우회 배포에서는 해당 엔드포인트가 동작하지 않는다.
**조치(완료)**: 주석 해제로 관리자 검증 복구 + 응답 직전 `api_key` 를 뒤 4자만 남기고 마스킹(`****7890`), `api_key_masked` 플래그도 `True` 로 정정. 마스킹 로직은 단위 검증(평문·짧은 값·`None` 전부 마스킹). 엔드포인트 실행은 MariaDB 가 있어야 해 **런타임 미확인** — 인증 거부/통과는 실측 확인됨.

### ✅ S-2. `verify_admin_key()` 가 항상 `True` — **수정 완료(2026-07-27)**

```python
# aibot_restapi_auth.py:700
def verify_admin_key(admin_key: str) -> bool:
    # 관리자 키 검증 비활성화 — 모든 요청 허용
    return True
```

의도적 비활성화이며 원복 방법까지 주석에 적혀 있다(개발 편의로 추정). 호출부는 `/api/embedding-status-all`(:851).
**영향**: 관리자 전용으로 선언된 엔드포인트가 실제로는 무인증. S-1을 고쳐도 이 함수가 `True` 인 한 무의미하다.
**조치(완료)**: `api_keys/admin.key` 와 `secrets.compare_digest` 비교로 복구. **키 파일이 없으면 `False`**(fail-closed) — "키 없으면 통과"는 예전 상태로 조용히 되돌아가는 길이라 택하지 않았다. `run.sh get_admin_key` 도 파일을 읽도록 복원하고 더미 기본값 제거.

### ✅ S-3. `/api/generate` 인증 검사 없음 — **수정 완료(2026-07-27)**

`qa_llm.py:2847`. 데코레이터에 `Depends` 가 없고 함수 본문에도 인증·권한 확인이 없다.
전역 인증 미들웨어도 없다(`qa_llm.py:279` 이후 CORS·GZip 둘뿐).
**영향**: DB 모드에서 누구나 유효기간 365일짜리 API 키를 스스로 발급받아 이후 모든 Bearer 엔드포인트를 정상 이용. DB off 인 기본 설치에서는 `get_connection()` 이 실패해 500 (fail-closed).
**조치(완료)**: `GenerateApiKeyRequest` 에 `admin_key` 필드 추가 + 핸들러 진입부에서 `verify_admin_key` 검사. 실측: 필드 누락 422 · 틀린 키 401 · 사용자 Bearer 키 401.

> **참고(정상 동작하는 부분)**: 일반 사용자 인증은 건전하다 — Bearer 키 비교는 `secrets.compare_digest`(:614)로 타이밍 안전하고,
> IP ACL(CIDR)·Pydantic 입력 검증(이름/계정 화이트리스트 정규식, 설명 태그 차단)이 걸려 있다.
> 34개 엔드포인트 중 26곳이 `get_bearer_api_key_user` 의존성을 쓴다. 문제는 **관리 평면**에 한정된다.

### ✅ S-4. CORS 와일드카드 + 자격증명 허용 — **수정 완료(2026-07-31)**

`allow_origins=["*"]` 와 `allow_credentials=True` 동시 설정.
Bearer 키는 브라우저가 자동 첨부하지 않아 그 경로로는 새지 않으나, 쿠키 기반 관리자 세션(`/api/admin/login`·`/api/admin/session`)에서는 임의 사이트발 교차출처 요청 표면이 된다.

**추가 확인(2026-07-31, starlette 1.3.1 소스 대조)**: 이 조합은 단순히 "권장 안 함"이 아니다.
`CORSMiddleware.send` 가 `allow_all_origins and allow_credentials` 일 때 `*` 대신 **요청 Origin 을 그대로 되비춘다**
(`allow_explicit_origin`). 즉 브라우저의 와일드카드+자격증명 금지 규칙이 우회되어, 임의 사이트가 쿠키를 실은
교차출처 요청을 보내고 **응답까지 읽을 수 있다**.

**조치(완료)**: `[server] cors_origins` 신설.
- 비움(기본) → 와일드카드로 열되 **자격증명 차단**(`allow_credentials=False`). 동봉 web-UI(`/wizard`·`/chat`)는 동일 출처라 영향 없다.
- 지정 → 그 오리진만 허용 + 자격증명 ON. 교차출처 관리 UI 를 붙일 때 쓴다.
기동 시 어느 쪽으로 걸렸는지 로그로 찍는다.

### ✅ S-5. `eval()` 로 오류 메시지 파싱 — **수정 완료(2026-07-27)**

`aibot_validation.py:382`: `result['params'] = eval(params_str)` — 정규식 `params=(\{[^}]*\})` 로 뽑은 문자열을 그대로 `eval`.
입력은 쿼리 검증기 오류 메시지이므로 완전한 외부 통제는 아니지만, 신뢰 경계 밖 문자열을 코드로 실행할 이유가 없다.
**조치**: `ast.literal_eval` 로 교체 — **본 점검에서 수정함**.

### ✅ S-6. PII 마스킹이 한 핸들러 경로에만 적용 — **수정 완료(2026-07-31)**

`aibot_PII.py`(한국형 주민·사업자·법인·전화·여권·운전면허·카드·계좌·건강보험 인식기)는 잘 갖춰져 있으나,
**임포트 지점이 `handler_gemma` 하나뿐**이었다. 외부 API 핸들러(openai·claude) 사용 시 원문 PII 가 외부 사업자로
나갈 수 있었다 — 온프레미스 제품에서 가장 아픈 조합.

**조치(완료)**: 게이트를 `handler_base` 로 올려 전 핸들러가 물려받게 하고, 외부 API 경로에 배선했다.
- `handler_base`: `_init_pii_gate` · `_pii_mask_input`(question+history) · `_pii_mask_messages`(OpenAI messages) ·
  `_pii_unmask`(비스트리밍) · `_pii_stream_restorer`(스트리밍 — 토큰이 청크 경계에서 쪼개지지 않게 꼬리 버퍼).
  초기화는 **지연**이다. 게이트가 `rag_system.bge_model` 을 재사용하는데 그 배선 시점이 핸들러마다 달라
  `__init__` 순서에 의존하지 않게 했다.
- `handler_base.agent_complete` 가 `_agent_generate` 호출을 마스킹으로 감싼다 → 로컬 핸들러
  (gemma·gpt-oss·llama·qwen)는 구현 없이 completions2 경로가 덮인다. gemma 의 중복 정의는 제거(동작 동일).
- `handler_openai`·`handler_claude`: `generate_stream`·`generate_complete`·`agent_complete` +
  각 `generate_stream_error` 에 마스킹·복원 배선.

**현재 적용 범위** (README 도식은 이 표에 맞춰 정정함):

| 핸들러 | 외부 전송 | RAG 경로(`/api/query/stream`·`chats`) | completions2 경로 |
| :-- | :-- | :-- | :-- |
| openai · claude | **있음** | ✅ 마스킹 | ✅ 마스킹 |
| gemma | 없음(로컬) | ✅ 마스킹 | ✅ 마스킹 |
| gpt-oss · llama · qwen | 없음(로컬) | ⬜ 미적용 | ✅ 마스킹 |

로컬 핸들러의 RAG 경로를 남겨둔 것은 **외부 유출 경로가 아니기 때문**이다(심층방어 관점의 로그 노출은 남는다).
전면 적용은 3개 핸들러의 생성 경로를 더 건드려야 해서 실기동 검증과 함께 다루는 것이 맞다.

⚠️ **게이트는 기본 off** 다(`[pii] pii_mode = False`). 이 절의 보호는 켜야 작동한다 — 템플릿에 `[pii]` 절이
아예 없어 존재조차 안 보이던 것도 이번에 함께 고쳤다.

### 🟡 S-10. sync 엔드포인트 2개 무인증 — **수정 완료(2026-07-31)**

`GET /api/ai/sync/manifest` 와 `POST /api/ai/sync/points` 둘 다 인증 의존성이 없었다. 체인이 그대로 성립한다:

```
GET  /api/ai/sync/manifest        → 전 문서의 guid 목록
POST /api/ai/sync/points {guids}  → with_payload=True, with_vectors=True
                                     = 문서 본문 + 임베딩 벡터 전량
```

**영향 — 이 제품의 위협 모델에서는 제한적이다.** 처음엔 심각(🔴)으로 적었으나, 아래를 확인하고 내렸다:

- **Qdrant(6333) 자체가 무인증이다.** `install.sh` 는 `QDRANT__SERVICE__API_KEY` 를 설정하지 않고
  포트만 지정해 띄운다. 같은 망에서 `POST /collections/<컬렉션>/points/scroll` 한 번이면
  콘솔을 거치지 않고 **같은 데이터(payload+vector)가 그대로 나온다**.
- 즉 이 엔드포인트를 잠가도 **공격자의 능력은 달라지지 않는다.** 문 두 개 중 하나를 닫은 것이다.
- 이 배포 형태(개인이 clone 해서 자기 PC·내부망에 설치)에서 실제로 의미 있는 통제는 인증이 아니라
  **네트워크 노출**(방화벽·바인드 주소)이며, 그것이 S-8 이다.

그럼에도 고친 이유: (1) 클라이언트가 이미 Bearer 를 보내고 있어 **비용이 0**이고,
(2) 이 리포는 공개되며 문서가 **리버스 프록시로 콘솔 포트만 외부에 여는 구성**을 안내하는데,
그 구성에서는 6333 이 닫혀 있으므로 이 경로가 유일한 문이 된다.

**조치(완료)**: 두 엔드포인트에 `Depends(get_bearer_api_key_user)` 추가.
클라이언트(`aibot_sync.py`)는 **이미 `Authorization: Bearer` 를 보내고 있었다** — 서버가 검사만 안 했다.
따라서 기존 싱크 동작은 그대로다.

### 🟡 S-11. `GET /init` 무인증 구독 생성 — **수정 완료(2026-07-31)**

DB 테이블 초기화 + `ai_subscriptions` 행 삽입(신규 api_key 생성)을 무인증으로 수행했다.
응답에 api_key 를 싣지는 않아 즉시 키 탈취로 이어지지는 않으나, 익명으로 행을 무제한 생성할 수 있었다.
DB 모드(`use_db_mode=True`)에서만 동작하므로 기본 설치는 영향 없음.
**조치(완료)**: Bearer 인증 추가.

### 🟡 S-12. `POST /api/ai/validate` 무인증 — **수정 완료(2026-07-31)**

쿼리 검증(반복 최대 3회, LLM 경유 가능)을 무인증으로 호출할 수 있었다 — 연산 자원 소모 경로.
`[validation]` 기본 off 라 실효는 제한적. **조치(완료)**: Bearer 인증 추가.

### 🟡 S-13. `DELETE /api/ai/stream/{guid}` 무인증 — **수정 완료(2026-07-31)**

같은 경로의 `GET` 은 Bearer 인증인데 `DELETE` 만 없었다(비대칭). guid 가 uuid4 라 추측은 어렵지만
캐시 삭제가 익명으로 가능했다. **조치(완료)**: Bearer 인증 추가.

> ⚠️ **등급 판단 기준**: 이 문서의 위협 모델은 **개인이 자기 PC·내부망에 설치해 쓰는 온프레미스 도구**다.
> 공용 인터넷에 노출된 다중 사용자 서비스가 아니다. 같은 결함이라도 후자 기준으로 등급을 매기면
> 우선순위가 틀어지므로, 항목별로 **"이 배포 형태에서 공격자의 능력이 실제로 늘어나는가"** 를 기준으로 적는다.
>
> **남은 무인증 4개는 의도된 것이다**: `GET /wizard`·`GET /chat`(정적 UI — 키는 사용자가 화면에서 입력),
> `GET /`(헬스체크, api-integration.md 에 무인증으로 계약됨), `POST /api/reload`(본문 `api_key` 를
> DB 대조해 자체 인증. DB 모드 전용).

### ✅ S-7. 기본 관리자 키가 `0` 32자 — **수정 완료(2026-07-27)**

`run.sh:16`: `ADMIN_KEY="${ADMIN_KEY:-00000000000000000000000000000000}"`.
S-2를 복구하더라도 기본값이 이대로면 검증이 형식뿐이다.
**조치(완료)**: `install.sh` 단계 10-b 가 `openssl rand -hex 32`(64자) 로 `api_keys/admin.key` 를 0600 생성. `run.sh` 는 그 파일을 읽고 더미 fallback 제거(없으면 오류로 중단). 부수 수정: `Admin*Request` 의 `admin_key` 길이 상한이 레거시 `40` 이라 64자 키를 422 로 튕겼다 — **키 강도를 상한에 맞춰 낮추는 대신 상한을 64 로 올렸다**(3곳).

### 🟡 S-8. 신뢰 네트워크 전제 — 낮음(설계상 수용, 문서화 필요)

- 콘솔은 `0.0.0.0` 바인드 + **자체서명** 인증서(`ssl/selfsigned.{crt,key}`) — 브라우저 경고, MITM 방어 없음.
- Qdrant는 API 키 없이 기동(`install.sh:462`) — 포트에 닿으면 지식 컬렉션 전체 읽기·삭제 가능.
- **조치**: 온프레미스 폐쇄망 전제를 운영 가이드 운영 가이드에 명시하고, 외부 노출 시 리버스 프록시 + 실인증서 + Qdrant `service.api_key` 를 요구사항으로 건다.

### 🟢 S-9. 점검했으나 문제 없음

| 항목 | 확인 결과 |
| :-- | :-- |
| 커밋된 시크릿 | 없음. `.gitignore` 가 `config.ini`·`api_keys/`·`ssl/`·`utils/.encryption_key` 차단. 추적 파일 중 개인키·AWS·OpenAI 키 패턴 매치는 **PII 탐지 정규식과 문서 예시**뿐(오탐) |
| SQL 인젝션 | `aibot_db_manager.py` 전 경로가 `cursor.execute(sql, params)` 파라미터 바인딩. 문자열 결합 조립 없음 |
| YAML 역직렬화 | 확인한 전 지점이 `yaml.safe_load`. `unsafe_load`·`yaml.load(Loader=)` 없음 |
| 명령 실행 | `shell=True`·`os.system` 없음 |
| 의존성 | fastapi 0.116.1 · starlette 0.47.2 · pydantic 2.11.7 · cryptography 48.0.0 · urllib3 2.5.0 · requests 2.32.4 · PyYAML 6.0.2 — 알려진 고위험 구버전 아님. **단, `pip-audit` 미설치로 CVE 대조는 미수행** |

> **`pickle.loads` 다수 사용**(`aibot_db_manager.py`·`aibot_embedding*.py`·`aibot_rag_module*.py`)은 임베딩 인덱스 캐시 직렬화용이며,
> 입력이 자체 DB·자체 캐시 파일이라 현 구조에서는 신뢰 경계 안이다. 다만 **DB·캐시가 오염되면 곧 RCE**이므로 S-8의 전제(신뢰 네트워크)에 의존한다는 점을 기록해 둔다.

---

## 권고 (우선순위 순)

1. ~~S-2 → S-1 → S-3 복구~~ · ~~S-7 관리자 키 랜덤 생성~~ — **2026-07-27 완료.**
2. ~~**출시 전 권장** — S-4 CORS 오리진 제한.~~ → **완료(2026-07-31)** `[server] cors_origins`.
4. ~~**별도 과제** — S-6 PII 마스킹 공통화.~~ → **완료(2026-07-31)**. 잔여: 로컬 핸들러(gpt-oss·llama·qwen) RAG 경로는 미적용(외부 유출 경로 아님) — 실기동 검증과 함께 판단.
5. **운영 문서화** — S-8 폐쇄망 전제와 외부 노출 시 요구사항.
6. **다음 점검** — `pip-audit` 도입해 CVE 대조 자동화, 인증 우회 실증(DAST).

---

_5축: ① [요구사항](requirements.md) · ② [설계](architecture.md) · ③ [인터페이스](api-reference.md) · ④ 보안(이 문서) · ⑤ [검증](docs/testing-guide.md)_

# API 연동 가이드 — 외부 제품에서 ai-console 붙이기 (2026-07-23)

> 🇬🇧 English: [api-integration.en.md](api-integration.en.md)

> **5축 ③ 인터페이스의 연동 정본.** 인터페이스 표면 전체 지도는 [../api-reference.md](../api-reference.md),
> 요청/응답 스키마는 기동 중인 서버의 `/docs`(OpenAPI). 문서 구조 규칙은 [README §문서](../README.md#-문서).

> 이 콘솔은 **이중 성격**이다: ① 독립 채팅(`/chat`, `/wizard` web-UI) · ② **엔드포인트 제공자** —
> 외부 제품이 REST로 자유롭게 붙는다. 이 문서는 ②의 연동 계약.
> **전제(A안)**: 콘솔 1대 = 도메인 1개. 어떤 도메인으로 답하는지는 그 콘솔에 장착된 카트리지가 결정
> (카트리지 제작·장착은 [build-your-own-agent.md](build-your-own-agent.md)). 스키마 전체는 `https://<host>:8443/docs`(OpenAPI).

## 인증 — Bearer API 키

모든 엔드포인트는 `Authorization: Bearer <API_KEY>`. 키 발급/관리는 CLI:

```bash
./aibotctl keys generate <이름> <계정> --model gemma4-26b-moe-offload   # 키 발급(ACL·모델 지정 가능)
./aibotctl keys list                                                    # 목록
```

키마다 **모델·ACL(IP 화이트리스트)·지식 격리(sub_id)**를 줄 수 있다(멀티 테넌트 지식). 단 A안에서
도메인 페르소나(프롬프트)는 콘솔 전역이다.

---

## 핵심 엔드포인트 (연동 우선순위 순)

### 1. `POST /api/query/stream` — RAG 채팅 (SSE, 가장 간단)

> **스트리밍 모드 스위치(2026-07-28, FR-10)**: `[server] stream_tokens` 가 계약을 정한다.
> - `true`(신규 설치 기본): **토큰 단위 스트리밍** — `chunk` N프레임(조각이 생기는 대로) → (`sources`) → `usage`.
>   T2 실측: TTFT 9.96s < 총 11.85s, 7프레임(258자 응답). 클라이언트는 `chunk` 를 이어붙이면 된다(기존 계약과 호환).
>   ⚠ 이 모드의 `usage` 프레임은 **토큰 미집계(0)** 로 온다 — usage 가 필요한 소비자는 `false` 를 쓸 것.
> - `false`(구 배포 기본): 종전 계약 — 완성 후 `chunk` 1프레임 + (`sources`) + `usage`(실측 토큰값).
>   (2026-07-27 실측 6.31s/6.31s 의 "첫 프레임 = 완료 시각" 동작이 이것.)

한 방에 질문→답변 스트림. 요청은 `query` 하나. 응답은 **SSE**(`data: {json}\n\n`).

```bash
curl -sk -N -X POST https://<host>:8443/api/query/stream \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query":"질문 내용", "query_type":"general"}'
```

스트림 이벤트:
- `data: {"chunk":"답변"}` — 답변 본문. **현재 구현은 1회**(위 실측 주의 참조). 클라이언트는 여러 번 올 수 있다고 가정해 누적 처리할 것 — 토큰 스트리밍이 들어가면 그대로 동작한다
- `data: {"sources":["qna/x.yaml", ...]}` — RAG 참조 문서
- `data: {"error":"..."}` — 오류

→ **임베드형 채팅 위젯**에 최적(웹 `chat.html`이 이 엔드포인트를 소비).

### 1-b. `POST /agent/chat/completions` — **OpenAI 호환** (기존 LLM 클라이언트를 그대로 붙일 때)

이미 OpenAI 호환 클라이언트(`openai-full` 등)를 쓰는 서비스라면 **코드 수정 없이 URL 만 바꿔** 붙일 수 있다.

```bash
curl -sk -X POST https://<host>:8443/agent/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemma","messages":[{"role":"user","content":"질문"}]}'
```

응답은 표준 `chat.completion` 형태(`choices[].message` · `finish_reason` · `usage`)에 **표준 밖 `sources` 배열**이 덧붙는다(RAG 근거 추적용, 표준 클라이언트는 무시).

> 🔑 **`[server] agent_rag` 를 반드시 확인할 것.**
>
> | 값 | `/agent/chat/completions` 동작 |
> | :-- | :-- |
> | `true` (신규 설치 기본) | intent 분류 → PII 마스킹 → RAG 검색 → **카트리지 프롬프트** = 도메인 에이전트 |
> | `false` | messages 를 LLM 에 직결 = **생 LLM** (카트리지 무시) |
>
> `false` 인 채로 붙이면 Qdrant·BGE-M3·카트리지가 전부 놀고 일반 LLM 답변만 돌아온다.
> **`/agent/chat/completions2` 는 이름(Direct)대로 이 스위치와 무관하게 항상 직결**이므로, 카트리지를 쓰려면 `2` 가 아닌 쪽을 지정한다.

**멀티턴**: `messages` 의 마지막 `user` 발화가 질문, 그 앞이 history 로 전달된다(별도 변환 불필요).
**모델명**: 이 경로는 body 의 `model` 을 무시하고 서버의 `[model] model=` 을 쓴다 — 소비자가 모델명을 고정해 보내도 프리셋과 어긋나 실패하지 않는다.
**구 경로 호환**: `[server] agent_alias_prefix = /legacy` 를 주면 `/legacy/agent/chat/completions{,2}` 로도 같은 핸들러가 등록된다.

<details>
<summary>docker-compose 소비자 예시</summary>

```yaml
environment:
  # id|url|model|api_type|token
  LLM_MODELS: "gemma4|${LLM_AI_URL:-https://<ai-console-host>:8443/agent/chat/completions}|gemma|openai-full|${LLM_TOKEN}"
  LLM_DEFAULT_MODEL: "gemma4"
  LLM_TIMEOUT: "120"     # 로컬 모델 생성은 30s 로 부족할 수 있다
```
자체서명 인증서를 쓰므로 클라이언트에서 검증을 끄거나 실인증서를 넣어야 한다.
</details>

### 2. `POST /api/search` — RAG 검색만 (동기 JSON)

LLM 생성 없이 **참조 문서만** 검색. 앱이 자체 UI로 근거만 쓸 때.

```bash
curl -sk -X POST https://<host>:8443/api/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query":"질문", "search_type":"QNA", "top_k":5}'
```

응답: `{"success":true, "context":"...", "sources":["qna/x.yaml", ...]}`. `search_type`: `QNA|ACTION|PLAN`.

### 3. `POST /api/ai/chats` — 전체 대화 (세션·저장 포함)

정식 대화 파이프라인. 요청 스키마가 크고(`user_guid`·`type`·`prompt_count`·`prompt_token`·`locale` 등 필수)
`{"guid":...}`를 반환한 뒤 답변은 스트림 버퍼에서 조회하는 2단계다. 세션 로깅·복잡한 라우팅이 필요한
제품용 — 단순 질의응답이면 1번을 권장.

### 4. 지식 적재/삭제 — `POST /api/ai/prompts/bulk` · `DELETE /api/ai/prompts`

앱 데이터를 지식으로 넣기(멀티파트 YAML 업로드)·빼기(guid 목록). 보통 `aibotctl cartridge mount/unmount`가
자동 수행하므로 직접 호출은 커스텀 파이프라인에서만.

---

## 연동 패턴 3가지

| 패턴 | 방법 | 엔드포인트 |
|---|---|---|
| **임베드형** | 기존 앱 UI에 채팅 위젯 | `/api/query/stream` |
| **백엔드 호출형** | 앱 서버가 RAG 검색·요약만 | `/api/search` · `/api/ai/summarize` |
| **데이터 적재형** | 앱 DB/문서를 지식화(전용 카트리지) | `/api/ai/prompts/bulk` |

## 최소 연동 예 (Python)

```python
import requests, json
KEY = open("api_keys/default.key").read().strip()
r = requests.post("https://localhost:8443/api/query/stream",
    headers={"Authorization": f"Bearer {KEY}"},
    json={"query": "git rebase 뭐야?", "query_type": "general"},
    stream=True, verify=False)
for line in r.iter_lines():
    if line and line.startswith(b"data: "):
        d = json.loads(line[6:])
        if "chunk" in d: print(d["chunk"], end="", flush=True)
        if "sources" in d: print("\n참조:", d["sources"])
```

## 주의

- **TLS**: 자체서명 인증서(`ssl/selfsigned.crt`) — 개발은 `verify=False`/`-k`, 운영은 실인증서.
- **헬스체크**: `GET /` — **무인증**, 200 + 상태 JSON(실측 2026-07-27, 응답 ~3ms):
  ```json
  {"status":"online", "default_model":"gpt-oss",
   "models":{"gpt":false,"gpt-oss":true,"gemma":false,...}, ...}
  ```
  - **판정**: `HTTP 200` 이고 `status == "online"` 이면 healthy. 연결 거부·타임아웃·그 외 상태 코드는 전부 unhealthy 로 취급.
  - **깊은 판정**(선택): `models[default_model] == true` 까지 확인 — 기동 모델 백엔드 연결 여부.
  - **권장값**: 타임아웃 3s · 체크 간격 10s · 연속 3회 실패 시 unhealthy. **기동 유예는 넉넉히**(아래 모델 로딩 참조 — 26b 로딩 수 분간 연결 거부가 정상이므로 기동 직후 실패를 장애로 오판하지 말 것).
- **모델 로딩**: 재시작 직후 26b 로딩(수 분) 동안 연결 거부/503 — 재시도 로직 권장(위 헬스체크로 준비 완료를 감지).
- **도메인**: 응답 도메인은 콘솔 장착 카트리지가 결정. 다른 도메인이 필요하면 별도 콘솔(A안).
- **버전 안정성**: 위 4개 핵심 경로는 유지 대상. 그 외 내부 엔드포인트(`/api/reload` 등)는 변경 가능 — `/docs` 참조.

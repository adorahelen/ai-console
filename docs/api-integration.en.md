# API integration guide — connecting an external product to ai-console (2026-07-23)

> 🇰🇷 Korean original: [api-integration.md](api-integration.md)
>
> **The canonical integration document.** For a map of the whole interface surface see
> [../api-reference.md](../api-reference.md); for request/response schemas use `/docs` (OpenAPI) on a running server.

> This console has a **dual nature**: ① a standalone chat app (`/chat`, `/wizard` web UI) and
> ② an **endpoint provider** that external products call over REST. This document is the contract for ②.
> **Premise**: one console = one domain. Which domain it answers as is decided by the cartridge mounted on it
> (building and mounting cartridges: [build-your-own-agent.en.md](build-your-own-agent.en.md)).
> The full schema lives at `https://<host>:8443/docs` (OpenAPI).

## Authentication — Bearer API keys

Every endpoint takes `Authorization: Bearer <API_KEY>`. Keys are issued and managed from the CLI:

```bash
./aibotctl keys generate <name> <account> --model gemma4-26b-moe-offload   # issue a key (model and ACL optional)
./aibotctl keys list                                                       # list keys
```

Each key can carry its own **model, ACL (IP allowlist), and knowledge isolation (`sub_id`)** — that is, multi-tenant knowledge. The domain persona (the prompts), however, is console-wide.

---

## Core endpoints (in integration priority order)

### 1. `POST /api/query/stream` — RAG chat over SSE (the simplest)

> **Streaming mode switch (2026-07-28, FR-10)**: `[server] stream_tokens` decides the contract.
> - `true` (default on new installs): **token-level streaming** — N `chunk` frames (as pieces are produced) → (`sources`) → `usage`.
>   Measured on T2: TTFT 9.96 s against 11.85 s total, 7 frames for a 258-character answer. Clients simply concatenate `chunk`s (compatible with the older contract).
>   ⚠ In this mode the `usage` frame arrives with **no token accounting (0)** — consumers that need usage should set `false`.
> - `false` (default on older deployments): the previous contract — one `chunk` frame after completion + (`sources`) + `usage` with real token counts.
>   (This is the "first frame = completion time" behavior measured at 6.31 s / 6.31 s on 2026-07-27.)

Question in, answer stream out, in one shot. The request is just `query`. The response is **SSE** (`data: {json}\n\n`).

```bash
curl -sk -N -X POST https://<host>:8443/api/query/stream \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query":"your question", "query_type":"general"}'
```

Stream events:
- `data: {"chunk":"answer"}` — the answer body. Always accumulate: multiple frames are normal under token streaming, and a single frame is the older contract.
- `data: {"sources":["qna/x.yaml", ...]}` — the RAG documents used
- `data: {"error":"..."}` — an error

→ Best fit for an **embedded chat widget** (the bundled `chat.html` consumes this endpoint).

### 1-b. `POST /agent/chat/completions` — **OpenAI-compatible** (drop-in for existing LLM clients)

If your service already speaks an OpenAI-compatible client (`openai-full` and friends), you can point it here by **changing only the URL** — no code changes.

```bash
curl -sk -X POST https://<host>:8443/agent/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemma","messages":[{"role":"user","content":"your question"}]}'
```

The response is a standard `chat.completion` (`choices[].message`, `finish_reason`, `usage`) with a **non-standard `sources` array** appended for RAG provenance; standard clients ignore it.

> 🔑 **Check `[server] agent_rag` — this one matters.**
>
> | Value | What `/agent/chat/completions` does |
> | :-- | :-- |
> | `true` (default on new installs) | intent classification → PII masking → RAG retrieval → **cartridge prompts** = a domain agent |
> | `false` | passes `messages` straight to the LLM = **raw LLM** (cartridge ignored) |
>
> Connect while it is `false` and Qdrant, BGE-M3, and your cartridge all sit idle while you get generic LLM answers.
> **`/agent/chat/completions2` is direct-by-name and always bypasses this switch**, so if you want your cartridge, use the path *without* the `2`.

**Multi-turn**: the last `user` message is the question and everything before it is passed as history — no conversion needed.
**Model name**: this path ignores the `model` in the body and uses the server's `[model] model=`, so a consumer that hard-codes a model name will not fail against a mismatched preset.
**Legacy paths**: setting `[server] agent_alias_prefix = /legacy` registers the same handlers at `/legacy/agent/chat/completions{,2}`.

<details>
<summary>docker-compose consumer example</summary>

```yaml
environment:
  # id|url|model|api_type|token
  LLM_MODELS: "gemma4|${LLM_AI_URL:-https://<ai-console-host>:8443/agent/chat/completions}|gemma|openai-full|${LLM_TOKEN}"
  LLM_DEFAULT_MODEL: "gemma4"
  LLM_TIMEOUT: "120"     # 30s can be too short for local model generation
```
The certificate is self-signed, so either disable verification on the client or install a real one.
</details>

### 2. `POST /api/search` — retrieval only (synchronous JSON)

Retrieves **only the reference documents**, without LLM generation. For apps that render the grounds in their own UI.

```bash
curl -sk -X POST https://<host>:8443/api/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query":"your question", "search_type":"QNA", "top_k":5}'
```

Response: `{"success":true, "context":"...", "sources":["qna/x.yaml", ...]}`. `search_type` is one of `QNA|ACTION|PLAN`.

### 3. `POST /api/ai/chats` — the full conversation path (sessions and persistence)

The formal conversation pipeline. The request schema is large (`user_guid`, `type`, `prompt_count`, `prompt_token`, `locale` and more are required) and it is two-phase: it returns `{"guid":...}` and the answer is then read from the stream buffer. Meant for products that need session logging and complex routing — for plain question-and-answer, prefer endpoint 1.

### 4. Loading and deleting knowledge — `POST /api/ai/prompts/bulk` · `DELETE /api/ai/prompts`

Push app data in as knowledge (multipart YAML upload) or remove it (a list of guids). `aibotctl cartridge mount/unmount` normally does this for you, so call it directly only in custom pipelines.

---

## Three integration patterns

| Pattern | Approach | Endpoint |
|---|---|---|
| **Embedded** | A chat widget inside your existing UI | `/api/query/stream` |
| **Backend call** | Your app server uses retrieval or summarization only | `/api/search` · `/api/ai/summarize` |
| **Data ingestion** | Turn your app's DB/documents into knowledge (a dedicated cartridge) | `/api/ai/prompts/bulk` |

## Minimal integration example (Python)

```python
import requests, json
KEY = open("api_keys/default.key").read().strip()
r = requests.post("https://localhost:8443/api/query/stream",
    headers={"Authorization": f"Bearer {KEY}"},
    json={"query": "what is git rebase?", "query_type": "general"},
    stream=True, verify=False)
for line in r.iter_lines():
    if line and line.startswith(b"data: "):
        d = json.loads(line[6:])
        if "chunk" in d: print(d["chunk"], end="", flush=True)
        if "sources" in d: print("\nsources:", d["sources"])
```

## Caveats

- **TLS**: the certificate is self-signed (`ssl/selfsigned.crt`) — use `verify=False` / `-k` in development, a real certificate in production.
- **Health check**: `GET /` — **unauthenticated**, returns 200 plus a status JSON (measured 2026-07-27, ~3 ms):
  ```json
  {"status":"online", "default_model":"gpt-oss",
   "models":{"gpt":false,"gpt-oss":true,"gemma":false,...}, ...}
  ```
  - **Verdict**: healthy when `HTTP 200` and `status == "online"`. Connection refused, timeouts, and any other status code all count as unhealthy.
  - **Deeper verdict** (optional): also require `models[default_model] == true`, which reflects whether the model backend is connected.
  - **Suggested values**: 3 s timeout, 10 s interval, unhealthy after 3 consecutive failures. **Be generous with the startup grace period** — see model loading below; a 26b model refuses connections for several minutes while it loads, and that is not an outage.
- **Model loading**: right after a restart, a 26b model takes minutes to load and connections are refused or 503 — add retry logic and use the health check above to detect readiness.
- **Domain**: the answering domain is decided by the mounted cartridge. If you need another domain, run another console.
- **Version stability**: the four core paths above are maintained. Other internal endpoints (`/api/reload` and the like) may change — consult `/docs`.

# Runtime verification guide — T1: VM smoke test

> 🇰🇷 Korean original: [testing-guide.md](testing-guide.md)
>
> This verifies by running what static analysis could not catch (compilation, AST, module execution, plus five code-review passes and some twenty fixes).
> Target environment: **Ubuntu 24.04 VM · 4 vCPU · 16GB RAM · 50GB disk · no GPU**
> Time: ~20–30 min to install (link-dependent) + ~30 min to test

## What is being verified

Six things that cannot be verified statically and are confirmed here for the first time:

| # | Area | Why static analysis missed it |
|:--:|---|---|
| V1 | install.sh actually building (pip, llama.cpp, Qdrant, HF downloads) | Only dry-run was exercised |
| V2 | The model actually loading (llama-cpp-python + Llama-3.1-8B Q4 GGUF) | No model file was present |
| V3 | File-based auth integrated (the whole stack through uvicorn) | Only individual functions were run |
| V4 | The three wizard LLM calls answering for real (draft, test chat, qna conversion — **quality**) | Requires an LLM |
| V5 | llama chat template correctness (the newly written agent path) | You have to see the answer to judge |
| V6 | The whole RAG pipeline (embed → store in Qdrant → retrieval hit) | Requires Qdrant and BGE |

T2 (GPU and performance) runs separately on a 5070 Ti desktop — see [the bottom](#t2--gpu-verification-desktop).

---

## T1-0. Preparation

```bash
# On the VM (the Ubuntu 24.04 cloud image already ships git and the build tools)
git clone https://github.com/adorahelen/ai-console-public.git ai-console && cd ai-console
```

> While the repo is private, `clone` will not work. On a multipass VM, copy it in from the host:
> ```bash
> git archive --format=tar HEAD | gzip > ~/ai-console.tar.gz   # HEAD only, no working-tree noise
> multipass transfer ~/ai-console.tar.gz <vm>:/home/ubuntu/ai-console.tar.gz
> multipass exec <vm> -- bash -lc 'mkdir -p ~/ai-console && tar xzf ~/ai-console.tar.gz -C ~/ai-console'
> ```
> ⚠️ snap confinement stops multipass from reading arbitrary paths under `/tmp` — stage the file in your home directory.

Expected: the `ai-console` directory contains `install.sh`, `models.yaml`, and `qa_llm.py`.

## T1-1. Install (V1)

```bash
./install.sh --preset llama31-8b-q4 --yes     # the combination that produced the output below
```

> ⚠️ **The `cpu-only` tier now defaults to `gemma4-e4b-q4`** (the Gemma-first switch). The block
> below records a measured run of `llama31-8b-q4` (`runtime=inprocess`), so the preset is pinned
> to reproduce it. **The default path (`./install.sh --tier cpu-only --yes`) has never been run**;
> being `runtime=server` it has no `llama-cpp-python` step and downloads a different repo —
> confirming that difference is the new work in V1.

Expected output per stage (reflecting the 2026-07-21 VM run):

```
✓ tier: cpu-only
✓ selected: llama31-8b-q4 (handler=llama, runtime=inprocess)
▸ checking system dependencies (git·cmake·compiler·openssl·python3-venv)
  (the multipass Ubuntu 24.04 cloud image passes straight through — measured 2026-08-19.
   On a minimal server image, detecting missing python3-venv/python3-dev and printing an apt command is normal)
▸ python environment (.venv)
▸ installing the torch CPU build (no GPU — avoids the CUDA build, saves ~5GB of disk)   ← ★ this line is required
▸ installing python dependencies (requirements.deploy.txt — includes source builds of annoy/hnswlib, several minutes)
▸ installing llama-cpp-python (in-process preset — source build, 5–10 min: all CPU cores at 100% is normal)
✓ pinned release tag: bXXXXX
✓ Qdrant vX.Y.Z → ./qdrant/qdrant
▸ downloading the LLM: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF (*Q4_K_M*)   ← ~5GB
  (+ BGE-M3 ~2.3GB — onnx excluded (`24b01e6`); 2.2G measured on disk. On a 1–2MB/s link this is around 30 min; waiting on the network is normal)
✓ install complete — tier: cpu-only / preset: llama31-8b-q4
```

⚠️ The three large pip installs (torch, requirements, llama-cpp-python) must show live progress bars. If it is silent, you are on an old script — `git pull` and re-run.
To watch progress from another terminal: `watch -n2 'du -sh ~/ai-console/.venv'`.

**Success criteria (all of them):**

```bash
ls config.ini api_keys/default.key ssl/selfsigned.key        # all three exist
.venv/bin/python -c "import llama_cpp; print('ok')"          # ok
ls models/llama31-8b/*.gguf models/bge-m3/pytorch_model.bin  # models present
grep "auth_mode" config.ini                                  # auth_mode = file
df -h . | tail -1                                            # ~15–18GB used (8B Q4 is ~5GB)
```

## T1-2. Startup (V2)

> ⚠️ **Corrected 2026-08-19 — the default path does not need the manual startup below.**
> With a `runtime=server` preset (the default for every tier since the Gemma-first switch),
> `install.sh` registers and starts two systemd units (`ai-console-<instance>` and
> `ai-console-qdrant-<instance>`) and waits for the endpoint to become ready before it exits.
> Use the manual steps only for a `--no-service` install or an in-process preset such as `llama31-8b-q4`.

The healthy state right after install — confirming this *is* V2:

```bash
systemctl is-active ai-console-default ai-console-qdrant-default  # active / active
pgrep -af "qdrant|llama-server|qa_llm"                            # three processes
curl -s http://127.0.0.1:8183/health                              # {"status":"ok"}  ← llama-server
curl -s http://127.0.0.1:6333/collections                         # {"result":...,"status":"ok"}
```

Manual startup (in-process preset, or a `--no-service` install):

```bash
./qdrant/qdrant > qdrant.log 2>&1 &
sleep 3 && curl -s http://127.0.0.1:6333/collections
.venv/bin/python qa_llm.py 2>&1 | tee console.log
```

**Success criterion:**
```bash
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/ \
  -H "Authorization: Bearer $(cat api_keys/default.key)"
# → 200 (API status JSON). A 401 means auth (V3) is broken; connection refused means startup failed
```

> ⚠️ BGE-M3 takes 1–2 minutes to load on CPU — connection refused during that window is not a failure.



```bash
KEY=$(cat api_keys/default.key)
# valid key → 200
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/api/wizard/prompt-draft \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"role":"test"}'
# invalid key → 401
curl -sk -o /dev/null -w "%{http_code}\n" https://localhost:8443/api/wizard/prompt-draft \
  -H "Authorization: Bearer WRONG" -H "Content-Type: application/json" -d '{"role":"x"}'
```

Success: the first call returns **200** (or 503 = handler not loaded, which means V2 failed), the second **401**.

## T1-4. The three wizard LLM calls (V4, V5)

Open `https://localhost:8443/wizard`, click through the self-signed warning, and paste the API key. Or from the CLI:

```bash
KEY=$(cat api_keys/default.key)
# ① prompt draft
curl -sk https://localhost:8443/api/wizard/prompt-draft -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"role":"guide to internal HR policy","audience":"all employees","tone":"polite","rules":"say you do not know when the policy does not cover it","needs_action":false}' | python3 -m json.tool
# ③ document → qna conversion
curl -sk https://localhost:8443/api/wizard/knowledge-convert -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"First-year employees get 15 days of annual leave. Remote work is allowed up to twice a week.","count":3}' | python3 -m json.tool
```

**Success criteria (judged by eye — this is about quality):**
- ① `draft` contains a system prompt that reflects the role and the rules. If broken tokens (`<|start_header_id|>`, `<|eot_id|>`, …) leak into the response, that is a **V5 (chat template) failure**.
- ③ `items` is a valid array whose question/answer pairs are grounded in the document. An empty array means the LLM failed to hold the JSON format → the meta-prompt needs adjusting.

> ⚠️ **Second unverified risk**: the agent path of the llama chat template (`handler_llama.build_agent_prompt`) was written during this refactor — answer quality and token leakage surface here for the first time. On an 8B model on CPU, a single draft can take **2–5 minutes**; that is not a timeout.

## T1-5. Mounting a cartridge + loading RAG (V6)

> ⚠️ **Corrected 2026-08-19 — `/api/wizard/cartridge-save` does not exist.**
> The wizard exposes exactly three endpoints: `prompt-draft`, `knowledge-convert`, and `cartridge-mount`.
> Following the old `cartridge-save` steps returns 404.
> The canonical path for saving and loading a cartridge is the **`aibotctl cartridge`** CLI.

Verify with the bundled example cartridge (`cartridges/console-guide` — the console's own usage guide, 8 knowledge docs):

```bash
./aibotctl cartridge validate cartridges/console-guide   # schema check (read-only)
./aibotctl cartridge mount    cartridges/console-guide   # wiring + embed → load into Qdrant
./aibotctl cartridge status                              # mount state
```

**Success criteria:**

```bash
# the collection does not exist before loading → after mount, points equals the number of docs
curl -s http://127.0.0.1:6333/collections/bge \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])"   # → 8
./aibotctl cartridge status    # mounted: console-guide · knowledge: 8
```

> `mount` takes effect **at runtime without a restart** (the handler reloads). Roll back with `./aibotctl cartridge purge`.
> ⚠️ If the cartridge's `model.recommended` differs from the active handler you get a warning — `mount` never changes `[model]`.

To build a cartridge from your own documents, extract QnA pairs with `knowledge-convert`, save them as
`cartridges/<name>/knowledge/*.yaml`, and run the same CLI. The wizard UI (step 3) calls `cartridge-mount` under the hood.



`/api/search` (request `SearchRequest{query, search_type, top_k}`) is the path that exercises retrieval alone. No generation, no permission gate — just **whether the knowledge you loaded actually comes back**.

```bash
KEY=$(cat api_keys/default.key)
curl -sk https://localhost:8443/api/search -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"how many days of annual leave?","search_type":"QNA","top_k":5}' | python3 -m json.tool
```

Success: the returned context/sources include **the "15 days" document you loaded in T1-5** — embed → store → retrieve works end to end.

> ✅ This path was **fixed during re-verification to route to the active model (DEFAULT_MODEL)** — the original code hard-coded the `gpt-oss` handler, so a qwen install returned 503 (a handler-key coupling that the Phase 1 string bleaching did not catch).
> ⚠️ The full generation paths (`/api/query/stream`, `/api/ai/chats`) have complex request schemas and permission models. The smoke test stops at retrieval; generation quality was already observed in the **T1-4 wizard test chat**. For full conversations, check the schema in `/docs` and test separately.

---

## Symptom → cause → fix

| Symptom | Likely cause | Fix |
|---|---|---|
| `ensurepip is not available` (venv creation fails) | python3-venv not installed — **seen on T1** (newer versions detect this up front) | `sudo apt install -y python3.12-venv && rm -rf .venv`, then re-run |
| `fatal error: Python.h` (annoy/hnswlib build fails) | python3-dev not installed — **seen on T1** (same) | `sudo apt install -y python3.12-dev`, then re-run (pip skips what is installed) |
| `Could not find nvcc … CUDA Toolkit not found` (llama-cpp-python / llama.cpp build) | A GPU machine without the CUDA toolkit — **seen on T2** (newer versions detect and explain at startup) | Install `cuda-toolkit-12-9` from the official NVIDIA repo (Ubuntu's stock `nvidia-cuda-toolkit` is 12.0 and lacks sm_120), then re-run |
| `import llama_cpp` fails | The source build failed (compiler or memory) | Check the log with `pip install llama-cpp-python -v`. If RAM is short, add swap |
| `[paths]/[prompts] required key missing` at startup | An old config.ini.template | `git pull`, then regenerate with `cp config.ini.template config.ini` |
| Every API returns 401 | `auth_mode` unset, or a key mismatch | `grep auth_mode config.ini` should be `file`; compare against `cat api_keys/default.key` |
| Wizard returns 503 | llm_handler not loaded (the model failed to load) | Look for RAG/model initialization errors in console.log |
| `<|start_header_id|>`-style leakage in the draft | A llama chat template bug (V5) | Fix `handler_llama.build_agent_prompt` — then regress against this guide |
| qna conversion 502 `qna parse failed` | A small LLM breaking JSON syntax (quotes inside strings) — **seen on T2** (reproduced with 8B on long English documents) | Newer versions use JSONL with lenient line-by-line parsing (only broken lines are dropped). If you are on an old build, `git pull`. If it still yields zero items, lower `count` and split the document |
| Qdrant connection refused | It is not running | `./qdrant/qdrant &`, then `curl :6333/collections` |
| Appears to hang while loading BGE | Normal on CPU (1–2 min) | Confirm python CPU usage with `top`, and wait |
| `/api/search` returns 503 | An old build (gpt-oss hard-coded) | `git pull` — you need the DEFAULT_MODEL routing fix |
| `/api/query/stream` returns 403 | `query_type` is not in permissions | File auth allows general/qwen/gemma and others in newer builds. If you are on an old build, `git pull` |

> 🔎 (from the fifth review) Every gpt-oss hard-coding is resolved — both v1 `/agent/chat/completions` and ticket_memo route to the active model. **v1 now requires Bearer auth too**, same as completions2. The validation plugin is disabled by default in the template (`plugin_module` empty) and activates only under a cartridge that provides one.

---

## Completion checklist

- [ ] V1 install.sh runs to completion (config, keys, SSL, models created)
- [ ] V2 console starts and answers on port 8443
- [ ] V3 file auth returns 200/401
- [ ] V4 prompt draft and qna conversion return valid responses
- [ ] V5 no chat-template token leakage
- [ ] V6 cartridge saved → knowledge loaded → collection vector count increased
- [ ] RAG knowledge appears in a full conversation

**If any of them fails**: work the symptom table first; if that does not resolve it, file an issue with `console.log` and `qdrant.log` attached. The ⚠️ items in this guide are the likeliest failures.

---

## T1 results — the cpu-only default path: ✅ V1–V6 all passed (measured 2026-08-19)

`./install.sh --tier cpu-only --yes` completed **the default path for the first time**.
multipass Ubuntu 24.04 VM · 4 vCPU · 16GB RAM · 50GB disk · no GPU.
Round 1 (2026-07-21) used `--preset llama31-8b-q4` (in-process), so preset, runtime and handler all differ.

| Axis | Result |
|---|---|
| V1 install | Completed. Preset `gemma4-e4b-q4` auto-selected · GGUF 4.0G · BGE-M3 2.2G · 18G on disk · `auth_mode=file` |
| V2 startup | Two systemd units active, three processes up, `/health` ok, console returns 200 |
| V3 auth | Valid key 200 / invalid key 401 |
| V4 wizard | `prompt-draft` reflected every rule · `knowledge-convert` produced 3 grounded pairs |
| V5 template | **Zero special-token leakage** (all three responses scanned). The `handler_gemma` path is correct |
| V6 RAG | validate → mount in 6.5s · Qdrant `points=8` · 5 sources cited, the correct doc ranked first |

**Latency (4 vCPU, no GPU)**: short chat 63s · draft 68s · qna conversion 78s · RAG query 125s.
That is slow, not broken — give `--max-time` plenty of room on this tier.

**Fixed as a result of this run**
- `install.sh` — a cpu-only install spawned llama-server with `-ngl 99` (the template default leaking through) → force `n_gpu_layers=0` when the tier is cpu-only
- `cartridges/console-guide` — `model.recommended` was `llama31-8b-q4`, out of step with the default install → `gemma4-e4b-q4`
- `cartridge_mount.py` — warned whenever `recommended` was **present at all**, leaving the user to compare a preset name against a handler name → resolve preset→handler through models.yaml and warn only on a real mismatch; a separate message for an unknown preset
- This document — the manual-startup premise in T1-2, the non-existent endpoint in T1-5, the BGE size, the dependency note, the clone URL

**Still unmeasured**: tok/s and runtime memory for this tier (the [unmeasured] labels in `models.yaml` stand).

---

## T2 — GPU verification (desktop)

### Round 1 — lightweight GPU verification with llama31-8b: ✅ V1–V6 all passed (measured 2026-07-21)

7500F + 5070 Ti 16GB + 30GiB RAM, Ubuntu 24.04, CUDA toolkit 12.9 (NVIDIA repo):

- **V1**: Installed to completion including both CUDA builds (llama-cpp-python + llama-server). CPU observed at 93°C during the build (within normal throttling range)
- **V2**: Loaded in full GPU mode — **35 GPU layers, 7,770 MiB VRAM** (8B Q4 + ctx 16384 KV + BGE-M3)
- **V3**: File auth 200/401 as expected
- **V4·V5**: The wizard ran end to end (draft → test chat → 5 qna items → save → load) with no chat-template token leakage.
  Two defects found and fixed in the process: qna extraction failed to parse JSON entirely on documents containing quotes → replaced with a **Q:/A:/AL: line-tag format plus a three-stage fallback parser**
- **V6**: `/api/search` retrieved all five loaded knowledge items, with the correct document ranked top at similarity 1.0 — **BGE-M3 computes correctly on the GPU (sm_120)**

### Round 2 — measuring the primary spec (the remaining work)

- `./install.sh --preset gpt-oss-20b --yes` → the **separate llama-server process runtime** (a different path from round 1's in-process)
- To measure: VRAM usage, **tok/s**, and two concurrent slots (`n_parallel`)
- **KV cache reuse A/B**: the gpt-oss spawn lacks `--cache-reuse` (only gemma inherited it) — repeat the same RAG query with 0 vs 256 and **compare TTFT** → decide the default
- If time allows afterwards, gemma-26b-moe-offload — measure `--cpu-moe` to clear the `[not measured]` labels in models.yaml (mind the 30GiB RAM boundary)

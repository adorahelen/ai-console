# install.sh paths — exactly what happens per option

> 🇰🇷 Korean original: [install-paths.md](install-paths.md)
>
> `./install.sh` branches on two axes: **① whether a GPU is present (VRAM detection)** and
> **② the preset's runtime** (inprocess / server / api). This document lays out which steps run or are
> skipped in each combination, and where the time goes.
> Timings marked [measured] come from 2026-07-21 runs on T1 (a 4-vCPU VM) and T2 (7500F + 5070 Ti);
> [estimated] means not yet measured.

## Decision tree

```
detect VRAM via nvidia-smi
├─ VRAM = 0  →  CPU path   (torch CPU wheels · no CUDA compilation at all · nvcc not needed)
└─ VRAM > 0  →  GPU path   (nvcc required · two CUDA kernel compiles · default (CUDA) torch wheels)
preset runtime
├─ inprocess (llama31-8b)      → build llama-cpp-python + build llama.cpp + download the model
├─ server    (gpt-oss, gemma)  → build llama.cpp only + download the model (no llama-cpp-python)
└─ api       (openai, claude)  → no builds, no model downloads (venv + config + keys only)
```

## A. CPU path — `--tier cpu-only` (VMs and GPU-less servers)

| Step | What it does | Time |
|---|---|---|
| Dependency check | git·cmake·g++·openssl + **python3-venv·python3-dev** (prints apt guidance if missing) | seconds |
| venv + torch **CPU** wheels | The CPU index (a few hundred MB) instead of the default PyPI CUDA build (~5GB) | [measured] a few minutes |
| requirements | Includes **building the annoy and hnswlib C extensions from source** | [measured] a few minutes |
| llama-cpp-python | C++ build — **CPU backend only, zero CUDA (.cu) kernels** | [measured] 5–10 min |
| llama.cpp (llama-server) | cmake CPU build (built as a tool even when the runtime is not `server`) | [estimated] 5–10 min |
| Qdrant | Binary download, no compilation | seconds |
| Model download | BGE-M3 ~4.5GB + llama31-8b Q4 ~5GB | [measured] ~1 hour on a 1–2MB/s link |
| config·SSL·keys | template → config.ini, self-signed cert, api_keys/default.key | seconds |

**Total**: ~15 min of building plus downloads (link-dependent). All cores pegged at 100% during the build is normal.

## B. GPU path — auto-detected (desktops and GPU servers)

**Prerequisite: the CUDA toolkit (nvcc).** If absent, the installer detects it up front, stops, and prints the official NVIDIA repository install command. (Ubuntu's stock `nvidia-cuda-toolkit` is 12.0 and does not support sm_120 on the newest GPUs — use `cuda-toolkit-12-9`.)

Only the differences from path A:

| Step | Difference | Time |
|---|---|---|
| torch | Skips the CPU index → requirements installs the default (CUDA) wheels (~3GB) | [estimated] a few minutes of download |
| llama-cpp-python (inprocess only) | `GGML_CUDA=on` — **compiles hundreds of CUDA kernels** (fattn/mmq template instantiations across every quantization format) | [measured, in progress] 10–30 min; CPU observed up to 93°C (throttling is expected) |
| llama.cpp (llama-server) | Also `GGML_CUDA=ON` — **a second CUDA compile** | [estimated] 10–20 min |
| At startup | The model loads onto the GPU (llama31-8b logs "full GPU mode · 35 layers") | — |

**Total**: ~30–60 min of building plus downloads. If temperatures concern you, `CMAKE_BUILD_PARALLEL_LEVEL=3 ./install.sh …` lowers parallelism (roughly 2× the time, and that wheel rebuilds from scratch).

[measured, T2 · 2026-07-21, 7500F + 5070 Ti] The llama31-8b preset **completed the full GPU path**, including both CUDA builds. Downloads on a 12–14MB/s link: BGE-M3 in 3 min, the 8B GGUF (4.92GB) in 5 min 41 s.

> Note: llama.cpp is built even for inprocess presets, as a tool (per the code comment). That makes CUDA compile twice, so whether it can be skipped on the inprocess path is a candidate improvement — pending verification.

## C. API path — `--tier api` (openai-api / claude-api)

Zero builds, zero bytes of model download. venv (+requirements) → config.ini → SSL and key generation, and that is all. [measured] a few minutes on a dry run. After installing, fill in `[openai] api_key` or the `[bedrock]` credentials in config.ini.

## Download size per preset

Embeddings (BGE-M3) cost a flat **2.3GB** for every preset — `install.sh` excludes `onnx/` and the
illustrations (the full repo is 4.3GB, but 2.2GB of that is the ONNX runtime and this console only
uses the torch path). [measured 2026-07-28] Already present, it is reused, so a second preset only
adds its LLM share.

Every LLM size below is a measured HF `content-length`. [confirmed 2026-07-31]

| Preset | LLM | + embeddings 2.3GB | Notes |
|---|---|---|---|
| gemma4-e2b-q4 | 2.62GB (UD-Q4_K_XL) | **~4.9GB** | cpu-only lowest-end fallback |
| gemma4-e4b-q4 | 4.22GB (UD-Q4_K_XL) | **~6.5GB** | cpu-only · gpu-8gb default |
| gemma4-12b-q4 | 6.72GB (UD-Q4_K_XL) | **~9.0GB** | gpu-16gb default |
| llama31-8b-q4 | 4.92GB (Q4_K_M) | **~7.2GB** | `--preset` only |
| gpt-oss-20b | 13.79GB (F16) | **~16.1GB** | `--preset` only |
| gemma4-26b-moe-offload / -full | 16.95GB (UD-Q4_K_M) | **~19.3GB** | both presets share one file |
| openai-api / claude-api | 0 | **0** | no download, no build |

## Swapping models, and the gemma4 line (2026-07-21)

**Three ways to swap** — in every case only families whose handler is in the registry work
(gpt / gpt-oss / gemma / llama / claude / qwen; other families such as mistral cannot, for lack of a handler):

1. `./install.sh --preset <name> --yes` — switch presets. Builds are reused, only the new model downloads, config is regenerated.
2. Add a preset to models.yaml, then do ① — the path for testing a new GGUF variant.
3. Manually: edit `config.ini [model] model=`, the `[paths]` model path, and the matching `llama_server_*` section.

**The official Gemma 4 lineup** (released 2026-03-31 · Apache 2.0 · official QAT GGUF for every model, so llama.cpp-compatible. Sources: ai.google.dev/gemma/docs/core, blog.google, unsloth):

| Model | Q4 memory (official) | Verdict on a 16GB GPU | Preset (`--preset` only, not yet measured) |
|---|---|---|---|
| E2B (2B active) | 2.9GB | Fits, but console-level quality unknown | `gemma4-e2b-q4` |
| E4B (4B active) | 4.5GB | Full GPU, a light place to start | `gemma4-e4b-q4` |
| 12B (unified multimodal) | 6.7GB | Full GPU with KV headroom — sweet-spot candidate | `gemma4-12b-q4` |
| 26B A4B (MoE) | 14.4GB | Needs offload — the model verified upstream | `gemma4-26b-moe-offload` (tier-suggested) |
| 31B (dense) | 17.5GB | **Not recommended** — exceeds VRAM, and being dense it collapses under offload. 24GB+ only | not registered |

Recommended sweep order on 16GB: `e4b` → `12b` → `26b-moe-offload`. At each step, measure:
VRAM (`nvidia-smi`), tok/s, and **whether chat-template tokens leak**. Do not assume a new variant is fully compatible with the handler_gemma template before measuring — if tokens leak, inspect `handler_gemma.build_agent_prompt`.
Once measured, clear the `[not measured]` note on the preset and decide whether to promote it into the tier list.

**Not recommended**: 31B (on 16GB), qwen (the crash branch was removed in the third review, but answer quality is unverified — it stays out of the tier suggestions), and any family without a handler.

## Checking your installed model inventory

```bash
# All GGUF files with sizes (only >100MB — excludes llama.cpp's ggml-vocab-* test files)
find ~ -name "*.gguf" -size +100M -exec du -h {} + 2>/dev/null

# The HF cache (BGE-M3 embeddings, safetensors, and other non-GGUF models/datasets)
hf cache scan
```

Models are kept per preset under `models/<preset>/`, so the output above tells you at a glance which presets you have downloaded.

## Re-running, switching, cleaning up

- **Safe to re-run**: pip skips what is installed, models are separated per preset, and config.ini is overwritten only with `--yes`. If it died partway, fix the cause (a missing dependency, say) and **re-run the same command** — that is the intended recovery.
- **Switching presets**: `./install.sh --preset <other> --yes` — builds are reused, only the new model downloads.
- **Cleanup**: the only system-wide residue is apt packages (plus the CUDA toolkit). Everything else goes with `rm -rf ~/ai-console ~/.cache/huggingface ~/.cache/pip`.

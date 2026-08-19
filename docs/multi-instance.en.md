# Running multiple instances (2026-07-27)

> 🇰🇷 Korean original: [multi-instance.md](multi-instance.md)
>
> **Decision: one instance per customer (domain).** Stand up N consoles on one host, each with its own
> clone, config, ports, Qdrant, and cartridge.
> This extends the mounting design's "one console per domain, mounted globally" model across several machines — the cartridge model itself is unchanged.
> **Docker was rejected** for this: re-verifying GPU passthrough, model volumes, and the llama.cpp CUDA build costs more than it returns, given that the installer is already a one-shot bare-metal path. `docker/` remains as an asset.
> **Related**: [install-paths.en.md](install-paths.en.md)

---

## What an instance is

**An instance is one clone directory.** That is the premise; everything else follows from it.

| Asset | How it is isolated |
|---|---|
| Code and config.ini | Per clone directory |
| Knowledge (Qdrant) | **Its own Qdrant process per instance** (see below) |
| Cartridge mount state | `cartridges/.mounted.json` (per clone) |
| API keys | `api_keys/*.key` (per clone) |
| Model weights | `models/` per clone — symlink them to share |

Registering one directory under two instance names is **refused**
([scripts/alloc_ports.py](../scripts/alloc_ports.py)) — ports, cartridges, and Qdrant would tangle.

### Sharing Qdrant — parameterized collection names (2026-07-28)

The knowledge collection name has a single source: **`[qdrant] collection`** (default `bge`) via `config_utils.qdrant_collection()`. All 21 code literals (embedding, both RAG modules, the prompts API, sync, mount/purge, utils) were externalized through that helper.

- **Behavior is unchanged at the default** — with the key absent it stays `bge`, compatible with existing collections (verified on T2 at runtime).
- **Sharing one Qdrant**: give each instance a different `collection` and knowledge is isolated per collection.
  If the names match, however, instances **overwrite each other's knowledge** exactly as before — when sharing, name assignment is the operator's responsibility.
- A separate Qdrant per instance (with its own port) is still the default recommendation — it buys fault isolation too.

---

## Port assignment

`install.sh` assigns six ports via [scripts/alloc_ports.py](../scripts/alloc_ports.py).

| Role | Default | Note |
|---|---|---|
| `server` | 8443 | Console HTTPS |
| `qdrant` | 6333 | Qdrant HTTP |
| `qdrant_grpc` | 6334 | **No config section** — used only as a startup environment variable |
| `llama_server` | 8181 | The gpt-oss family |
| `llama_server_translation` | 8182 | Translation only |
| `llama_server_gemma` | 8183 | The gemma presets |

**Registry**: `~/.ai-console/instances.tsv` (relocatable via `AI_CONSOLE_HOME`)

```
acme	/srv/console-acme	server=8444,qdrant=6335,qdrant_grpc=6336,llama_server=8184,...
```

The registry exists because a LISTEN check alone would mistake a **stopped** sibling instance's ports for free. The assignment rules:

1. If the instance is already registered, **return its recorded ports** so reinstalling never shuffles them.
2. If it is new but a `config.ini` exists, **inherit those ports** (installs that predate the registry).
   `qdrant_grpc` has no section, so it is inherited as HTTP+1 per Qdrant's convention.
3. Otherwise, count up from the defaults, avoiding both sibling reservations and anything actually listening.

When you retire an instance, **delete its line from the registry** — that reclaims the ports.

---

## Installing

```bash
# The first instance (default)
curl -fsSL https://raw.githubusercontent.com/adorahelen/ai-console-public/main/install.sh | sh

# A per-customer instance — clones to ~/ai-console-acme with ports auto-avoided
curl -fsSL .../install.sh | sh -s -- --instance acme --yes

# To choose the clone location yourself
AI_CONSOLE_DIR=/srv/console-acme  curl -fsSL .../install.sh | sh -s -- --instance acme
```

If it fails partway, **run the same command again**. Heavy steps (pip, model downloads) are skipped via markers in `.install-state/`. Use `--fresh` to start over from scratch.

---

## Starting up

From the instance directory, **on its own ports**.

```bash
cd /srv/console-acme

# 1) Qdrant — specify both HTTP and gRPC
QDRANT__SERVICE__HTTP_PORT=6335 QDRANT__SERVICE__GRPC_PORT=6336 ./qdrant/qdrant &

# 2) The console
.venv/bin/python qa_llm.py        # or ./run.sh
```

- Qdrant's `storage/` and `snapshots/` are created **relative to cwd**, so simply starting it from the clone directory separates the data naturally. (Verified.)
- Omitting `QDRANT__SERVICE__GRPC_PORT` makes the second instance log `Error while starting gRPC server: transport error`. This console does not use gRPC so nothing dies, but the log gets noisy — set it. (Verified.)

---

## Managing cartridges (per instance)

`install.sh` creates symlinks in `~/.local/bin` — `aibotctl` for `default`, and **`aibotctl-<instance>`** for the rest.

```bash
aibotctl-acme cartridge list                       # list, with the mounted one marked
aibotctl-acme cartridge validate cartridges/acme   # validate before mounting
aibotctl-acme cartridge mount    cartridges/acme   # wire + load knowledge + apply at runtime (no restart)
# fallback if the reload is not confirmed: ./run.sh restart
aibotctl-acme cartridge status

aibotctl-acme cartridge unmount                    # remove only the knowledge this cartridge added
aibotctl-acme cartridge purge                      # drop the whole collection → a clean console
```

`aibotctl-*` always **cd's to its own repository root** before running, wherever you call it from, because the `[prompts]` wiring in config is stored as paths relative to cwd
([cartridge_mount.py:99](../cartridge_mount.py), `relpath(cart_abs, os.getcwd())`).
So pass cartridge paths **relative to the repository root**, or use absolute paths.

### unmount vs purge

| | What it deletes | Console required |
|---|---|---|
| `unmount` | Only the guids recorded in `.mounted.json` (over REST) | **Yes** |
| `purge` | The entire collection (directly against Qdrant) | No |

`purge` is the starting point for provisioning a new domain or for reproducible testing. It also removes untracked residue — manual uploads, interrupted mounts, traces of a previous cartridge. It cannot be undone.

---

## Network exposure

**The console serves HTTPS on its own, with no extra tooling.** `install.sh` generates a self-signed certificate (`ssl/selfsigned.crt`) and the console terminates TLS directly on its port (8443 by default). VPNs, overlay networks, and proxies are **optional** — none of them is a prerequisite for installing or running.

| Access route | What it needs | Note |
|---|---|---|
| Same host | Nothing | `https://localhost:8443/` (self-signed, so `curl -k`) |
| Direct on an internal network | Firewall rules | See the bind warning below |
| Reverse proxy | nginx, Caddy, etc. | Terminate outer TLS with a real certificate; disable upstream certificate verification since the inner one is self-signed |
| VPN / overlay network | That product | The console is not involved — you reach it over a private IP exactly as in "direct on an internal network" |

With several instances, split exposure **per instance port** (8443, 8444, …). If you use a proxy, **splitting by port or host is safer than by path** — the console assumes it is served at the root.

> ⚠️ **Bind warning (verified 2026-07-28)**: the default bind is **`0.0.0.0`** (uvicorn's default).
> The console therefore **answers on every interface the host has** — which is why private IPs work, and equally why a host on an untrusted network is exposed outright.
> Authentication is nothing more than the `api_keys/*.key` files, so **unless you are on a closed network, block the port at the firewall and let only the reverse proxy through.**
> There is not yet a config option to narrow the bind address to `127.0.0.1` (that is a behavior change, tracked separately — same root as S-8 "closed-network premise" in [security-review.md](../security-review.md)).

---

## GPU VRAM budget (measured 2026-07-27 · T2, RTX 5070 Ti, 16,303 MiB)

The first resource you run out of with multiple instances is neither ports nor disk — it is **VRAM**. Every number here was measured on that machine.

### The key property — VRAM is **fixed at load time**

Across six samples with two instances under simultaneous load (generation requests on both), total usage stayed **perfectly flat at 14,716 MiB**. Weights and KV cache are reserved at startup, which makes **budgeting deterministic** — you do not worry about peaks, only about whether it fits at load.

### Measured footprint of a single instance

| Configuration | Preset | Measured VRAM | Note |
| :-- | :-- | --: | :-- |
| Whole console (model + BGE-M3) | `gpt-oss-20b` (F16, `-ngl 99`, ctx 32768) | **13,238 MiB** | Spawns a separate `llama-server` process |
| Whole console (model + BGE-M3) | `llama31-8b-q4` (35 layers, ctx 16384) | **7,884 MiB** | llama-cpp-python **in-process** |
| Bare `llama-server` alone | Same model, layers, and ctx | **6,826 MiB** | Server only, no console |
| **BGE-M3's share** | fp16, dense + colbert | **≈ 1.0 – 1.4 GB** | See below |

> **BGE-M3's share, cross-checked two ways**: ① the difference in the table above (7,884 − 6,826 = **1,014 MiB**), and ② loading and encoding with BGE-M3 alone in a separate process (**+1,386 MiB**). The latter is larger because it **allocates its own CUDA context (~300 MiB)**; inside the console that context is shared, so treat the marginal cost as **≈ 1.0 GB per instance**.
> (The README's earlier "1.8GB" has been updated to match this measurement.)

### The budget formula

```
one instance ≈ (model weights + KV cache)  +  BGE-M3 ≈ 1.0GB  +  CUDA context ≈ 0.3GB
total GPU ≥ Σ(the above per instance) + 1GB headroom
```

Lowering `n_ctx` and `n_parallel` saves the most, because KV scales with context length × slot count and is reserved in full even while idle.

### How many fit on a single 16GB card (measured and derived)

| Combination | Total | On 16,303 MiB | Basis |
| :-- | --: | :-- | :-- |
| `llama31-8b-q4` × 2 | **14,716 MiB** | ✅ **Measured to fit** (1,587 MiB spare) | Console 7,840 + separate llama-server 6,826, flat even under simultaneous load. The second server generated at **152.9 tok/s** |
| `gpt-oss-20b` × 2 | ≈ 26,476 MiB | ❌ Does not fit | Measured 13,238 × 2 |
| `gpt-oss-20b` + `llama31-8b-q4` | ≈ 20,064 MiB | ❌ Does not fit | Sum of measurements |
| `llama31-8b-q4` × 3 | ≈ 21,500 MiB | ❌ Does not fit | 6,826 × 2 + 7,840 |

**Conclusion**: on a single 16GB card, **two 8B-class instances is the ceiling**. A 20B-class model should assume one GPU per instance.
To put three or more customers on one host, pick one of: ① a 24GB+ GPU, ② a GPU per instance (`CUDA_VISIBLE_DEVICES`), or ③ running some instances on the `api` tier (the OpenAI/Claude handlers) or `cpu-only`.

> **Unverified**: per-instance `CUDA_VISIBLE_DEVICES` separation could not be checked, as that machine has a single GPU.
> install.sh does not currently assign GPUs — on a multi-GPU host you configure it manually.

---

## Known constraints

- **Model weights are duplicated per instance.** If several instances use the same preset, symlink `models/` to a shared path (safe, since it is read-only).
- **The GPU is a shared resource.** N instances each loading a model costs N× the VRAM. Before running multiple GPU-preset instances on one host, work through the VRAM budget above.
- **Collection names are parameterized via `[qdrant] collection` (2026-07-28)** — see above. Give them different names and a single Qdrant can be shared.
- **Removing an instance is manual**: kill the processes → delete the clone → delete the registry line → delete the `~/.local/bin/aibotctl-<name>` symlink.

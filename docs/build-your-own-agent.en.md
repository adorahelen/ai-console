# Build your own agent — start to finish

> 🇰🇷 Korean original: [build-your-own-agent.md](build-your-own-agent.md)
>
> Assumes: [install.sh](../install.sh) has finished and both Qdrant and the console are running.
> If not → [README quick start](../README.en.md#-quick-start); install details in [install-paths.md](install-paths.md).

There are two paths to a domain agent. **If this is your first time, take path A (the wizard).**

---

## Path A — the onboarding wizard (browser, no files touched)

```
browser → https://localhost:8443/wizard
```

1. **Create the character** — describe in plain language what the agent is for (e.g. "a guide to our company's internal policies").
2. **Fill in the knowledge** — paste or upload your documents.
3. Conversion into the internal formats (intent prompts, qna YAML) is **done by the installed LLM itself** — the self-bootstrapping principle. You never need to learn the YAML schema.
4. **Save as a cartridge → launch.**

Design rationale: [onboarding-design.md](onboarding-design.md)

---

## Path B — handcraft a cartridge (file-based, full control)

### 1. Copy the template

```bash
cp -r cartridges/_template cartridges/my-domain
```

Open `cartridges/my-domain/cartridge.yaml` and fill in the name, description, and recommended model. That single file is your domain's manifest.

### 2. Write the prompts — this is the domain brain

Create at least two kinds under `prompts/`:

- **An intent classification prompt** — how user questions get sorted. The original SecOps cartridge used four: `QNA` (knowledge answers), `ACTION` (query generation), `PLAN` (multi-step plans), and `PLAYBOOK` (procedures). You design your own — a policy guide might do fine with just `QNA` and `FORM` (form guidance).
- **A system prompt per intent** — the generation rules the model follows within that intent.

For the base format, look at the existing files in `prompts/system/` (the engine's default prompts) and follow the same structure.

### 3. Fill in the knowledge — RAG documents

Create YAML files under `knowledge/`. There are two document types — full spec in [\_template/knowledge/README.md](../cartridges/_template/knowledge/README.md):

- `qna` — knowledge Q&A. `question` + `answer` + `aliases` (question variants — the key lever on retrieval recall).
- `action` — generating work products (queries, code, configuration). `question` + `cot` (generation rules) + `answer` (a correct example).

### 4. Pick a model

```bash
./install.sh --preset <key>     # a preset key from models.yaml
```

If you already installed, just adjust `config.ini [model] model=` and the llama-server settings to match the preset. For the tier guide see the comments in [models.yaml](../models.yaml) and the [model table in the README](../README.en.md#%EF%B8%8F-choosing-a-model--by-hardware-tier).

### 5. Mount — wire the config and upload the knowledge

**Wire the prompts**: in `config.ini [prompts]`, point the intent/qna/action keys at your cartridge's files:

```ini
[prompts]
intent = ./cartridges/my-domain/prompts/intent_classification.yaml
action = ./cartridges/my-domain/prompts/action.yaml
...
```

**Upload the knowledge** (the API key is the first line of a `*.key` file in `api_keys/`):

```bash
KEY=$(head -1 api_keys/*.key)
for f in cartridges/my-domain/knowledge/*.yaml; do
  curl -sk -X POST https://localhost:8443/api/ai/prompts/bulk \
    -H "Authorization: Bearer $KEY" \
    -F "files=@$f"
done
```

Uploaded YAML is embedded with BGE-M3 and stored in Qdrant. Knowledge takes effect the moment it lands; prompt wiring is applied at runtime by `aibotctl cartridge mount` (or the wizard), so no restart is needed — if the reload is not confirmed, `./run.sh restart` is the fallback.

### 6. Smoke test

Three kinds of question exercise the whole pipeline:

| Question | Expected |
| :-- | :-- |
| Something you put in the knowledge, asked verbatim | Classified into the right intent, and your uploaded knowledge retrieved as context and reflected in the answer |
| A domain question that is *not* in the knowledge | Intent still classified correctly, and the answer admits it has no grounds (this is your hallucination check) |
| Off-domain small talk | Handled by the default/fallback intent |

The full verification procedure (install → start → auth → wizard → RAG end-to-end) is in [testing-guide.md](testing-guide.md).

---

## A living reference

[cartridges/console-guide/](../cartridges/console-guide/) is a finished product that walks through everything in this document — a cartridge that explains the console's own usage. Its cartridge.yaml, qna prompt, eight knowledge entries, and mount commands are all there to copy from.

## Tips

- **Start with few intents.** Begin with two and split only when you see question types the classifier keeps getting wrong.
- **Invest in `aliases`.** Perceived RAG quality tracks how thoroughly you filled in question variants.
- **`cot` decides the quality of `action`.** The more explicitly you spell out generation rules step by step, the more stable the output gets — especially on small models.
- **No secrets in knowledge documents.** Get into the habit of scanning for internal hostnames, keys, and account patterns before uploading.

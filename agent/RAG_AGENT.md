# RAG explanation agent (WK4.1-WK4.4)

Turns a flagged bill into a plain-language explanation that cites the
specific rule it broke, e.g. "codes X and Y were billed together, but
NCCI edits indicate Y is bundled into X for this service date" (the
exact example from the WK4.3 plan).

## Architecture

```
rule_docs.py    -- WK4.1 support: builds ~60 LangChain Documents from the
                    Week 1 reference tables (NCCI edits, MUE caps) plus
                    hand-written concept documents for the two violation
                    types that have no per-code reference table.
build_index.py  -- WK4.1: embeds those documents (FastEmbed, local, free)
                    and persists them to a Chroma vector store on disk.
retrieve.py     -- WK4.2: given a flagged bill, finds the specific rule
                    document it violated.
explain.py      -- WK4.3: retrieved rule + bill facts -> a prompt ->
                    Azure OpenAI (LangChain) -> plain-language explanation.
config.py       -- loads Azure OpenAI settings from agent/.env (never
                    committed -- see .env.example).
```

**Framework note (WK4.4):** built in LangChain (`langchain`,
`langchain-openai`, `langchain-chroma`), not the Azure AI Foundry Agent
SDK used in the oncology project -- deliberately, so the portfolio shows
two different agent frameworks rather than the same one twice. Azure
OpenAI is still the model backend for generation (via
`langchain_openai.ChatOpenAI`, pointed at the Azure resource's v1 API
endpoint -- see `config.py`), since that's what an existing Azure AI
Foundry license already pays for -- LangChain is the orchestration layer,
Azure OpenAI is just the model underneath it, same as it would be
underneath the Foundry Agent SDK.

## Why local embeddings, hosted generation

Indexing (WK4.1) uses FastEmbed's `BAAI/bge-small-en-v1.5` -- a small
model that runs entirely on your machine, free, no API key. Generation
(WK4.3) uses Azure OpenAI -- a hosted, more capable model, but one that
costs per call. The split isn't arbitrary: the ~60 rule documents barely
ever change (only when a real NCCI/MUE update happens, per
`DRIFT_MONITORING.md`'s discussion of quarterly/annual code-set churn),
so there's no ongoing reason to pay a hosted API for embeddings computed
once and reused indefinitely. Generation happens per flagged bill, is
the part that actually benefits from a stronger model's fluency, and
is the part your Foundry license is meant to be used for.

## Why two different retrieval strategies

Not every violation type gets the same lookup, and that's deliberate:

- **unbundling / fragmented_billing** — the exact NCCI code pair or
  MUE-capped code is a fact this project can compute directly from the
  bill's own lines (reusing `generate_bills.py`'s own ground-truth
  checker, `violates_any_rule` — the same function that labeled the
  corpus). Once the exact code(s) are known, retrieval is an **exact
  metadata filter** on the vector store (`code1`/`code2`, or `code`) —
  not a similarity search hoping to land on the right document. This is
  still retrieval through the vector store, just the reliable kind:
  there's no reason to guess when you already know the answer.
- **causality_mismatch / upcoding** — there is no per-code reference row
  for either of these (no CMS table says "diagnosis region X doesn't
  belong with procedure region Y" — that's this project's own derived
  rule, not a real external rule text to cite). This is what the
  hand-written concept documents in `rule_docs.py` are for, and where
  semantic similarity search actually earns its place: matching the
  bill's specific region-mismatch or documentation-gap situation against
  the general principle (medical necessity, code-to-the-documentation)
  that best explains it.

## Why retrieval-then-generation, not just asking the model directly

The model is handed the specific retrieved rule text and specific bill
facts in the prompt, and told explicitly to use only those — not asked to
recall workers' comp coding rules from its own training data. That
matters here specifically: this project's reference tables are a curated
subset (documented honestly in `DATA_CARD.md`, not the full CMS
universe), so a model answering from general knowledge could easily cite
a real-sounding NCCI pair or MUE cap that isn't actually the one this
project's synthetic corpus is using, or could be behind on which codes
are current. Grounding the prompt in the exact retrieved document is what
keeps the explanation consistent with what this project actually checked
— the retrieval step exists to constrain the generation step, not just to
look sophisticated.

## What's been verified

Everything has now run for real, on a real machine with real network
access and a real Azure OpenAI deployment — not just the mechanical
stub-based wiring tests this project relied on early on (the sandbox
this was originally built in cannot reach huggingface.co, so FastEmbed's
model download couldn't be tested there; that's no longer a limitation,
just how the first draft was checked before a real run was possible):

- `rule_docs.py` — confirmed it builds all 60 documents with correct
  metadata (12 unbundling, 44 fragmented_billing, 2 causality_mismatch,
  2 upcoding).
- `build_index.py` — real run: FastEmbed downloaded `BAAI/bge-small-en-v1.5`
  (~67MB) and indexed all 60 documents into `agent/chroma_db/`.
- `retrieve.py` — real run against real bills from the corpus, real
  embeddings (no stub). The exact-metadata-filter lookups (unbundling,
  fragmented_billing) returned exact matches against the corpus's own
  ground-truth violation details (e.g. bill WC-BILL-003634's "billed
  97116 with 97110" matched a retrieved document with `code1=97116,
  code2=97110` exactly). More importantly, this also answered the open
  question about the semantic-search path: with only 2 hand-written
  concept documents each for causality_mismatch and upcoding, real
  `bge-small-en-v1.5` embeddings correctly ranked the right one first
  for both — the causality_mismatch query matched the region-consistency
  doc (not the unrelated upcoding one), and vice versa.
- `explain.py` — real run against the actual `gpt-5.6-luna` Azure OpenAI
  deployment, for all four violation types. Every explanation stayed
  grounded in the retrieved rule text and the bill's own facts (no
  invented codes, no outside-knowledge claims), matched the ground truth
  from `generate_bills.py`, and landed in the requested 2-3 sentence
  plain-language format.
- One real bug this caught before it ever ran for real: `retrieve.py`'s
  first draft built the `mue` lookup as a flat `{code: units}` dict, but
  `violates_any_rule()` (imported from `generate_bills.py`) expects
  `{code: {"units": ..., "rationale": ...}}` — a dict-shape mismatch that
  would have crashed on the very first bill. Caught by the mechanical
  test, fixed before this was ever handed over.
- A second real fix, caught only once real Azure resources existed: this
  project's Microsoft Foundry resource (created September 2026) issues a
  "v1" API endpoint (`.../openai/v1/`) rather than the older per-resource
  endpoint that takes an explicit `api_version` query parameter.
  `config.py` originally used `langchain_openai.AzureChatOpenAI`, which
  is built for that older pattern; it's been switched to plain
  `langchain_openai.ChatOpenAI` (`base_url` + `api_key`, no
  `api_version`), which is what the v1 endpoint actually expects.
  `agent/.env.example` documents this.

## Known, low-priority follow-up

`pip install` flags `langchain-community` (used for `FastEmbedEmbeddings`
in `build_index.py`/`retrieve.py`) as being sunset upstream in favor of
standalone integration packages. It still works correctly today (this is
what actually built the index and ran retrieval above), so there's no
functional problem — just a future migration worth doing if this project
gets revisited well after 2026.

## Setup

```bash
cd agent
cp .env.example .env
# edit .env with your own Azure OpenAI v1 endpoint, key, and chat
# deployment name -- from your Foundry project's Deployments page (click
# your chat deployment). .env is not committed and its values should
# never be pasted into chat. See .env.example for exactly where to find
# each value, and why the endpoint must be the .../openai/v1/ form.
python3 config.py          # confirms your .env is filled in correctly
```

## Running it

```bash
cd agent
python3 build_index.py     # WK4.1 -- one-time (or rerun after a rule-table update)
python3 retrieve.py        # WK4.2 -- prints retrieval results for one example bill per violation type, no Azure needed
python3 explain.py         # WK4.3 -- generates real explanations via Azure OpenAI
```

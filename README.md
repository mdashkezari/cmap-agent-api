[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19456049.svg)](https://doi.org/10.5281/zenodo.19456049)

# CMAP Agent (Agentic RAG) 

[https://agent.simonscmap.ai/docs](https://agent.simonscmap.ai/docs)

CMAP Agent is an agentic, retrieval-augmented interface for interacting with the Simons CMAP harmonized ocean and atmospheric data system.
The system enables users to express scientific intent in natural language and obtain verifiable results through a combination of
(i) semantic retrieval over a knowledge base derived from Simons CMAP catalog metadata and documentation,
(ii) tool-based execution for dataset discovery, data subsetting, visualization, and dataset colocalization, and (iii) optional emission of reproducible analysis code in the CMAP software ecosystem.

The service supports:
- Natural-language chat about CMAP datasets / variables / coverage / references
- Semantic search over a local knowledge base built from CMAP catalog metadata, dataset references, and full-text scientific literature from the reference bank
  - **Qdrant** (default for new installs): hybrid dense + BM25 sparse search via Reciprocal Rank Fusion — fixes keyword-mismatch failures in pure-dense retrieval
  - **ChromaDB** (legacy): dense-only cosine similarity
- Tool-calling to:
  - retrieve raw CMAP subsets via `pycmap` (returns Parquet by default; optional CSV)
  - generate custom static plots (PNG) including Cartopy maps
  - (optional) web search for general science context

It also:
- Persists conversation history + tool traces in a SQL Server schema (`agent.*`)
- Returns optional reproducible pycmap code snippets (when tools use pycmap) in the API response
  - set `options.return_code=true` in `/chat`


On each user turn the server:
1. Retrieves relevant context from the KB (datasets, variables, references)
2. Injects that context into the system prompt as primary context
3. Lets the agent decide whether to call tools (catalog lookup, data retrieval, plotting, web search)

The KB can be refreshed on demand:
- `cmap-agent-download-refs`  (download scientific papers and documents into the reference bank)
- `cmap-agent-sync-kb`        (rebuild/update the KB from live CMAP metadata + reference bank)

---

## Setup

### 1) Create the conda environment



```bash
conda env create -f environment.yml
conda activate cmap-agent
```

> The `environment.yml` file installs all dependencies needed to run the server + RAG + tools (FastAPI/uvicorn, Qdrant client, OpenAI/Anthropic clients, etc.). It also installs the geo stack (Cartopy + PROJ + GDAL) from `conda-forge` and installs this project in editable mode (`pip -e .`).

### 2) Apply SQL schema
Run once against the target database before first deployment:

```bash
# In Azure Data Studio or sqlcmd:
sql/agent_schema.sql
```

Creates: `agent.Threads` (with `AgentState`), `agent.Messages`, `agent.ToolRuns`, `agent.ThreadSummaries`.
Safe to re-run — all statements are idempotent. If upgrading from a pre-v0.2.55 install,
also run `sql/drop_agent_catalog_tables.sql` to remove the obsolete `agent.Catalog*` tables.

### 3) Configure environment variables

Create a `.env` file (or export env vars):

```bash
# SQL Server
export CMAP_SQLSERVER_HOST="..."
export CMAP_SQLSERVER_PORT="..."
export CMAP_SQLSERVER_DB="..."
export CMAP_SQLSERVER_USER="..."
export CMAP_SQLSERVER_PASSWORD="..."

# CMAP API key (used by pycmap + catalog sync)
export CMAP_API_KEY_FALLBACK="..."

# LLMs (OpenAI is needed for embeddings even if Anthropic is used for chat)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."   # optional

# Optional web search tool
export TAVILY_API_KEY="..."      # optional

# Knowledge Base backend: "qdrant" (recommended) or "chroma" (legacy)
export CMAP_AGENT_KB_BACKEND="qdrant"
export CMAP_AGENT_KB_COLLECTION="cmap_kb_v1"

# Qdrant connection (when KB_BACKEND=qdrant)
export QDRANT_URL="http://localhost:6333"   # local Docker
export QDRANT_API_KEY=""                     # empty for local; set for Qdrant Cloud

# ChromaDB settings (when KB_BACKEND=chroma, legacy)
export CMAP_AGENT_CHROMA_DIR="chroma"
```

### 4) Start Qdrant (dev)

When using the Qdrant backend (recommended), a local Qdrant instance is needed
for development.  The simplest approach is Docker:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

The `-v` flag persists the database to `./qdrant_data/` so data survives
container restarts.  The Qdrant dashboard is available at
`http://localhost:6333/dashboard`.

> **Note:** ChromaDB requires no separate server — its data is stored as local
> files in `CMAP_AGENT_CHROMA_DIR`.  If still using the legacy `chroma` backend,
> skip this step.


### 5) Build the KB

The KB is built from two sources: live CMAP metadata tables (`udfCatalog()` and
`tblDataset_References`) and the reference bank (`notrack/reference_bank/`).

**Download scientific references** (optional but recommended):
```bash
# Download all references (PDFs, GitHub READMEs, HTML docs)
cmap-agent-download-refs

# Or target specific datasets
cmap-agent-download-refs --dataset GRUMP --dataset HOT_Bottle_ALOHA
```

Documents can also be placed manually into `notrack/reference_bank/{Dataset_Short_Name}/`
— the sync will ingest whatever it finds regardless of how it arrived.

**Build / update the KB**:

The `--target` flag controls which backend to sync into.  If omitted, the
`CMAP_AGENT_KB_BACKEND` environment variable is used (default: `chroma`).

```bash
# Qdrant — full rebuild from scratch (recommended for first setup)
cmap-agent-sync-kb --target qdrant --rebuild

# Qdrant — incremental update (catalog + reference bank, delete stale docs)
cmap-agent-sync-kb --target qdrant --delete-stale

# Qdrant — catalog metadata only, skip reference bank
cmap-agent-sync-kb --target qdrant --skip-bank

# Quick dev test (10 datasets, no reference bank)
cmap-agent-sync-kb --target qdrant --rebuild --limit 10 --skip-bank

# ChromaDB (legacy) — same flags apply
cmap-agent-sync-kb --target chroma --rebuild
```

**Syncing to Qdrant Cloud (production)**:

Point the sync at the cloud cluster by setting connection env vars:

```bash
QDRANT_URL="https://<cluster>.aws.cloud.qdrant.io:6333" \
QDRANT_API_KEY="<api-key>" \
cmap-agent-sync-kb --target qdrant --rebuild
```

No Docker image rebuild or ECS redeployment is needed — the KB is a live service,
not a build artifact baked into the container.

---

## Run the API

```bash
uvicorn cmap_agent.server.app:app --reload --host 127.0.0.1 --port 8000
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Chat:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <CMAP_API_KEY>" \
  -d '{
    "user_id": 0,
    "message": "when does the MODIS chlorophyll dataset start?",
    "llm": {"provider":"openai","model":"gpt-4.1-mini"},
    "options": {"return_code": true, "max_tool_calls": 8}
  }'
```

Auth:
- If `CMAP_AGENT_AUTH_MODE=apikey` (or `auto` on ECS), `X-API-Key` is required on protected endpoints (e.g., `/chat`).
- `user_id` is optional when authenticating; if supplied, it must match the key's `User_ID` (unless `0`).

Artifacts:
- Parquet/CSV and PNG plots are written to `./artifacts/` and served at `/artifacts/...`

---

## Colocalization tool (pycmap.Sample)

The agent supports colocalization (aka integration / join) via `cmap.colocalize`, which is a thin wrapper around `pycmap.sample.Sample()`.

Conceptually:
- **Source**: a *small* point-based dataset (typically in-situ) with columns: `time`, `lat`, `lon`, and optional `depth`.
- **Targets**: one-or-more CMAP datasets to sample at each source point within per-target tolerances.

Tolerance meaning (per target):
- `dt_tol_days`: temporal tolerance in **days** (floats allowed).
- `lat_tol_deg`, `lon_tol_deg`: spatial tolerance in **degrees**.
- `depth_tol_m`: vertical tolerance in **meters** (use `0` for surface-only targets).

If the user does not specify tolerances, the assistant should propose reasonable defaults (e.g. for a daily 0.25° grid: `dt=1`, `lat=0.25`, `lon=0.25`).

Notes:
- This tool deliberately does **not** override pycmap server selection.
- The tool calls `Sample(..., replaceWithMonthlyClimatolog=False)`.

### Example (Swagger `/docs`)

To run a multi-turn chat in Swagger:
1) For the *first* request, omit `thread_id`.
2) In the response, copy the returned `thread_id`.
3) For follow-up requests, send the same `thread_id`.

Example user message:

> “Colocalize the 'C-MORE BiG-RAPA MV1015 Downcast Bottle Data' dataset with the daily 0.25° SST satellite dataset.”

The agent will:
1) Resolve source + target dataset tables/variables
2) Propose tolerances (if missing)
3) Call `cmap.colocalize`
4) Return a **CSV artifact by default** (or Parquet if requested) containing the original source rows plus appended target variables

---

## Notes

### Returning raw data
Data tools (`cmap.space_time`, `cmap.time_series`, `cmap.depth_profile`) return raw subsets, not aggregates, as Parquet by default.

### Returning pycmap code
When the agent uses pycmap-based tools, the API can return a `code` string containing the corresponding pycmap snippets:
- request: `options.return_code=true`
- response: `code` contains concatenated snippets (not executed)

### Updating the KB on demand
Any time datasets are added or updated in CMAP:
```bash
# Download new references, then sync
cmap-agent-download-refs
cmap-agent-sync-kb --target qdrant --delete-stale
```
To add a document not in the reference tables, place it in
`notrack/reference_bank/{Dataset_Short_Name}/` and re-run `cmap-agent-sync-kb`.
The download and sync steps are fully independent.

For production (Qdrant Cloud), set `QDRANT_URL` and `QDRANT_API_KEY` env vars
before running the sync — no container rebuild required.

### Regression testing (hybrid search validation)

A regression test script validates that the Qdrant hybrid search correctly
surfaces specific technical details that pure-dense ChromaDB retrieval missed.

**Mock mode** — no Qdrant server or OpenAI key needed.  Uses real fastembed
BM25 sparse vectors with fake dense embeddings.  The BM25 channel alone is the
hard gate (dense channel is random noise in this mode).

```bash
python scripts/test_qdrant_regression.py --mock
```

Expected output: `RESULT: ALL HARD-PASS TESTS PASSED ✓`

**Live mode** — requires a running Qdrant instance with a synced KB and a valid
`OPENAI_API_KEY`.  This is the definitive test confirming end-to-end retrieval.

```bash
python scripts/test_qdrant_regression.py
# or with a custom URL:
python scripts/test_qdrant_regression.py --url http://localhost:6333
```

The key test case is: *"minimum sequencing depth threshold used to filter GRUMP
samples"* — expected to return the chunk containing `5000` at rank 1.  This
query ranked #13 (below TOP\_K=8) with ChromaDB pure-dense retrieval.

---

---

## Knowledge base operations

The default KB backend is **Qdrant** (hybrid dense + BM25 search).
ChromaDB is kept as a supported fallback and is selected by setting
`CMAP_AGENT_KB_BACKEND=chroma`.  Both backends share the same sync
CLI and the same in-process KB protocol, so switching is a matter of
rebuilding the index against the alternate backend — no application
code changes are required.

### Qdrant lifecycle (local dev)

Start the container (one-time):

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$PWD/qdrant_data:/qdrant/storage" \
  qdrant/qdrant
```

Stop / start / remove:

```bash
docker stop qdrant
docker start qdrant
docker rm -f qdrant     # remove container (data in qdrant_data/ persists)
rm -rf qdrant_data      # wipe all data (requires a full rebuild after)
```

Inspect collections:

```bash
# List all collections
curl -s http://localhost:6333/collections | python -m json.tool

# Collection details (point count, vector config, payload index)
curl -s http://localhost:6333/collections/cmap_kb_v1 | python -m json.tool

# Delete a collection (reversible only by rebuild)
curl -X DELETE http://localhost:6333/collections/cmap_kb_v1
```

Sync the KB to Qdrant:

```bash
# Full rebuild (delete + re-create + re-index everything)
cmap-agent-sync-kb --target qdrant --rebuild

# Incremental update (add new docs, optionally drop stale ones)
cmap-agent-sync-kb --target qdrant --delete-stale

# Dry run — preview counts without writing anything
cmap-agent-sync-kb --target qdrant --dry-run
```

### Qdrant tuning knobs (no code change)

Two settings tune Qdrant behaviour without requiring a rebuild:

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_FUSION` | `rrf` | Fusion method combining dense + sparse channels. `rrf` (Reciprocal Rank Fusion) is rank-based and insensitive to score magnitudes. `dbsf` (Distribution-Based Score Fusion) normalizes score distributions before combining; sometimes helpful when one channel dominates. |
| `QDRANT_PREFETCH_FACTOR` | `3` | How many candidates per channel are fetched before fusion, as a multiple of TOP_K. Higher values give fusion more to work with at a small latency cost. |

These are query-time knobs.  Changing them only affects retrieval
behaviour and does not require re-indexing.  Measure changes against
the retrieval harness:

```bash
# Baseline
python scripts/eval_retrieval.py --backend qdrant --top-k 32

# Change a knob and re-run
QDRANT_FUSION=dbsf python scripts/eval_retrieval.py --backend qdrant --top-k 32
```

### Switching to ChromaDB (fallback)

To fall back to ChromaDB — for example, to A/B test backends or to
run without Qdrant available — set the backend and rebuild:

```bash
export CMAP_AGENT_KB_BACKEND=chroma
cmap-agent-sync-kb --target chroma --rebuild
```

ChromaDB persists to a local directory (`chroma/` by default, overridable
via `CMAP_AGENT_CHROMA_DIR`).  To reset:

```bash
rm -rf chroma              # wipe persisted index (requires rebuild)
```

ChromaDB is dense-only (cosine similarity).  It does not provide the
BM25 channel.  Expect a modest hit-rate loss on queries where keyword
anchors in the question overlap with the answer chunk (about 4 pp on
the current 108-query harness).  See the session handoff for the full
A/B result and trade-offs.

### Backend A/B

To measure the difference between backends on the same queries:

```bash
# 1. Both KBs must be populated from the same source data
cmap-agent-sync-kb --target chroma --rebuild
cmap-agent-sync-kb --target qdrant --rebuild

# 2. (Optional) confirm substrings round-trip through PDF extraction
python scripts/verify_eval_targets.py

# 3. Run the A/B — two runs per backend to capture tie-break variance
python scripts/ab_backend.py --top-k 32
```

The A/B output reports mean hit rate, stdev across runs, a per-shape
breakdown, and per-query disagreements where the two backends differ.
The "practical-effect" labels in the output are heuristics for the
operational decision ("is the gain big enough to act on?"), not a
statistical significance test.


## Project layout

```
src/cmap_agent/
  agent/              # tool-calling loop + system prompt
  llm/                # OpenAI / Anthropic adapters
  rag/                # KB backends (Qdrant hybrid / ChromaDB), embeddings, retrieval
  server/             # FastAPI app + request/response models
  storage/            # SQL Server persistence
  sync/               # KB sync, reference download, catalog sync
  tools/              # tool implementations (catalog, cmap, viz, kb, web)
sql/
  agent_schema.sql
environment.yml
pyproject.toml
```


## Docker

Build (Apple Silicon users may want `--platform linux/amd64` to match ECS):

```bash
docker build -t cmap-agent:geo .
# or:
docker build --platform linux/amd64 -t cmap-agent:geo .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="..." \
  -e CMAP_SQLSERVER_HOST="..." \
  -e CMAP_SQLSERVER_DB="..." \
  -e CMAP_SQLSERVER_USER="..." \
  -e CMAP_SQLSERVER_PASSWORD="..." \
  cmap-agent:geo
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

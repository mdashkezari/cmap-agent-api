[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19456049.svg)](https://doi.org/10.5281/zenodo.19456049)

# CMAP Agent (Agentic RAG) 

[https://agent.simonscmap.ai/docs](https://agent.simonscmap.ai/docs)

CMAP Agent is an agentic, retrieval-augmented interface for interacting with the Simons CMAP harmonized ocean and atmospheric data system.
The system enables users to express scientific intent in natural language and obtain verifiable results through a combination of
(i) semantic retrieval over a knowledge base derived from Simons CMAP catalog metadata and documentation,
(ii) tool-based execution for dataset discovery, data subsetting, visualization, and dataset colocalization, and (iii) optional emission of reproducible analysis code in the CMAP software ecosystem.

The service supports:
- Natural-language chat about CMAP datasets / variables / coverage / references
- Semantic search over a local knowledge base (Chroma) built from CMAP catalog metadata, dataset references, and full-text scientific literature from the reference bank
- Tool-calling to:
  - retrieve raw CMAP subsets via `pycmap` (returns Parquet by default; optional CSV)
  - generate custom static plots (PNG) including Cartopy maps
  - (optional) web search for general science context

It also:
- Persists conversation history + tool traces in a SQL Server schema (`agent.*`)
- Returns optional reproducible pycmap code snippets (when tools use pycmap) in the API response
  - set `options.return_code=true` in `/chat`


On each user turn the server:
1. Retrieves relevant context from the Chroma KB (datasets, variables, references)
2. Injects that context into the system prompt as primary context
3. Lets the agent decide whether to call tools (catalog lookup, data retrieval, plotting, web search)

The KB can be refreshed on demand:
- `cmap-agent-download-refs`  (download scientific papers and documents into the reference bank)
- `cmap-agent-sync-kb`        (rebuild/update the Chroma KB from live CMAP metadata + reference bank)

---

## Setup

### 1) Create the conda environment



```bash
conda env create -f environment.yml
conda activate cmap-agent
```

> The `environment.yml` file installs all dependencies needed to run the server + RAG + tools (FastAPI/uvicorn, Chroma, OpenAI/Anthropic clients, etc.). It also installs the geo stack (Cartopy + PROJ + GDAL) from `conda-forge` and installs this project in editable mode (`pip -e .`).

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

# LLMs (we can use OpenAI for embeddings even if we use Anthropic for chat)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."   # optional

# Optional web search tool
export TAVILY_API_KEY="..."      # optional

# Optional overrides
export CMAP_AGENT_CHROMA_DIR="chroma"
export CMAP_AGENT_KB_COLLECTION="cmap_kb_v1"
```

### 4) Build the KB

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
```bash
# Build / update Chroma KB (catalog + reference bank)
cmap-agent-sync-kb --delete-stale

# Full rebuild from scratch
cmap-agent-sync-kb --rebuild

# Catalog metadata only, skip reference bank
cmap-agent-sync-kb --skip-bank
```

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
cmap-agent-sync-kb --delete-stale
```
To add a document not in the reference tables, place it in
`notrack/reference_bank/{Dataset_Short_Name}/` and re-run `cmap-agent-sync-kb`.
The download and sync steps are fully independent.

---

## Project layout

```
src/cmap_agent/
  agent/              # tool-calling loop + system prompt
  llm/                # OpenAI / Anthropic adapters
  rag/                # Chroma KB + embeddings + retrieval formatting
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

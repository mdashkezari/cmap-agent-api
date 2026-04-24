from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # SQL Server
    CMAP_SQLSERVER_DSN: str | None = None
    CMAP_SQLSERVER_HOST: str | None = None
    CMAP_SQLSERVER_PORT: int = 1433
    CMAP_SQLSERVER_DB: str | None = None
    CMAP_SQLSERVER_USER: str | None = None
    CMAP_SQLSERVER_PASSWORD: str | None = None
    CMAP_SQLSERVER_DRIVER: str = "ODBC Driver 18 for SQL Server"
    CMAP_SQLSERVER_TRUST_CERT: bool = True

    # Agent artifacts
    CMAP_AGENT_ARTIFACT_DIR: str = "./artifacts"

    # Artifact publishing backend
    # - local: write to CMAP_AGENT_ARTIFACT_DIR and serve via /artifacts static mount
    # - s3: upload artifacts to S3 and return short-lived pre-signed URLs
    CMAP_AGENT_ARTIFACT_BACKEND: str = "local"
    CMAP_AGENT_ARTIFACT_S3_BUCKET: str | None = None
    CMAP_AGENT_ARTIFACT_S3_PREFIX: str = "artifacts"
    CMAP_AGENT_ARTIFACT_PRESIGN_TTL_SECONDS: int = 600
    CMAP_AGENT_ARTIFACT_S3_REGION: str | None = None

    # Upload settings (used by /files/presign_upload)
    CMAP_AGENT_UPLOAD_ALLOWED_EXTS: str = "csv,parquet"
    CMAP_AGENT_UPLOAD_MAX_MB: int = 200
    CMAP_AGENT_UPLOAD_PRESIGN_TTL_SECONDS: int = 3600
    # If True, keep a local copy after uploading to S3 (mainly for debugging).
    # Default False avoids filling ECS ephemeral storage.
    CMAP_AGENT_ARTIFACT_S3_KEEP_LOCAL_COPY: bool = False

    # Limits
    CMAP_AGENT_MAX_INLINE_ROWS: int = 5000
    CMAP_AGENT_MAX_EXPORT_ROWS: int = 5_000_000
    CMAP_AGENT_MAX_TOOL_CALLS: int = 6

    # Auth / Rate limiting
    # Auth modes:
    # - off: no authentication required
    # - apikey: require X-API-Key header, look up User_ID in CMAP SQL
    # - jwt: (reserved) require Authorization: Bearer <JWT>
    # - apikey_or_jwt: accept either
    # - auto: enable apikey when running in ECS; otherwise off
    CMAP_AGENT_AUTH_MODE: str = "auto"
    CMAP_AGENT_AUTH_APIKEY_HEADER: str = "X-API-Key"
    CMAP_AGENT_AUTH_APIKEY_TABLE: str = "dbo.tblAPI_keys"
    CMAP_AGENT_AUTH_APIKEY_COLUMN: str = "API_Key"
    CMAP_AGENT_AUTH_USERID_COLUMN: str = "User_ID"
    CMAP_AGENT_AUTH_CACHE_TTL_SECONDS: int = 600
    CMAP_AGENT_AUTH_PROTECT_DOCS: bool = False

    # Rate limiting (in-process). Note: if scale to multiple ECS tasks,
    # use a shared store (e.g., Redis/ElastiCache) for consistent limits.
    CMAP_AGENT_RATE_LIMIT_ENABLED: bool = True
    CMAP_AGENT_RATE_LIMIT_RPM: int = 60
    CMAP_AGENT_RATE_LIMIT_WINDOW_SECONDS: int = 60

    # CORS (for browser-based frontends)
    #
    # If host a separate web UI (e.g., https://simonscmap.ai), the browser
    # will preflight requests to https://agent.simonscmap.ai. Enable CORS to
    # allow those origins.
    #
    # NOTE: This is not a security boundary for non-browser clients; it only
    # controls which browser origins are permitted to call the API.
    CMAP_AGENT_CORS_ENABLED: bool = True
    # Comma-separated list of allowed origins.
    CMAP_AGENT_CORS_ALLOW_ORIGINS: str = "https://simonscmap.ai,https://chat.simonscmap.ai"
    CMAP_AGENT_CORS_ALLOW_CREDENTIALS: bool = False
    # Comma-separated list. Defaults cover typical SPA fetch() usage.
    CMAP_AGENT_CORS_ALLOW_METHODS: str = "GET,POST,OPTIONS"
    CMAP_AGENT_CORS_ALLOW_HEADERS: str = "*"
    CMAP_AGENT_CORS_MAX_AGE_SECONDS: int = 600

    # CMAP API key fallback (if not loading from CMAP Users tables yet)
    CMAP_API_KEY_FALLBACK: str | None = None

    # LLM keys
    OPENAI_API_KEY: str | None = None
    # Optional override (useful for proxies/self-hosted gateways)
    OPENAI_BASE_URL: str | None = None
    OPENAI_ORG: str | None = None
    OPENAI_PROJECT: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    # Web search (optional)
    TAVILY_API_KEY: str | None = None

    # RAG / Knowledge Base
    # Backend selector: "chroma" (legacy) or "qdrant" (hybrid search)
    CMAP_AGENT_KB_BACKEND: str = "qdrant"
    CMAP_AGENT_KB_COLLECTION: str = "cmap_kb_v1"
    # Top-K chunks returned by KB search.  Raised from 16 to 32 in v228
    # based on the 108-query retrieval harness: hit rate at top-16 was
    # 92.6%, at top-24 was 93.5%, at top-32 was 95.4%.  The top-32
    # setting closes the full gap between "retrievable within scan
    # depth 40" and "surfaced to the LLM."  Rescued queries are
    # concentrated in the sequence_literal / numeric_threshold /
    # numeric_count shape classes — exactly the factual-technical
    # queries (primer sequences, minimum depth thresholds) where
    # real-world agent failures were observed.  No harness regressions
    # across any shape class or source paper.
    CMAP_AGENT_KB_TOP_K: int = 32

    # Reference bank PDF chunk size (chars).  Smaller chunks improve BM25
    # term density — the target sentence dominates a 2000-char chunk but is
    # diluted in a 7000-char chunk.  Dataset/variable docs are short enough
    # that their 7000-char limit is rarely reached.
    CMAP_AGENT_KB_REFBANK_CHUNK_SIZE: int = 2_000
    CMAP_AGENT_KB_CATALOG_CHUNK_SIZE: int = 7_000

    # ChromaDB settings (used when KB_BACKEND=chroma)
    CMAP_AGENT_CHROMA_DIR: str = "chroma"

    # Qdrant settings (used when KB_BACKEND=qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    # Dense vector dimension (must match embedding model output)
    QDRANT_DENSE_DIM: int = 1536
    # Hybrid fusion method: "rrf" (Reciprocal Rank Fusion, default — fuses
    # rank positions, insensitive to raw score magnitudes) or "dbsf"
    # (Distribution-Based Score Fusion — normalizes the dense and sparse
    # score distributions before adding them, which can help when the two
    # channels operate at very different score scales).  This is a
    # per-query tuning knob and does not require a KB rebuild; changing
    # it only affects how retrieval candidates from the two channels
    # are combined at query time.
    QDRANT_FUSION: str = "rrf"
    # Over-fetch factor per channel before fusion.  Qdrant fetches
    # ``top_k * QDRANT_PREFETCH_FACTOR`` candidates from each of the
    # dense and sparse channels before fusing them.  Larger values give
    # fusion more to work with at the cost of a modestly slower query.
    QDRANT_PREFETCH_FACTOR: int = 3
    # Upsert batch size (number of points per single upsert request to
    # Qdrant).  Smaller batches reduce the per-request payload sent to the
    # cloud proxy, which can avoid 502 Bad Gateway responses on managed
    # tiers that have proxy-level request-size or request-time limits.
    # Larger batches reduce overhead on cleanly-behaved clusters.  The
    # default of 64 matches the in-process value used through v225 and
    # was measured working on Qdrant Cloud Starter tier with retry.
    QDRANT_UPSERT_BATCH_SIZE: int = 64
    # Maximum retry attempts on transient proxy-layer errors (502/503/504)
    # during upsert.  The client retries with exponential backoff
    # (1s, 2s, 4s, 8s, 16s, capped at 30s) between attempts.  Terminal
    # failures after all retries are re-raised so the caller can abort.
    # Set to 0 to disable retry entirely (not recommended — see the v202
    # cutover notes on Qdrant Cloud proxy behaviour during bulk ingestion).
    QDRANT_UPSERT_MAX_RETRIES: int = 5

    # Embeddings (used for KB build + retrieval)
    CMAP_EMBEDDINGS_PROVIDER: str = "openai"  # openai
    CMAP_EMBEDDINGS_MODEL: str = "text-embedding-3-small"


settings = Settings()

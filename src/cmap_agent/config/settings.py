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

    # RAG / Knowledge Base (Chroma)
    CMAP_AGENT_CHROMA_DIR: str = "chroma"
    CMAP_AGENT_KB_COLLECTION: str = "cmap_kb_v1"
    CMAP_AGENT_KB_TOP_K: int = 8

    # Embeddings (used for KB build + retrieval)
    CMAP_EMBEDDINGS_PROVIDER: str = "openai"  # openai
    CMAP_EMBEDDINGS_MODEL: str = "text-embedding-3-small"


settings = Settings()

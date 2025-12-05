"""Application configuration using Pydantic Settings - Fully environment-based."""
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with complete .env support and validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # APPLICATION
    # -------------------------------------------------------------------------
    app_name: str = Field(default="RAG File Assistant")
    app_version: str = Field(default="0.2.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # -------------------------------------------------------------------------
    # FILE STORAGE
    # -------------------------------------------------------------------------
    files_root_path: Path = Field(
        default=Path("./data/files"),
        description="Root directory for file storage"
    )
    max_file_size_mb: int = Field(default=500, ge=1, le=5000)
    supported_extensions: str = Field(
        default=".pdf,.docx,.txt,.csv,.xlsx,.md,.json",
        description="Comma-separated file extensions"
    )

    # -------------------------------------------------------------------------
    # GOOGLE GEMINI AI
    # -------------------------------------------------------------------------
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_embedding_model: str = Field(default="models/text-embedding-004")
    embedding_dimensions: int = Field(default=768)
    gemini_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    gemini_max_tokens: int = Field(default=4096, ge=512, le=32768)
    gemini_timeout: int = Field(default=120, ge=10, le=300)

    # -------------------------------------------------------------------------
    # VECTOR DATABASE (QDRANT)
    # -------------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_collection_name: str = Field(default="file_chunks")
    vector_size: int = Field(default=768, ge=128, le=2048)

    # -------------------------------------------------------------------------
    # POSTGRESQL DATABASE
    # -------------------------------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = Field(default="rag_assistant")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="changeme")

    # -------------------------------------------------------------------------
    # REDIS CACHE
    # -------------------------------------------------------------------------
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_password: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # CHUNKING STRATEGY
    # -------------------------------------------------------------------------
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=512)
    min_chunk_size: int = Field(default=10, ge=5, le=100)

    # -------------------------------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------------------------------
    top_k_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank_top_k: int = Field(default=5, ge=1, le=50)

    # -------------------------------------------------------------------------
    # PROCESSING
    # -------------------------------------------------------------------------
    batch_size: int = Field(default=20, ge=1, le=100)
    max_workers: int = Field(default=4, ge=1, le=16)
    max_concurrent_ingestion: int = Field(default=5, ge=1, le=20)

    # -------------------------------------------------------------------------
    # TELEGRAM BOT
    # -------------------------------------------------------------------------
    telegram_bot_token: str = Field(default="")
    telegram_allowed_users: str = Field(default="")
    telegram_max_voice_size_mb: int = Field(default=20, ge=1, le=50)

    # -------------------------------------------------------------------------
    # WHISPER (VOICE TRANSCRIPTION)
    # -------------------------------------------------------------------------
    whisper_model: str = Field(default="base")
    whisper_device: str = Field(default="cpu")
    whisper_compute_type: str = Field(default="int8")

    # -------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------
    agent_max_iterations: int = Field(default=10, ge=1, le=50)
    agent_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    agent_thinking_enabled: bool = Field(default=True)

    # -------------------------------------------------------------------------
    # MEMORY
    # -------------------------------------------------------------------------
    memory_max_messages: int = Field(default=20, ge=5, le=100)
    memory_summary_threshold: int = Field(default=10, ge=5, le=50)

    # -------------------------------------------------------------------------
    # RATE LIMITING
    # -------------------------------------------------------------------------
    rate_limit_messages: int = Field(default=20, ge=1, le=1000)
    rate_limit_voice: int = Field(default=5, ge=1, le=100)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    # -------------------------------------------------------------------------
    # MONITORING
    # -------------------------------------------------------------------------
    sentry_dsn: Optional[str] = Field(default=None)
    enable_performance_monitoring: bool = Field(default=False)

    # -------------------------------------------------------------------------
    # VALIDATORS
    # -------------------------------------------------------------------------
    @classmethod
    def validate_files_path(cls, v: Path) -> Path:
        """Ensure files path exists."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v.resolve()

    @classmethod
    def validate_extensions(cls, v: str) -> list[str]:
        """Parse and validate file extensions."""
        if isinstance(v, list):
            return v
        return [ext.strip().lower() for ext in v.split(",") if ext.strip()]

    @classmethod
    def validate_gemini_key(cls, v: str) -> str:
        """Validate Gemini API key is set (in production)."""
        # Warning only, not blocking
        if not v and not cls.model_config.get("debug", False):
            import warnings
            warnings.warn("GEMINI_API_KEY not set - AI features will not work!")
        return v

    @classmethod
    def validate_telegram_token(cls, v: str) -> str:
        """Validate Telegram bot token format."""
        if v and ":" not in v:
            raise ValueError("Invalid Telegram bot token format")
        return v

    # -------------------------------------------------------------------------
    # COMPUTED PROPERTIES
    # -------------------------------------------------------------------------
    @property
    def database_url(self) -> str:
        """PostgreSQL async connection string."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """PostgreSQL sync connection string (for Alembic)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Redis connection string."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def allowed_user_ids(self) -> list[str]:
        """Parse allowed Telegram user IDs."""
        if not self.telegram_allowed_users:
            return []
        return [uid.strip() for uid in self.telegram_allowed_users.split(",") if uid.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug

    def get_supported_extensions_list(self) -> list[str]:
        """Get list of supported file extensions."""
        if isinstance(self.supported_extensions, list):
            return self.supported_extensions
        return [ext.strip().lower() for ext in self.supported_extensions.split(",")]


# -------------------------------------------------------------------------
# GLOBAL SETTINGS INSTANCE
# -------------------------------------------------------------------------
settings = Settings()
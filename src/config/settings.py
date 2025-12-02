"""Application configuration using Pydantic Settings - Phase 2."""
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "RAG File Assistant"
    app_version: str = "0.2.0"
    debug: bool = False
    log_level: str = "INFO"

    # File Storage
    files_root_path: Path = Field(default=Path("C:/Users/yakuz/PycharmProjects/RAG-File_Assistant/data/files"))
    max_file_size_mb: int = 500
    supported_extensions: list[str] = Field(
        default=[".pdf", ".docx", ".txt", ".csv", ".xlsx", ".md", ".json"]
    )

    # Ollama
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.1
    max_tokens: int = 4096

    # Qdrant Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "file_chunks"
    vector_size: int = 768

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_assistant"
    postgres_user: str = "postgres"
    postgres_password: str = "root"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Chunking Strategy
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 10

    # Retrieval
    top_k_results: int = 10
    similarity_threshold: float = 0.7
    rerank_top_k: int = 5

    # Processing
    batch_size: int = 20
    max_workers: int = 4

    # Telegram Bot (Phase 2)
    telegram_bot_token: str = ""
    telegram_allowed_users: str = ""  # Comma-separated user IDs
    telegram_max_voice_size_mb: int = 20

    # Whisper (Voice Transcription)
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Celery
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None

    # Rate Limiting
    rate_limit_messages: int = 20
    rate_limit_voice: int = 5

    # Agent
    agent_max_iterations: int = 10
    agent_temperature: float = 0.1
    agent_thinking_enabled: bool = True

    # Memory
    memory_max_messages: int = 20
    memory_summary_threshold: int = 10

    # Sentry (optional)
    sentry_dsn: Optional[str] = None

    @classmethod
    def validate_files_path(cls, v: Path) -> Path:
        """Ensure files path exists."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v.resolve()

    @property
    def database_url(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Synchronous PostgreSQL connection string."""
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
    def celery_broker(self) -> str:
        """Celery broker URL."""
        return self.celery_broker_url or self.redis_url

    @property
    def celery_backend(self) -> str:
        """Celery result backend URL."""
        return self.celery_result_backend or self.redis_url

    @property
    def allowed_user_ids(self) -> List[str]:
        """Parse allowed user IDs."""
        if not self.telegram_allowed_users:
            return []
        return [uid.strip() for uid in self.telegram_allowed_users.split(",") if uid.strip()]


# Global settings instance
settings = Settings()
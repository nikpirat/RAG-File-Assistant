"""SQLAlchemy database models."""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, BigInteger, DateTime, Text, Index, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class FileMetadata(Base):
    """Store file metadata for filtering and tracking."""

    __tablename__ = "file_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    file_name: Mapped[str] = mapped_column(String(512), index=True)
    file_type: Mapped[str] = mapped_column(String(50), index=True)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, index=True)

    # Content metadata
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timestamps
    file_created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    file_modified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc), index=True)
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc)
    )

    # Full-text search
    search_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)

    # Additional metadata as JSON
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Processing status
    is_indexed: Mapped[bool] = mapped_column(default=False, index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_file_metadata_search", "search_vector", postgresql_using="gin"),
        Index("ix_file_metadata_size_type", "file_size_bytes", "file_type"),
    )


class DocumentChunk(Base):
    """Store document chunks for tracking (not the embeddings)."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_metadata_id: Mapped[int] = mapped_column(Integer, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)

    # Chunk content
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    # Chunk metadata
    chunk_type: Mapped[str] = mapped_column(String(50))  # paragraph, table, code, etc.
    token_count: Mapped[int] = mapped_column(Integer)

    # Position in document
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Vector DB reference
    vector_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_chunk_file_index", "file_metadata_id", "chunk_index"),
    )


class ChatHistory(Base):
    """Store chat conversations for memory."""

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    chat_id: Mapped[str] = mapped_column(String(128), index=True)

    # Message
    role: Mapped[str] = mapped_column(String(20))  # user, assistant
    content: Mapped[str] = mapped_column(Text)

    # Metadata
    message_type: Mapped[str] = mapped_column(String(50))  # text, voice, command
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Context
    retrieved_files: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_chat_user_created", "user_id", "created_at"),
    )


class IndexingJob(Base):
    """Track indexing jobs for monitoring."""

    __tablename__ = "indexing_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_type: Mapped[str] = mapped_column(String(50))  # full, incremental, single_file
    status: Mapped[str] = mapped_column(String(20), index=True)  # pending, running, completed, failed

    # Progress tracking
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed_files: Mapped[int] = mapped_column(Integer, default=0)
    failed_files: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Results
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_summary: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
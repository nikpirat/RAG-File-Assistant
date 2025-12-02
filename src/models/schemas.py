"""Pydantic schemas for data validation."""
from datetime import datetime
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator, field_validator


class FileMetadataSchema(BaseModel):
    """File metadata schema."""

    file_path: str
    file_name: str
    file_type: str
    file_size_bytes: int
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    file_created_at: Optional[datetime] = None
    file_modified_at: Optional[datetime] = None
    extra_metadata: Optional[dict] = None

    class Config:
        from_attributes = True


class ChunkMetadata(BaseModel):
    """Metadata stored with each chunk in vector DB."""

    file_path: str
    file_name: str
    file_type: str
    file_size_bytes: int
    chunk_index: int
    chunk_type: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    file_modified_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentChunkSchema(BaseModel):
    """Document chunk with content and metadata."""

    content: str
    metadata: ChunkMetadata
    token_count: int
    chunk_index: int

    @field_validator("content")
    def validate_content(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    """Single search result."""

    chunk_id: str
    content: str
    score: float
    metadata: ChunkMetadata

    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    """Search request parameters."""

    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Filters
    file_types: Optional[list[str]] = None
    file_name_pattern: Optional[str] = None
    min_file_size: Optional[int] = None
    max_file_size: Optional[int] = None
    modified_after: Optional[datetime] = None
    modified_before: Optional[datetime] = None


class SearchResponse(BaseModel):
    """Search response with results."""

    query: str
    results: list[SearchResult]
    total_results: int
    search_time_ms: float


class IngestionProgress(BaseModel):
    """Track ingestion progress."""

    total_files: int
    processed_files: int
    failed_files: int
    current_file: Optional[str] = None
    status: str
    started_at: datetime
    estimated_completion: Optional[datetime] = None
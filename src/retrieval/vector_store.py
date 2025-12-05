"""Qdrant vector store operations."""
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchParams,
    PayloadSchemaType,
)
from src.config.settings import settings
from src.config.logging_config import get_logger
from src.models.schemas import DocumentChunkSchema, SearchResult, ChunkMetadata

logger = get_logger(__name__)


class VectorStore:
    """Qdrant vector store for document embeddings."""

    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.vector_size

        logger.info(
            "vector_store_initialized",
            host=settings.qdrant_host,
            collection=self.collection_name,
        )

    async def create_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            # FIXED: Correct method is collection_exists()
            if await self.client.collection_exists(self.collection_name):
                logger.info("collection_exists", collection=self.collection_name)
                return

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            logger.info("collection_created", collection=self.collection_name)
            await self._create_payload_indexes()

        except Exception as e:
            logger.error("collection_creation_failed", error=str(e))
            raise

    async def _create_payload_indexes(self) -> None:
        """Create indexes on payload fields for faster filtering."""
        indexes = [
            ("file_path", PayloadSchemaType.KEYWORD),
            ("file_name", PayloadSchemaType.KEYWORD),
            ("file_type", PayloadSchemaType.KEYWORD),
            ("file_size_bytes", PayloadSchemaType.INTEGER),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("page_number", PayloadSchemaType.INTEGER),
        ]

        for field_name, field_type in indexes:
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.info("payload_index_created", field=field_name)
            except Exception as e:
                logger.warning("payload_index_creation_failed", field=field_name, error=str(e))

    async def upsert_chunks(
        self,
        chunks: List[DocumentChunkSchema],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Insert or update chunks with embeddings."""
        logger.info("upserting_chunks", count=len(chunks))

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count mismatch")

        points = []
        chunk_ids = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            payload = {
                "content": chunk.content,
                "file_path": chunk.metadata.file_path,
                "file_name": chunk.metadata.file_name,
                "file_type": chunk.metadata.file_type,
                "file_size_bytes": chunk.metadata.file_size_bytes,
                "chunk_index": chunk.metadata.chunk_index,
                "chunk_type": chunk.metadata.chunk_type,
                "token_count": chunk.token_count,
            }

            if chunk.metadata.page_number is not None:
                payload["page_number"] = chunk.metadata.page_number
            if chunk.metadata.section_title:
                payload["section_title"] = chunk.metadata.section_title
            if chunk.metadata.file_modified_at:
                payload["file_modified_at"] = chunk.metadata.file_modified_at.isoformat()

            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info("chunks_upserted", count=len(points))
        return chunk_ids

    async def search(
            self,
            query_embedding: List[float],
            top_k: int = settings.top_k_results,
            score_threshold: float = settings.similarity_threshold,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks using query_points."""
        logger.info(
            "searching_vectors",
            top_k=top_k,
            threshold=score_threshold,
            has_filters=filters is not None,
        )

        query_filter = self._build_filter(filters) if filters else None

        # FIXED: query_points() uses query_filter, not filter
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,  # CHANGED: filter -> query_filter
            search_params=SearchParams(
                hnsw_ef=128,
                exact=False,
            ),
        )

        results = []
        for point in response.points:
            metadata = ChunkMetadata(
                file_path=point.payload["file_path"],
                file_name=point.payload["file_name"],
                file_type=point.payload["file_type"],
                file_size_bytes=point.payload["file_size_bytes"],
                chunk_index=point.payload["chunk_index"],
                chunk_type=point.payload.get("chunk_type", "text"),
                page_number=point.payload.get("page_number"),
                section_title=point.payload.get("section_title"),
                file_modified_at=datetime.fromisoformat(point.payload["file_modified_at"])
                if "file_modified_at" in point.payload
                else None,
            )

            results.append(
                SearchResult(
                    chunk_id=str(point.id),
                    content=point.payload["content"],
                    score=point.score,
                    metadata=metadata,
                )
            )

        logger.info("search_complete", results_count=len(results))
        return results

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dict."""
        conditions = []

        if "file_types" in filters and filters["file_types"]:
            conditions.append(
                FieldCondition(
                    key="file_type",
                    match=MatchValue(any=filters["file_types"]),
                )
            )

        if "file_name_pattern" in filters and filters["file_name_pattern"]:
            conditions.append(
                FieldCondition(
                    key="file_name",
                    match=MatchValue(value=filters["file_name_pattern"]),
                )
            )

        if "min_file_size" in filters or "max_file_size" in filters:
            conditions.append(
                FieldCondition(
                    key="file_size_bytes",
                    range=Range(
                        gte=filters.get("min_file_size"),
                        lte=filters.get("max_file_size"),
                    ),
                )
            )

        return Filter(must=conditions) if conditions else None

    async def delete_by_file_path(self, file_path: str) -> None:
        """Delete all chunks for a specific file."""
        logger.info("deleting_chunks_by_file", file_path=file_path)

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path),
                    )
                ]
            ),
        )

        logger.info("chunks_deleted", file_path=file_path)

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = await self.client.get_collection(self.collection_name)
        return {
            "total_points": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }

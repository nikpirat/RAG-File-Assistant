"""Tools for the agent."""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.config.logging_config import get_logger
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.models.database import FileMetadata

logger = get_logger(__name__)


class AgentTools:
    """Collection of tools for the agent."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.session = session
        self.embedding_service = embedding_service
        self.vector_store = vector_store

        logger.info("agent_tools_initialized")

    async def search_files(
        self,
        query: str,
        top_k: int = 5,
        file_types: Optional[List[str]] = None,
    ) -> str:
        """
        Search for files by semantic similarity to query.
        Returns relevant file content and metadata.
        """
        logger.info("tool_search_files", query=query, top_k=top_k)

        try:
            # Generate embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Build filters
            filters = {}
            if file_types:
                filters["file_types"] = file_types

            # Search
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=0.5,
                filters=filters if filters else None,
            )

            if not results:
                return "No relevant files found for your query."

            # Format results
            output_parts = [f"Found {len(results)} relevant results:\n"]

            for i, result in enumerate(results, 1):
                output_parts.append(f"\n--- Result {i} ---")
                output_parts.append(f"File: {result.metadata.file_name}")
                output_parts.append(f"Type: {result.metadata.file_type}")
                output_parts.append(f"Relevance: {result.score:.2%}")
                output_parts.append(f"Content: {result.content[:300]}...")

                if result.metadata.page_number:
                    output_parts.append(f"Page: {result.metadata.page_number}")

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("search_files_failed", error=str(e))
            return f"Error searching files: {str(e)}"

    async def list_files(
        self,
        pattern: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """
        List files in the system with optional filters.
        Returns file names, types, sizes, and dates.
        """
        logger.info("tool_list_files", pattern=pattern, file_type=file_type)

        try:
            query = select(FileMetadata).where(FileMetadata.is_indexed == True)

            if pattern:
                query = query.where(FileMetadata.file_name.ilike(f"%{pattern}%"))

            if file_type:
                query = query.where(FileMetadata.file_type == file_type)

            query = query.limit(limit)

            result = await self.session.execute(query)
            files = result.scalars().all()

            if not files:
                return "No files found matching your criteria."

            output_parts = [f"Found {len(files)} files:\n"]

            for file in files:
                size_mb = file.file_size_bytes / (1024 * 1024)
                output_parts.append(
                    f"📄 {file.file_name} ({file.file_type}) - "
                    f"{size_mb:.2f}MB - "
                    f"Modified: {file.file_modified_at.strftime('%Y-%m-%d') if file.file_modified_at else 'Unknown'}"
                )

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("list_files_failed", error=str(e))
            return f"Error listing files: {str(e)}"

    async def get_file_stats(
        self,
        file_type: Optional[str] = None,
    ) -> str:
        """
        Get statistics about indexed files.
        Returns counts, sizes, and types.
        """
        logger.info("tool_file_stats", file_type=file_type)

        try:
            # Total files
            query = select(func.count(FileMetadata.id)).where(
                FileMetadata.is_indexed == True
            )
            if file_type:
                query = query.where(FileMetadata.file_type == file_type)

            result = await self.session.execute(query)
            total_files = result.scalar()

            # Total size
            query = select(func.sum(FileMetadata.file_size_bytes)).where(
                FileMetadata.is_indexed == True
            )
            if file_type:
                query = query.where(FileMetadata.file_type == file_type)

            result = await self.session.execute(query)
            total_bytes = result.scalar() or 0
            total_gb = total_bytes / (1024 ** 3)

            # By type
            query = select(
                FileMetadata.file_type,
                func.count(FileMetadata.id).label("count"),
                func.sum(FileMetadata.file_size_bytes).label("size"),
            ).where(
                FileMetadata.is_indexed == True
            ).group_by(FileMetadata.file_type)

            result = await self.session.execute(query)
            by_type = result.all()

            output_parts = [
                f"📊 File Statistics:",
                f"Total files: {total_files}",
                f"Total size: {total_gb:.2f} GB\n",
                "By type:",
            ]

            for row in by_type:
                type_gb = (row.size or 0) / (1024 ** 3)
                output_parts.append(
                    f"  {row.file_type}: {row.count} files, {type_gb:.2f} GB"
                )

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("file_stats_failed", error=str(e))
            return f"Error getting file stats: {str(e)}"
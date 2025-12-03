"""Enhanced tools for the agent with file sending and precise search."""
from typing import List, Optional, Dict, Any
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.config.logging_config import get_logger
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.models.database import FileMetadata

logger = get_logger(__name__)


class AgentTools:
    """Collection of enhanced tools for the agent."""

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
    ) -> Dict[str, Any]:
        """
        Search for files by semantic similarity to query.
        Returns relevant file content, metadata, AND file paths for sending.
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
                score_threshold=0.4,  # Lower threshold for better recall
                filters=filters if filters else None,
            )

            if not results:
                return {
                    "text": "No relevant files found for your query.",
                    "files": []
                }

            # Format results with file info for sending
            output_parts = [f"Found {len(results)} relevant results:\n"]
            files_info = []

            for i, result in enumerate(results, 1):
                output_parts.append(f"\n--- Result {i} ---")
                output_parts.append(f"File: {result.metadata.file_name}")
                output_parts.append(f"Type: {result.metadata.file_type}")
                output_parts.append(f"Relevance: {result.score:.2%}")
                output_parts.append(f"Content: {result.content[:400]}...")

                if result.metadata.page_number:
                    output_parts.append(f"Page: {result.metadata.page_number}")

                # Collect file info for potential sending
                files_info.append({
                    "path": result.metadata.file_path,
                    "name": result.metadata.file_name,
                    "type": result.metadata.file_type,
                    "score": result.score,
                    "content": result.content
                })

            return {
                "text": "\n".join(output_parts),
                "files": files_info
            }

        except Exception as e:
            logger.error("search_files_failed", error=str(e))
            return {
                "text": f"Error searching files: {str(e)}",
                "files": []
            }

    async def find_specific_content(
        self,
        query: str,
        file_name: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Find specific content like 'question 8' or specific text in files.
        More precise search for exact information retrieval.
        """
        logger.info("tool_find_specific_content", query=query, file_name=file_name)

        try:
            # Generate embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Build filters
            filters = {}
            if file_name:
                # Search for file by name pattern
                result = await self.session.execute(
                    select(FileMetadata).where(
                        FileMetadata.file_name.ilike(f"%{file_name}%"),
                        FileMetadata.is_indexed == True
                    ).limit(1)
                )
                file_meta = result.scalar_one_or_none()

                if file_meta:
                    filters["file_path"] = file_meta.file_path

            # Search with high precision
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=0.3,  # Even lower for specific searches
                filters=filters if filters else None,
            )

            if not results:
                return {
                    "text": f"Could not find specific content matching: {query}",
                    "files": []
                }

            # Return most relevant chunk
            best_result = results[0]

            output_parts = [
                f"📄 Found in: {best_result.metadata.file_name}",
                f"Relevance: {best_result.score:.2%}",
            ]

            if best_result.metadata.page_number:
                output_parts.append(f"Page: {best_result.metadata.page_number}")

            output_parts.append(f"\nContent:\n{best_result.content}")

            # If there are more results, mention them
            if len(results) > 1:
                output_parts.append(
                    f"\n(Found {len(results)-1} more related sections in this file)"
                )

            return {
                "text": "\n".join(output_parts),
                "files": [{
                    "path": best_result.metadata.file_path,
                    "name": best_result.metadata.file_name,
                    "type": best_result.metadata.file_type,
                    "score": best_result.score,
                    "content": best_result.content
                }]
            }

        except Exception as e:
            logger.error("find_specific_content_failed", error=str(e))
            return {
                "text": f"Error finding content: {str(e)}",
                "files": []
            }

    async def get_file_for_sending(
        self,
        file_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get file path and metadata for sending via Telegram.
        Returns file info if found and accessible.
        """
        logger.info("tool_get_file_for_sending", file_name=file_name)

        try:
            # Search for file by name
            result = await self.session.execute(
                select(FileMetadata).where(
                    FileMetadata.file_name.ilike(f"%{file_name}%"),
                    FileMetadata.is_indexed == True
                ).limit(1)
            )

            file_meta = result.scalar_one_or_none()

            if not file_meta:
                logger.info("file_not_found", file_name=file_name)
                return None

            file_path = Path(file_meta.file_path)

            # Check if file exists and is accessible
            if not file_path.exists():
                logger.warning("file_not_accessible", path=str(file_path))
                return None

            # Check file size (Telegram limit: 50MB for bots)
            file_size_mb = file_meta.file_size_bytes / (1024 * 1024)
            if file_size_mb > 50:
                logger.warning("file_too_large", size_mb=file_size_mb)
                return {
                    "path": str(file_path),
                    "name": file_meta.file_name,
                    "size_mb": file_size_mb,
                    "too_large": True
                }

            return {
                "path": str(file_path),
                "name": file_meta.file_name,
                "type": file_meta.file_type,
                "size_mb": file_size_mb,
                "too_large": False
            }

        except Exception as e:
            logger.error("get_file_for_sending_failed", error=str(e))
            return None

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
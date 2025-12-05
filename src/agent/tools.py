"""Agent tools - COMPLETE FIX: Better file matching, CSV support, exact content."""
from typing import List, Optional, Dict, Any
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from src.config.logging_config import get_logger
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.models.database import FileMetadata

logger = get_logger(__name__)


class AgentTools:
    """Enhanced tools with better file matching and CSV support."""

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
        """Search for files by semantic similarity - FULL RAG content previews."""
        logger.info("tool_search_files", query=query, top_k=top_k)

        try:
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            filters = {"file_types": file_types} if file_types else None

            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 3,
                score_threshold=0.3,
                filters=filters,
            )

            if not results:
                return {"text": "❌ No files found matching your search.", "files": []}

            # Deduplicate by file_name, keep best score per file
            seen_files: Dict[str, Any] = {}
            for result in results:
                file_name = result.metadata.file_name

                if file_name not in seen_files or result.score > seen_files[file_name]["score"]:
                    seen_files[file_name] = {
                        "path": result.metadata.file_path,
                        "name": file_name,
                        "type": result.metadata.file_type,
                        "score": result.score,
                        "content": result.content,
                        "page_number": result.metadata.page_number,
                    }

            unique_files = sorted(seen_files.values(), key=lambda x: x["score"], reverse=True)[:top_k]

            logger.info("files_deduplicated", found=len(unique_files), total_chunks=len(results))

            # FIXED: Return FULL content previews for RAG
            content_parts = []
            for f in unique_files:
                # Truncate content preview to 400 chars for Telegram
                preview = f['content'][:400]
                if len(f['content']) > 400:
                    preview += "..."

                content_parts.append(
                    f"📄 {f['name']} ({f['score']:.1%})\n"
                    f"TYPE: {f['type']}\n"
                    f"CONTENT:\n{preview}\n"
                    f"{'─' * 60}"
                )

            return {
                "text": f"✅ Found {len(unique_files)} file(s) matching '{query}':\n\n" +
                        "\n\n".join(content_parts),
                "files": unique_files,
                "files_count": len(unique_files),
                "has_content": True
            }

        except Exception as e:
            logger.error("search_files_failed", error=str(e))
            return {"text": f"❌ Error searching: {str(e)}", "files": []}

    async def find_specific_content(
        self,
        query: str,
        file_name: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Find specific content (paragraphs, sections, questions).
        FIX: Returns COMPLETE content without truncation.
        """
        logger.info("tool_find_specific_content", query=query)

        try:
            # Extract file name if in query
            query_lower = query.lower()

            # Try to find file name in query
            if "from" in query_lower or "in" in query_lower:
                words = query.split()
                for i, word in enumerate(words):
                    if word.lower() in ["from", "in", "file"]:
                        if i + 1 < len(words):
                            file_name = " ".join(words[i+1:])
                            # Clean up file name
                            file_name = file_name.strip().strip("?.,!\"'")
                            break

            logger.info("extracted_file_name", file_name=file_name)

            # Generate embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Build filters if file name provided
            filters = {}
            if file_name:
                # Search for file in database
                result = await self.session.execute(
                    select(FileMetadata).where(
                        FileMetadata.file_name.ilike(f"%{file_name}%"),
                        FileMetadata.is_indexed == True
                    )
                )
                file_metas = result.scalars().all()

                if file_metas:
                    # Use first matching file
                    file_meta = file_metas[0]
                    filters = {"file_path": file_meta.file_path}
                    logger.info("found_file_in_db", file=file_meta.file_name)
                else:
                    logger.warning("file_not_found_in_db", file_name=file_name)
                    return {
                        "text": f"❌ No file found with name containing '{file_name}'",
                        "files": []
                    }

            # Search
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=0.25,  # Even lower for specific searches
                filters=filters if filters else None,
            )

            if not results:
                return {
                    "text": f"❌ No content found matching: {query}",
                    "files": []
                }

            # Combine multiple chunks if from same file (for complete paragraphs)
            best_result = results[0]
            complete_content = [best_result.content]

            # Add additional chunks if they're from same file and relevant
            for result in results[1:3]:  # Up to 3 chunks for completeness
                if result.metadata.file_path == best_result.metadata.file_path:
                    if result.score > 0.4:  # Only if relevant
                        complete_content.append(result.content)

            full_content = "\n\n".join(complete_content)

            output_parts = [
                f"✅ Found in: {best_result.metadata.file_name}",
                f"Relevance: {best_result.score:.1%}",
            ]

            if best_result.metadata.page_number:
                output_parts.append(f"Page: {best_result.metadata.page_number}")

            # Show COMPLETE content
            output_parts.append(f"\n📋 Content:\n\n{full_content}")

            if len(results) > 3:
                output_parts.append(f"\n💡 {len(results)-3} more sections available")

            return {
                "text": "\n".join(output_parts),
                "files": []  # Don't auto-send files
            }

        except Exception as e:
            logger.error("find_specific_content_failed", error=str(e), exc_info=True)
            return {
                "text": f"❌ Error: {str(e)}",
                "files": []
            }

    async def get_file_for_sending(
        self,
        file_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get file for sending.
        FIX: Better file name extraction and matching, CSV support.
        """
        logger.info("tool_get_file_for_sending", file_name=file_name)

        try:
            # Clean file name - remove common words
            query_lower = file_name.lower()
            stop_words = {'send', 'me', 'the', 'file', 'give', 'share', 'download', 'get', 'please', 'can', 'you'}

            words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]

            if not words:
                # Use original if all filtered
                search_term = file_name
            else:
                search_term = " ".join(words)

            logger.info("cleaned_search_term", term=search_term)

            # Search in database with multiple patterns
            # Try exact match first, then partial
            queries = [
                # Exact file name
                select(FileMetadata).where(
                    FileMetadata.file_name.ilike(f"{search_term}%"),
                    FileMetadata.is_indexed == True
                ),
                # Contains anywhere
                select(FileMetadata).where(
                    FileMetadata.file_name.ilike(f"%{search_term}%"),
                    FileMetadata.is_indexed == True
                ),
                # Any word matches
                select(FileMetadata).where(
                    or_(*[FileMetadata.file_name.ilike(f"%{word}%") for word in words[:3]]),
                    FileMetadata.is_indexed == True
                )
            ]

            file_meta = None
            for query in queries:
                result = await self.session.execute(query.limit(1))
                file_meta = result.scalar_one_or_none()
                if file_meta:
                    logger.info("file_found_with_query", file=file_meta.file_name)
                    break

            if not file_meta:
                logger.info("no_file_found", search_term=search_term)
                return None

            file_path = Path(file_meta.file_path)

            # Triple verification
            if not file_path.exists():
                logger.warning("file_not_on_disk", path=str(file_path))
                return None

            if not file_path.is_file():
                logger.warning("path_not_file", path=str(file_path))
                return None

            # Check size
            file_size_mb = file_meta.file_size_bytes / (1024 * 1024)

            if file_size_mb > 50:
                return {
                    "path": str(file_path),
                    "name": file_meta.file_name,
                    "size_mb": file_size_mb,
                    "too_large": True,
                    "exists": True
                }

            logger.info("file_ready_for_sending", file=file_meta.file_name)

            return {
                "path": str(file_path),
                "name": file_meta.file_name,
                "type": file_meta.file_type,
                "size_mb": file_size_mb,
                "too_large": False,
                "exists": True
            }

        except Exception as e:
            logger.error("get_file_for_sending_failed", error=str(e), exc_info=True)
            return None

    async def list_files(
        self,
        pattern: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """
        List files with verification.
        FIX: Better display, CSV included.
        """
        logger.info("tool_list_files", pattern=pattern, file_type=file_type)

        try:
            query = select(FileMetadata).where(FileMetadata.is_indexed == True)

            if pattern:
                query = query.where(FileMetadata.file_name.ilike(f"%{pattern}%"))

            if file_type:
                query = query.where(FileMetadata.file_type == file_type)

            # Order by modified date (newest first)
            query = query.order_by(FileMetadata.file_modified_at.desc())
            query = query.limit(limit)

            result = await self.session.execute(query)
            files = result.scalars().all()

            if not files:
                return "❌ No files found."

            # Verify files exist
            existing_files = []
            for file in files:
                if Path(file.file_path).exists():
                    existing_files.append(file)

            if not existing_files:
                return "❌ No accessible files found."

            # Format output by type
            by_type: Dict[str, List] = {}
            for file in existing_files:
                if file.file_type not in by_type:
                    by_type[file.file_type] = []
                by_type[file.file_type].append(file)

            output_parts = [f"✅ Found {len(existing_files)} file(s):\n"]

            for file_type, type_files in sorted(by_type.items()):
                output_parts.append(f"\n📁 {file_type.upper()} files ({len(type_files)}):")
                for file in type_files[:10]:  # Max 10 per type
                    size_mb = file.file_size_bytes / (1024 * 1024)
                    output_parts.append(
                        f"  • {file.file_name} ({size_mb:.1f}MB)"
                    )

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("list_files_failed", error=str(e), exc_info=True)
            return f"❌ Error: {str(e)}"

    async def get_file_stats(
        self,
        file_type: Optional[str] = None,
    ) -> str:
        """Get file statistics with CSV support."""
        logger.info("tool_file_stats", file_type=file_type)

        try:
            query = select(func.count(FileMetadata.id)).where(
                FileMetadata.is_indexed == True
            )
            if file_type:
                query = query.where(FileMetadata.file_type == file_type)

            result = await self.session.execute(query)
            total_files = result.scalar()

            query = select(func.sum(FileMetadata.file_size_bytes)).where(
                FileMetadata.is_indexed == True
            )
            if file_type:
                query = query.where(FileMetadata.file_type == file_type)

            result = await self.session.execute(query)
            total_bytes = result.scalar() or 0
            total_gb = total_bytes / (1024 ** 3)

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
                "📊 File Statistics:",
                f"Total: {total_files} files, {total_gb:.2f} GB\n",
                "By type:"
            ]

            for row in sorted(by_type, key=lambda x: x.count, reverse=True):
                type_gb = (row.size or 0) / (1024 ** 3)
                output_parts.append(
                    f"  {row.file_type}: {row.count} files, {type_gb:.2f} GB"
                )

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("file_stats_failed", error=str(e), exc_info=True)
            return f"❌ Error: {str(e)}"
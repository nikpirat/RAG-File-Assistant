"""Agent tools - Normalize Windows paths."""
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
    """Enhanced tools"""

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

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for comparison.
        Handle Windows vs Unix path separators.
        """
        # Convert to Path and back to string to normalize
        normalized = str(Path(path).as_posix())  # Use forward slashes
        return normalized.lower()  # Case-insensitive comparison

    def _extract_file_name_from_query(self, query: str) -> Optional[str]:
        """Extract file name from query."""
        query_lower = query.lower()

        stop_words = {
            'what', 'whats', 'what\'s', 'is', 'are', 'the', 'in', 'inside',
            'of', 'from', 'file', 'content', 'contents', 'show', 'me',
            'tell', 'about', 'a', 'an', 'this', 'that', 'search', 'for',
            'find', 'paragraph', 'section', 'page', 'chapter', 'question',
            'answer', 'line'
        }

        words = query.split()

        # Strategy 1: Look for quoted file name
        if '"' in query or "'" in query:
            import re
            quoted = re.findall(r'["\']([^"\']+)["\']', query)
            if quoted:
                return quoted[0]

        # Strategy 2: Look for file with extension
        for word in words:
            if '.' in word and any(ext in word.lower() for ext in ['.txt', '.pdf', '.csv', '.xlsx', '.docx', '.md']):
                return word.strip('.,!?;:')

        # Strategy 3: Get significant words
        significant_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]

        if significant_words:
            file_name = " ".join(significant_words)
            file_name = file_name.strip('.,!?;:')
            return file_name

        return None

    async def _find_file_in_db(self, file_name_query: str) -> Optional[FileMetadata]:
        """Find file in database."""
        if not file_name_query:
            return None

        logger.info("searching_for_file", query=file_name_query)

        # Strategy 1: Exact match
        result = await self.session.execute(
            select(FileMetadata).where(
                FileMetadata.file_name.ilike(file_name_query),
                FileMetadata.is_indexed == True
            ).limit(1)
        )
        file_meta = result.scalar_one_or_none()
        if file_meta:
            logger.info("found_exact_match", file=file_meta.file_name)
            return file_meta

        # Strategy 2: Starts with
        result = await self.session.execute(
            select(FileMetadata).where(
                FileMetadata.file_name.ilike(f"{file_name_query}%"),
                FileMetadata.is_indexed == True
            ).limit(1)
        )
        file_meta = result.scalar_one_or_none()
        if file_meta:
            logger.info("found_starts_with", file=file_meta.file_name)
            return file_meta

        # Strategy 3: Contains
        result = await self.session.execute(
            select(FileMetadata).where(
                FileMetadata.file_name.ilike(f"%{file_name_query}%"),
                FileMetadata.is_indexed == True
            ).limit(1)
        )
        file_meta = result.scalar_one_or_none()
        if file_meta:
            logger.info("found_contains", file=file_meta.file_name)
            return file_meta

        # Strategy 4: Word-by-word
        words = file_name_query.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 3:
                    result = await self.session.execute(
                        select(FileMetadata).where(
                            FileMetadata.file_name.ilike(f"%{word}%"),
                            FileMetadata.is_indexed == True
                        ).limit(1)
                    )
                    file_meta = result.scalar_one_or_none()
                    if file_meta:
                        logger.info("found_by_word", word=word, file=file_meta.file_name)
                        return file_meta

        logger.info("file_not_found_in_db", query=file_name_query)
        return None

    async def search_files(
        self,
        query: str,
        top_k: int = 5,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search for files."""
        logger.info("tool_search_files", query=query, top_k=top_k)

        try:
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            filters = {"file_types": file_types} if file_types else None

            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                score_threshold=0.3,
                filters=filters,
            )

            if not results:
                return {
                    "text": "❌ No files found matching your search.",
                    "files": []
                }

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

            logger.info("files_found", count=len(unique_files))

            content_parts = [f"✅ Found {len(unique_files)} file(s) matching '{query}':\n"]

            for i, f in enumerate(unique_files, 1):
                preview = f['content'][:500]
                if len(f['content']) > 500:
                    preview += "..."

                content_parts.append(
                    f"\n{i}. 📄 **{f['name']}** ({f['score']:.0%} match)\n"
                    f"Type: {f['type']}\n"
                    f"Content:\n{preview}\n"
                    f"{'─' * 50}"
                )

            return {
                "text": "\n".join(content_parts),
                "files": unique_files,
            }

        except Exception as e:
            logger.error("search_files_failed", error=str(e), exc_info=True)
            return {
                "text": f"❌ Error searching: {str(e)}",
                "files": []
            }

    async def find_specific_content(
        self,
        query: str,
        file_name: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Find specific content in files.
        Normalize paths for comparison.
        """
        logger.info("tool_find_specific_content", query=query)

        try:
            extracted_name = self._extract_file_name_from_query(query)

            logger.info("extracted_file_name", name=extracted_name)

            file_meta = None
            if extracted_name:
                file_meta = await self._find_file_in_db(extracted_name)

            if not file_meta:
                logger.warning("file_not_identified_searching_all")

                query_embedding = await self.embedding_service.generate_query_embedding(query)

                results = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    score_threshold=0.25,
                    filters=None,
                )

                if not results:
                    return {
                        "text": f"❌ Could not find any content matching: {query}",
                        "files": []
                    }

                best_result = results[0]

                output = f"""❌ Could not identify specific file from query: "{query}"

However, I found related content in **{best_result.metadata.file_name}** ({best_result.score:.0%} relevance):

📋 Content:
{best_result.content[:800]}

💡 Tip: Specify the exact file name, e.g., "What's in file.txt?"
"""
                return {
                    "text": output,
                    "files": []
                }

            logger.info("file_found_in_db", file=file_meta.file_name, path=file_meta.file_path)

            # CRITICAL PATH FIX: Normalize the target path
            target_path_normalized = self._normalize_path(file_meta.file_path)
            logger.info("normalized_target_path", path=target_path_normalized)

            # Search without filters
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            all_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=20,
                score_threshold=0.15,
                filters=None,
            )

            logger.info("total_results_before_filter", count=len(all_results))

            # CRITICAL PATH FIX: Normalize paths for comparison
            file_specific_results = []
            for result in all_results:
                result_path_normalized = self._normalize_path(result.metadata.file_path)

                # Debug: Log first result for comparison
                if len(file_specific_results) == 0:
                    logger.info(
                        "path_comparison_debug",
                        target=target_path_normalized,
                        result=result_path_normalized,
                        match=target_path_normalized == result_path_normalized
                    )

                if result_path_normalized == target_path_normalized:
                    file_specific_results.append(result)

            logger.info("results_after_manual_filter", count=len(file_specific_results))

            if not file_specific_results:
                return {
                    "text": f"""⚠️ File **{file_meta.file_name}** is indexed but I couldn't find content matching your query.

File info:
- Type: {file_meta.file_type}
- Size: {file_meta.file_size_bytes / 1024:.2f} KB
- Chunks: {file_meta.chunk_count}

Try:
- "What's inside {file_meta.file_name}?" (broader search)
- Check if the file has readable content

Debug: Got {len(all_results)} total results but 0 matched this file's path.
""",
                    "files": []
                }

            results = file_specific_results[:top_k]

            all_content = []
            for result in results[:3]:
                all_content.append(result.content)

            full_content = "\n\n".join(all_content)

            relevance_info = ", ".join([f"{r.score:.0%}" for r in results[:3]])

            output = f"""✅ Found in: **{file_meta.file_name}**
File type: {file_meta.file_type}
Size: {file_meta.file_size_bytes / 1024:.2f} KB
Relevance: {relevance_info}

📋 Content:
{full_content}

{f"💡 Showing {len(results)} sections from this file" if len(results) > 1 else ""}"""

            return {
                "text": output,
                "files": []
            }

        except Exception as e:
            logger.error("find_specific_content_failed", error=str(e), exc_info=True)
            return {
                "text": f"❌ Error: {str(e)}",
                "files": []
            }

    async def get_file_for_sending(
        self,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Get file for sending."""
        logger.info("tool_get_file_for_sending", query=query)

        try:
            file_name_query = self._extract_file_name_from_query(query)

            if not file_name_query:
                logger.warning("could_not_extract_file_name", query=query)
                return None

            logger.info("extracted_for_sending", name=file_name_query)

            file_meta = await self._find_file_in_db(file_name_query)

            if not file_meta:
                logger.info("no_file_found_for_sending", query=file_name_query)
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
                logger.warning("file_too_large", size_mb=file_size_mb)
                return {
                    "path": str(file_path),
                    "name": file_meta.file_name,
                    "size_mb": file_size_mb,
                    "too_large": True,
                    "exists": True
                }

            logger.info("file_ready_for_sending", file=file_meta.file_name, size_mb=f"{file_size_mb:.2f}")

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
        limit: int = 50,
    ) -> str:
        """List files."""
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
                output_parts.append(f"\n**{file_type.upper()} files ({len(type_files)}):**")
                for file in type_files:
                    size_mb = file.file_size_bytes / (1024 * 1024)
                    output_parts.append(f"  • `{file.file_name}` ({size_mb:.2f}MB)")

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("list_files_failed", error=str(e), exc_info=True)
            return f"❌ Error: {str(e)}"

    async def get_file_stats(
        self,
        file_type: Optional[str] = None,
    ) -> str:
        """Get file statistics."""
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
                "📊 **File Statistics:**",
                f"Total: {total_files} files, {total_gb:.2f} GB\n",
                "**By type:**"
            ]

            for row in sorted(by_type, key=lambda x: x.count, reverse=True):
                type_gb = (row.size or 0) / (1024 ** 3)
                output_parts.append(
                    f"  • {row.file_type}: {row.count} files, {type_gb:.2f} GB"
                )

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("file_stats_failed", error=str(e), exc_info=True)
            return f"❌ Error: {str(e)}"
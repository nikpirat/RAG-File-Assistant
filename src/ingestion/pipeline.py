"""Main ingestion pipeline orchestrating all components."""
import asyncio
from pathlib import Path
from typing import List, Optional, cast
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select
from tqdm.asyncio import tqdm

from src.config.settings import settings
from src.config.logging_config import get_logger
from src.models.database import FileMetadata, DocumentChunk, Base
from src.models.schemas import ChunkMetadata
from src.ingestion.parsers import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrate document ingestion process."""

    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

        # Database setup
        self.engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
        )
        self.async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

        logger.info("ingestion_pipeline_initialized")

    async def initialize(self) -> None:
        """Initialize database and vector store."""
        logger.info("initializing_pipeline")

        # Create database tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create vector collection
        await self.vector_store.create_collection()

        logger.info("pipeline_initialized")

    async def scan_directory(
            self,
            root_path: Optional[Path] = None,
    ) -> List[Path]:
        """Scan directory for supported files."""
        root = root_path or settings.files_root_path
        logger.info("scanning_directory", path=str(root))

        files = []
        for ext in settings.supported_extensions:
            files.extend(root.rglob(f"*{ext}"))

        logger.info("scan_complete", total_files=len(files))
        return files

    async def ingest_file(
            self,
            file_path: Path,
            session: AsyncSession,
    ) -> bool:
        """Ingest a single file."""
        file_metadata: FileMetadata | None = None
        try:
            logger.info("ingesting_file", file_path=str(file_path))

            # Check if already indexed
            result = await session.execute(
                select(FileMetadata).where(FileMetadata.file_path == str(file_path))
            )
            existing = result.scalar_one_or_none()

            # Get file stats
            stat = file_path.stat()

            # Skip if not modified
            if existing and existing.file_modified_at:
                file_mtime = datetime.fromtimestamp(stat.st_mtime)
                if existing.file_modified_at >= file_mtime and existing.is_indexed:
                    logger.info("file_already_indexed", file_path=str(file_path))
                    return True

                # File modified, delete old chunks
                await self.vector_store.delete_by_file_path(str(file_path))
                await session.execute(
                    select(DocumentChunk).where(
                        DocumentChunk.file_metadata_id == existing.id
                    )
                )

            # Parse document
            parsed_doc = self.parser.parse(file_path)

            # Create/update file metadata
            file_metadata = existing or FileMetadata()
            file_metadata.file_path = str(file_path)
            file_metadata.file_name = file_path.name
            file_metadata.file_type = file_path.suffix.lower().lstrip(".")
            file_metadata.file_size_bytes = stat.st_size
            file_metadata.word_count = parsed_doc.word_count
            file_metadata.page_count = parsed_doc.page_count
            file_metadata.file_created_at = datetime.fromtimestamp(stat.st_ctime)
            file_metadata.file_modified_at = datetime.fromtimestamp(stat.st_mtime)
            file_metadata.extra_metadata = parsed_doc.extra_metadata

            if not existing:
                session.add(file_metadata)
            await session.flush()

            # Create chunk metadata
            chunk_meta = ChunkMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type=file_path.suffix.lower().lstrip("."),
                file_size_bytes=stat.st_size,
                chunk_index=0,
                chunk_type="text",
                file_modified_at=datetime.fromtimestamp(stat.st_mtime),
            )

            # Chunk document
            chunks = self.chunker.chunk_document(parsed_doc.content, chunk_meta)

            if not chunks:
                logger.warning("no_chunks_created", file_path=str(file_path))
                file_metadata.is_indexed = False
                file_metadata.error_message = "No chunks created"
                await session.commit()
                return False

            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.generate_embeddings_batch(chunk_texts)

            # Store in vector DB
            chunk_ids = await self.vector_store.upsert_chunks(chunks, embeddings)

            # Store chunk records in DB
            for chunk, chunk_id in zip(chunks, chunk_ids):
                db_chunk = DocumentChunk(
                    file_metadata_id=cast(int, file_metadata.id),
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    content_hash=self.chunker.create_hash(chunk.content),
                    chunk_type=chunk.metadata.chunk_type,
                    token_count=chunk.token_count,
                    page_number=chunk.metadata.page_number,
                    section_title=chunk.metadata.section_title,
                    vector_id=chunk_id,
                )
                session.add(db_chunk)

            # Update file metadata
            file_metadata.is_indexed = True
            file_metadata.chunk_count = len(chunks)
            file_metadata.indexed_at = datetime.now(timezone.utc)
            file_metadata.error_message = None

            await session.commit()

            logger.info(
                "file_ingested",
                file_path=str(file_path),
                chunks=len(chunks),
            )

            return True

        except Exception as e:
            logger.error(
                "file_ingestion_failed",
                file_path=str(file_path),
                error=str(e),
                exc_info=True,
            )

            # Record error
            if 'file_metadata' in locals():
                file_metadata.is_indexed = False
                file_metadata.error_message = str(e)
                await session.commit()

            return False

    async def ingest_all(
            self,
            root_path: Optional[Path] = None,
            max_concurrent: int = 5,
    ) -> dict:
        """Ingest all files in directory."""
        logger.info("starting_full_ingestion")

        # Scan directory
        files = await self.scan_directory(root_path)

        # Track progress
        stats = {
            "total": len(files),
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }

        # Process files with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_file(file_path: Path):
            async with semaphore:
                async with self.async_session() as session:
                    success = await self.ingest_file(file_path, session)
                    if success:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

        # Process with progress bar
        tasks = [process_file(f) for f in files]
        await tqdm.gather(*tasks, desc="Ingesting files")

        logger.info(
            "ingestion_complete",
            total=stats["total"],
            success=stats["success"],
            failed=stats["failed"],
        )

        return stats

    async def ingest_single_file(self, file_path: Path) -> bool:
        """Ingest a single file (public interface)."""
        async with self.async_session() as session:
            return await self.ingest_file(file_path, session)

    async def get_indexed_files_count(self) -> int:
        """Get count of indexed files."""
        async with self.async_session() as session:
            result = await session.execute(
                select(FileMetadata).where(FileMetadata.is_indexed == True)
            )
            return len(result.all())

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()
        logger.info("pipeline_closed")
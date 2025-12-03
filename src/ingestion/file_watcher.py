"""File system watcher for automatic indexing."""
import asyncio
from pathlib import Path
from typing import Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from src.config.settings import settings
from src.config.logging_config import get_logger
from src.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system changes."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.pending_files: Set[Path] = set()
        self.processing = False
        self.supported_extensions = set(settings.supported_extensions)

        logger.info("file_watcher_initialized")

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        file_path = Path(event.src_path)
        logger.info("file_created", file=str(file_path))
        self.pending_files.add(file_path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        file_path = Path(event.src_path)
        logger.info("file_modified", file=str(file_path))
        self.pending_files.add(file_path)

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        file_path = Path(event.src_path)
        logger.info("file_deleted", file=str(file_path))

        # Schedule deletion from vector store
        asyncio.create_task(self._delete_file(file_path))

    async def _delete_file(self, file_path: Path):
        """Delete file from vector store and database."""
        try:
            from src.retrieval.vector_store import VectorStore
            from sqlalchemy import select, delete
            from src.models.database import FileMetadata, DocumentChunk

            vector_store = VectorStore()

            # Delete from vector store
            await vector_store.delete_by_file_path(str(file_path))

            # Delete from database
            async with self.pipeline.async_session() as session:
                # Get file metadata
                result = await session.execute(
                    select(FileMetadata).where(FileMetadata.file_path == str(file_path))
                )
                file_meta = result.scalar_one_or_none()

                if file_meta:
                    # Delete chunks
                    await session.execute(
                        delete(DocumentChunk).where(
                            DocumentChunk.file_metadata_id == file_meta.id
                        )
                    )

                    # Delete file metadata
                    await session.delete(file_meta)
                    await session.commit()

                    logger.info("file_deleted_from_db", file=str(file_path))

        except Exception as e:
            logger.error("file_deletion_failed", file=str(file_path), error=str(e))

    async def process_pending_files(self):
        """Process pending files for indexing."""
        if self.processing or not self.pending_files:
            return

        self.processing = True

        try:
            files_to_process = list(self.pending_files)
            self.pending_files.clear()

            logger.info("processing_pending_files", count=len(files_to_process))

            async with self.pipeline.async_session() as session:
                for file_path in files_to_process:
                    if file_path.exists():
                        try:
                            await self.pipeline.ingest_file(file_path, session)
                            logger.info("file_indexed", file=str(file_path))
                        except Exception as e:
                            logger.error(
                                "file_indexing_failed",
                                file=str(file_path),
                                error=str(e)
                            )

        finally:
            self.processing = False


class FileWatcher:
    """Watch file system for changes and auto-index."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.observer: Optional[Observer] = None
        self.handler = FileChangeHandler(pipeline)
        self.processing_task: Optional[asyncio.Task] = None

        logger.info("file_watcher_created")

    def start(self):
        """Start watching for file changes."""
        if self.observer is not None:
            logger.warning("file_watcher_already_running")
            return

        self.observer = Observer()
        self.observer.schedule(
            self.handler,
            str(settings.files_root_path),
            recursive=True
        )
        self.observer.start()

        # Start background processing task
        self.processing_task = asyncio.create_task(self._process_loop())

        logger.info(
            "file_watcher_started",
            path=str(settings.files_root_path)
        )

    async def _process_loop(self):
        """Background loop to process pending files."""
        while True:
            try:
                await asyncio.sleep(5)  # Process every 5 seconds
                await self.handler.process_pending_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("process_loop_error", error=str(e))

    def stop(self):
        """Stop watching for file changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None

        if self.processing_task:
            self.processing_task.cancel()
            self.processing_task = None

        logger.info("file_watcher_stopped")
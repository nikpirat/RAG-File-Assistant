"""File system watcher - SAFE: No accidental deletions."""
import asyncio
from pathlib import Path
from typing import Set, Optional
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from src.config.settings import settings
from src.config.logging_config import get_logger
from src.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system changes - SAFE version."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.pending_files: Set[Path] = set()
        self.processing = False
        self.supported_extensions = set(settings.supported_extensions)

        logger.info("file_watcher_initialized_safe")

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        file_path = Path(event.src_path)
        logger.info("file_created_detected", file=str(file_path))
        self.pending_files.add(file_path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        file_path = Path(event.src_path)

        # FIX: Only re-index if file actually changed (not just accessed)
        # Check if modification is recent (within last 5 seconds)
        try:
            mtime = file_path.stat().st_mtime
            current_time = datetime.now().timestamp()

            # Only re-index if modified within last 5 seconds
            if current_time - mtime < 5:
                logger.info("file_modified_detected", file=str(file_path))
                self.pending_files.add(file_path)
            else:
                logger.debug("file_modified_old", file=str(file_path))
        except Exception as e:
            logger.warning("file_stat_failed", file=str(file_path), error=str(e))

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        file_path = Path(event.src_path)

        # FIX: CRITICAL - Double-check file is actually deleted before removing from DB
        if file_path.exists():
            logger.warning(
                "delete_event_but_file_exists",
                file=str(file_path),
                message="Ignoring delete event because file still exists"
            )
            return

        logger.info("file_deleted_confirmed", file=str(file_path))

        # Schedule deletion from vector store (with verification)
        asyncio.create_task(self._delete_file_safe(file_path))

    async def _delete_file_safe(self, file_path: Path):
        """
        Delete file from vector store and database - SAFE version.
        FIX: Triple-check file is actually gone before deleting.
        """
        try:
            # CRITICAL: Wait a moment and check again
            await asyncio.sleep(2)

            # FIX: Final verification - if file exists, DON'T delete
            if file_path.exists():
                logger.warning(
                    "delete_cancelled_file_exists",
                    file=str(file_path),
                    message="CRITICAL: File exists, cancelling database deletion"
                )
                return

            from src.retrieval.vector_store import VectorStore
            from sqlalchemy import select, delete
            from src.models.database import FileMetadata, DocumentChunk

            vector_store = VectorStore()

            logger.info("deleting_from_db", file=str(file_path))

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

                    logger.info("file_deleted_from_db_success", file=str(file_path))
                else:
                    logger.info("file_not_in_db", file=str(file_path))

        except Exception as e:
            logger.error("file_deletion_failed", file=str(file_path), error=str(e), exc_info=True)

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
                    # FIX: Verify file still exists before indexing
                    if not file_path.exists():
                        logger.info("file_disappeared_before_indexing", file=str(file_path))
                        continue

                    try:
                        await self.pipeline.ingest_file(file_path, session)
                        logger.info("file_indexed_success", file=str(file_path))
                    except Exception as e:
                        logger.error(
                            "file_indexing_failed",
                            file=str(file_path),
                            error=str(e),
                            exc_info=True
                        )

        finally:
            self.processing = False


class FileWatcher:
    """Watch file system for changes and auto-index - SAFE version."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.observer: Optional[Observer] = None
        self.handler = FileChangeHandler(pipeline)
        self.processing_task: Optional[asyncio.Task] = None

        logger.info("file_watcher_created_safe")

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
            "file_watcher_started_safe",
            path=str(settings.files_root_path)
        )

    async def _process_loop(self):
        """Background loop to process pending files."""
        while True:
            try:
                await asyncio.sleep(10)  # FIX: Increased from 5 to 10 seconds
                await self.handler.process_pending_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("process_loop_error", error=str(e), exc_info=True)

    def stop(self):
        """Stop watching for file changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None

        if self.processing_task:
            self.processing_task.cancel()
            self.processing_task = None

        logger.info("file_watcher_stopped_safe")
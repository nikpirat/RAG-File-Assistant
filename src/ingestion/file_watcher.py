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
    """Handle file system changes - Triple-check deletions."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.pending_files: Set[Path] = set()
        self.processing = False
        self.supported_extensions = set(settings.get_supported_extensions_list())

        logger.info("file_watcher_initialized_fixed")

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def _is_temp_file(self, file_path: str) -> bool:
        """Check if file is temporary (ignore these)."""
        path = Path(file_path)
        name = path.name.lower()

        # Ignore temp files, system files, hidden files
        temp_patterns = [
            '.tmp', '.temp', '~', '.swp', '.lock',
            '.DS_Store', 'Thumbs.db', 'desktop.ini'
        ]

        return any(pattern in name for pattern in temp_patterns)

    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        if self._is_temp_file(event.src_path):
            return

        file_path = Path(event.src_path)

        # Wait a moment to ensure file is fully written
        try:
            if not file_path.exists():
                return

            # Check if file is accessible
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read 1 byte

            logger.info("file_created", file=str(file_path))
            self.pending_files.add(file_path)
        except Exception as e:
            logger.debug("file_not_ready", file=str(file_path), error=str(e))

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification - Only real changes."""
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        if self._is_temp_file(event.src_path):
            return

        file_path = Path(event.src_path)

        try:
            if not file_path.exists():
                return

            # Check modification time - only if recent (within 3 seconds)
            mtime = file_path.stat().st_mtime
            current_time = datetime.now().timestamp()

            if current_time - mtime < 3:
                logger.info("file_modified", file=str(file_path))
                self.pending_files.add(file_path)
            else:
                logger.debug("file_modified_old_ignoring", file=str(file_path))

        except Exception as e:
            logger.debug("file_stat_failed", file=str(file_path), error=str(e))

    def on_deleted(self, event: FileSystemEvent):
        """
        Handle file deletion - TRIPLE verification before DB deletion.
        """
        if event.is_directory or not self._is_supported_file(event.src_path):
            return

        file_path = Path(event.src_path)

        # Check if file actually doesn't exist
        if file_path.exists():
            logger.warning(
                "delete_event_but_file_exists",
                file=str(file_path),
                message="File still exists, ignoring delete event"
            )
            return

        # Remove from pending queue
        if file_path in self.pending_files:
            self.pending_files.discard(file_path)
            logger.info("removed_from_pending", file=str(file_path))

        logger.info("file_deletion_detected", file=str(file_path))

        # Schedule safe deletion
        asyncio.create_task(self._delete_file_safe(file_path))

    async def _delete_file_safe(self, file_path: Path):
        """
        Ultra-safe file deletion with triple verification.
        """
        try:
            # Wait and verify multiple times
            await asyncio.sleep(3)  # Wait 3 seconds

            # TRIPLE CHECK: Verify file is really gone
            for attempt in range(3):
                if file_path.exists():
                    logger.error(
                        "delete_cancelled_file_reappeared",
                        file=str(file_path),
                        attempt=attempt + 1,
                        message="CRITICAL: File exists, ABORTING deletion from database"
                    )
                    return

                await asyncio.sleep(1)  # Wait between checks

            # File is definitely gone, safe to delete from DB
            from src.retrieval.vector_store import VectorStore
            from sqlalchemy import select, delete
            from src.models.database import FileMetadata, DocumentChunk

            vector_store = VectorStore()

            logger.info("confirmed_deletion_proceeding", file=str(file_path))

            # Delete from vector store
            await vector_store.delete_by_file_path(str(file_path))

            # Delete from database
            async with self.pipeline.async_session() as session:
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
                else:
                    logger.info("file_not_in_db", file=str(file_path))

        except Exception as e:
            logger.error("file_deletion_failed", file=str(file_path), error=str(e), exc_info=True)

    async def process_pending_files(self):
        """Process pending files for indexing - Better verification."""
        if self.processing or not self.pending_files:
            return

        self.processing = True

        try:
            files_to_process = list(self.pending_files)
            self.pending_files.clear()

            logger.info("processing_pending_files", count=len(files_to_process))

            async with self.pipeline.async_session() as session:
                for file_path in files_to_process:
                    # VERIFY: File still exists
                    if not file_path.exists():
                        logger.info("file_disappeared", file=str(file_path))
                        continue

                    # VERIFY: File is readable
                    try:
                        with open(file_path, 'rb') as f:
                            f.read(1)
                    except Exception as e:
                        logger.warning("file_not_readable", file=str(file_path), error=str(e))
                        continue

                    # Index the file
                    try:
                        await self.pipeline.ingest_file(file_path, session)
                        logger.info("file_indexed", file=str(file_path))
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
    """Watch file system for changes."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.observer: Optional[Observer] = None
        self.handler = FileChangeHandler(pipeline)
        self.processing_task: Optional[asyncio.Task] = None

        logger.info("file_watcher_created_fixed")

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
        """Background loop to process pending files - Longer interval."""
        while True:
            try:
                await asyncio.sleep(15)  # Increased to 15 seconds
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

        logger.info("file_watcher_stopped")
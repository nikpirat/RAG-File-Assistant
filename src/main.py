"""Enhanced main application - Zero warnings, fully typed."""
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.tg_bot.bot import create_bot_application
from src.tg_bot.handlers import BotHandlers
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.file_watcher import FileWatcher

logger = get_logger(__name__)


async def run_bot() -> None:
    """Run the bot with all services."""
    # Initialize variables
    application = None
    file_watcher: Optional[FileWatcher] = None
    pipeline: Optional[IngestionPipeline] = None

    try:
        # Initialize ingestion pipeline
        logger.info("initializing_ingestion_pipeline")
        pipeline = IngestionPipeline()
        await pipeline.initialize()

        # Start file watcher for auto-indexing
        logger.info("starting_file_watcher")
        file_watcher = FileWatcher(pipeline)
        file_watcher.start()
        logger.info("file_watcher_active", path=str(settings.files_root_path))

        # Create Telegram application
        logger.info("creating_telegram_bot")
        application = create_bot_application()

        # Create and register handlers
        handlers = BotHandlers()
        handlers.register_handlers(application)

        logger.info("bot_starting")

        # Initialize and start bot
        await application.initialize()
        await application.start()

        # Start polling - FIX: Use getattr to avoid type checker warning
        updater = getattr(application, 'updater', None)
        if updater is not None:
            await updater.start_polling(
                allowed_updates=["message", "callback_query"],
                drop_pending_updates=True,
            )
        else:
            raise RuntimeError("Application updater not initialized")

        logger.info("✓ Bot is running with file watcher!")
        logger.info("✓ Files will be auto-indexed when added/modified")
        logger.info("✓ Press Ctrl+C to stop.")

        # Run until interrupted
        try:
            # Keep the bot running
            await asyncio.Event().wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("shutdown_initiated")

    finally:
        logger.info("bot_stopping")

        # Stop file watcher
        if file_watcher is not None:
            try:
                logger.info("stopping_file_watcher")
                file_watcher.stop()
            except Exception as e:
                logger.error("file_watcher_stop_error", error=str(e))

        # Stop bot
        if application is not None:
            try:
                # Use getattr for updater to avoid warning
                updater = getattr(application, 'updater', None)
                if updater is not None:
                    await updater.stop()
                await application.stop()
                await application.shutdown()
            except Exception as e:
                logger.error("application_stop_error", error=str(e))

        # Close pipeline
        if pipeline is not None:
            try:
                await pipeline.close()
            except Exception as e:
                logger.error("pipeline_close_error", error=str(e))

        logger.info("✓ Bot stopped gracefully")


def main() -> None:
    """Main entry point."""
    configure_logging()
    logger.info("=== Starting Enhanced RAG File Assistant Bot ===")
    logger.info("app_version", version="Phase 2 Enhanced - v0.2.1")

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("✓ Keyboard interrupt received")
    except Exception as e:
        logger.error("fatal_error", error=str(e), exc_info=True)
        sys.exit(1)

    logger.info("✓ Shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
"""Main application entry point - Optimized with proper lifecycle management."""
import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.tg_bot.bot import create_bot_application
from src.tg_bot.handlers import BotHandlers
from src.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)


class Application:
    """Main application with proper lifecycle management."""

    def __init__(self):
        """Initialize application components."""
        self.bot_app = None
        self.pipeline: Optional[IngestionPipeline] = None
        self.handlers: Optional[BotHandlers] = None
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all services."""
        logger.info("initializing_application")

        try:
            # Initialize ingestion pipeline
            logger.info("initializing_pipeline")
            self.pipeline = IngestionPipeline()
            await self.pipeline.initialize()
            logger.info("pipeline_initialized")

            # Check indexed files
            indexed_count = await self.pipeline.get_indexed_files_count()
            logger.info("indexed_files_count", count=indexed_count)

            if indexed_count == 0:
                logger.warning(
                    "no_files_indexed",
                    message="No files indexed yet. Run 'python scripts/index_files.py' first.",
                )

            # Create Telegram bot application
            logger.info("creating_telegram_bot")
            self.bot_app = create_bot_application()

            # Create and register handlers
            self.handlers = BotHandlers()
            self.handlers.register_handlers(self.bot_app)

            logger.info("application_initialized_successfully")

        except Exception as e:
            logger.error("initialization_failed", error=str(e), exc_info=True)
            await self.cleanup()
            raise

    async def start(self) -> None:
        """Start the application."""
        logger.info("starting_application")

        try:
            # Initialize bot
            await self.bot_app.initialize()
            await self.bot_app.start()

            # Start polling
            updater = self.bot_app.updater
            if updater is None:
                raise RuntimeError("Bot updater not initialized")

            await updater.start_polling(
                allowed_updates=["message", "callback_query"],
                drop_pending_updates=True,
            )

            logger.info("✓ Bot is running!")
            logger.info("✓ Using Gemini 2.5 Flash for AI")
            logger.info("✓ Press Ctrl+C to stop")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error("application_start_failed", error=str(e), exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("cleaning_up_resources")

        # Stop bot
        if self.bot_app is not None:
            try:
                logger.info("stopping_bot")
                updater = self.bot_app.updater
                if updater is not None:
                    await updater.stop()
                await self.bot_app.stop()
                await self.bot_app.shutdown()
                logger.info("bot_stopped")
            except Exception as e:
                logger.error("bot_cleanup_error", error=str(e))

        # Close handlers (database connections)
        if self.handlers is not None:
            try:
                logger.info("closing_handlers")
                await self.handlers.cleanup()
                logger.info("handlers_closed")
            except Exception as e:
                logger.error("handlers_cleanup_error", error=str(e))

        # Close pipeline
        if self.pipeline is not None:
            try:
                logger.info("closing_pipeline")
                await self.pipeline.close()
                logger.info("pipeline_closed")
            except Exception as e:
                logger.error("pipeline_cleanup_error", error=str(e))

        logger.info("cleanup_complete")

    def signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info("shutdown_signal_received", signal=signum)
        self.shutdown_event.set()


async def run_application() -> None:
    """Run the application with proper error handling."""
    app = Application()

    # Setup signal handlers
    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)

    try:
        # Initialize and start
        await app.initialize()
        await app.start()

    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")

    except Exception as e:
        logger.error("application_error", error=str(e), exc_info=True)
        raise

    finally:
        # Cleanup
        await app.cleanup()


def main() -> None:
    """Main entry point."""
    # Configure logging first
    configure_logging()

    logger.info("=" * 60)
    logger.info("RAG FILE ASSISTANT - GEMINI POWERED")
    logger.info("=" * 60)
    logger.info("app_version", version=settings.app_version)
    logger.info("environment", debug=settings.debug)
    logger.info("ai_model", model=settings.gemini_model)
    logger.info("embedding_model", model=settings.gemini_embedding_model)
    logger.info("=" * 60)

    # Validate critical settings
    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY not set in environment!")
        logger.error("Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)

    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment!")
        logger.error("Create a bot with @BotFather on Telegram")
        sys.exit(1)

    try:
        # Run application
        asyncio.run(run_application())

    except KeyboardInterrupt:
        logger.info("✓ Keyboard interrupt - shutting down gracefully")

    except Exception as e:
        logger.error("fatal_error", error=str(e), exc_info=True)
        sys.exit(1)

    logger.info("✓ Application shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
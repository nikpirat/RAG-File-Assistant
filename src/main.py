"""Main application entry point."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.tg_bot.bot import create_bot_application
from src.tg_bot.handlers import BotHandlers

logger = get_logger(__name__)


def main():
    """Run the Telegram bot."""
    configure_logging()
    logger.info("=== Starting RAG File Assistant Bot ===")
    logger.info("app_version", version="Phase 2 - v0.2.0")

    # Create application
    application = create_bot_application()

    # Register handlers
    handlers = BotHandlers()
    handlers.register_handlers(application)

    logger.info("bot_starting")

    try:
        # PTB handles polling internally, Ctrl+C raises KeyboardInterrupt
        application.run_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        # Stop the bot cleanly
        application.stop()
        application.shutdown()
        logger.info("✓ Bot stopped gracefully")


if __name__ == "__main__":
    main()

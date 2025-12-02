"""Telegram bot initialization."""
from telegram.ext import Application
from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def create_bot_application() -> Application:
    """Create and configure Telegram bot application."""

    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )

    logger.info("bot_application_created")

    return application
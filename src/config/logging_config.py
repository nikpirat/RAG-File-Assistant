"""Structured logging configuration with structlog."""
import logging
import sys
from typing import Any, cast
import structlog
from structlog import BoundLogger
from structlog.typing import EventDict, BindableLogger
from src.config.settings import settings


def add_app_context(logger: BindableLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to all log entries."""
    event_dict["app"] = settings.app_name
    event_dict["version"] = settings.app_version
    return event_dict


def configure_logging() -> None:
    """Configure structured logging with structlog."""

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_app_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add JSON or console renderer based on debug mode
    if settings.debug:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=cast(type[BindableLogger], BoundLogger),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
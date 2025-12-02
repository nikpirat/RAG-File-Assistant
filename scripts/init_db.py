import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)


async def main():
    configure_logging()
    logger.info("=== Initializing Database and Vector Store ===")

    try:
        pipeline = IngestionPipeline()
        await pipeline.initialize()
        logger.info("✓ Database tables created")
        logger.info("✓ Vector collection created")
        logger.info("=== Initialization Complete ===")
        await pipeline.close()
    except Exception as e:
        logger.error("Initialization failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


async def test_full_pipeline():
    configure_logging()
    logger.info("=== Testing Complete RAG Pipeline ===\n")

    # Test 1: Check Ollama
    logger.info("Test 1: Checking Ollama connection...")
    try:
        emb_service = EmbeddingService()
        test_emb = await emb_service.generate_embedding("test")
        logger.info(f"✓ Ollama works! Embedding dimension: {len(test_emb)}")
    except Exception as e:
        logger.error(f"✗ Ollama failed: {e}")
        return

    # Test 2: Check Vector Store
    logger.info("\nTest 2: Checking Qdrant...")
    try:
        vector_store = VectorStore()
        await vector_store.create_collection()
        info = await vector_store.get_collection_info()
        logger.info(f"✓ Qdrant works! Collection: {info}")
    except Exception as e:
        logger.error(f"✗ Qdrant failed: {e}")
        return

    # Test 3: Ingest test files
    logger.info("\nTest 3: Ingesting test files...")
    try:
        pipeline = IngestionPipeline()
        await pipeline.initialize()

        # Get test files
        test_files = list(settings.files_root_path.glob("*.txt"))
        test_files.extend(settings.files_root_path.glob("*.md"))

        logger.info(f"Found {len(test_files)} test files")

        for file in test_files[:2]:  # Just first 2 files
            success = await pipeline.ingest_single_file(file)
            if success:
                logger.info(f"✓ Ingested: {file.name}")
            else:
                logger.error(f"✗ Failed: {file.name}")

        await pipeline.close()
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        return

    # Test 4: Search
    logger.info("\nTest 4: Testing search...")
    try:
        query = "what is machine learning?"
        logger.info(f"Query: '{query}'")

        query_emb = await emb_service.generate_query_embedding(query)
        results = await vector_store.search(
            query_embedding=query_emb,
            top_k=3,
            score_threshold=0.3,
        )

        logger.info(f"✓ Found {len(results)} results\n")

        for i, result in enumerate(results, 1):
            logger.info(f"--- Result {i} ---")
            logger.info(f"File: {result.metadata.file_name}")
            logger.info(f"Score: {result.score:.4f}")
            logger.info(f"Content: {result.content[:150]}...\n")

    except Exception as e:
        logger.error(f"✗ Search failed: {e}")
        return

    logger.info("=== All Tests Passed! ✓ ===")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
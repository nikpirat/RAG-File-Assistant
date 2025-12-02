"""Test script for the ingestion pipeline."""
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


async def test_single_file_ingestion():
    """Test ingesting a single file."""
    configure_logging()
    logger.info("=== Testing Single File Ingestion ===")

    # Initialize pipeline
    pipeline = IngestionPipeline()
    await pipeline.initialize()

    # Test file (replace with your actual test file)
    test_file = Path("/data/files/test_document.pdf")

    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        logger.info("Please create a test file or update the path in this script")
        return

    # Ingest file
    success = await pipeline.ingest_single_file(test_file)

    if success:
        logger.info("✓ File ingested successfully")
    else:
        logger.error("✗ File ingestion failed")

    # Get stats
    count = await pipeline.get_indexed_files_count()
    logger.info(f"Total indexed files: {count}")

    await pipeline.close()


async def test_search():
    """Test searching the vector store."""
    configure_logging()
    logger.info("=== Testing Vector Search ===")

    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    # Test query
    query = "artificial intelligence and machine learning"
    logger.info(f"Query: {query}")

    # Generate query embedding
    query_embedding = await embedding_service.generate_query_embedding(query)
    logger.info(f"Query embedding dimension: {len(query_embedding)}")

    # Search
    results = await vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        score_threshold=0.5,
    )

    logger.info(f"Found {len(results)} results")

    # Display results
    for i, result in enumerate(results, 1):
        logger.info(f"\n--- Result {i} ---")
        logger.info(f"Score: {result.score:.4f}")
        logger.info(f"File: {result.metadata.file_name}")
        logger.info(f"Content preview: {result.content[:200]}...")

    # Get collection info
    info = await vector_store.get_collection_info()
    logger.info(f"\nCollection stats: {info}")


async def test_full_ingestion():
    """Test ingesting all files in directory."""
    configure_logging()
    logger.info("=== Testing Full Directory Ingestion ===")

    # Initialize pipeline
    pipeline = IngestionPipeline()
    await pipeline.initialize()

    # Ingest all files
    stats = await pipeline.ingest_all(max_concurrent=3)

    logger.info("\n=== Ingestion Complete ===")
    logger.info(f"Total files: {stats['total']}")
    logger.info(f"✓ Success: {stats['success']}")
    logger.info(f"✗ Failed: {stats['failed']}")

    await pipeline.close()


async def test_collection_info():
    """Test getting collection information."""
    configure_logging()
    logger.info("=== Collection Information ===")

    vector_store = VectorStore()
    info = await vector_store.get_collection_info()

    logger.info(f"Total points: {info['total_points']}")
    logger.info(f"Vectors count: {info['vectors_count']}")
    logger.info(f"Indexed vectors: {info['indexed_vectors_count']}")
    logger.info(f"Status: {info['status']}")


async def main():
    """Main test runner."""
    print("\n" + "=" * 60)
    print("RAG Pipeline Test Suite")
    print("=" * 60 + "\n")

    print("Select test to run:")
    print("1. Test single file ingestion")
    print("2. Test vector search")
    print("3. Test full directory ingestion (WARNING: may take time)")
    print("4. Show collection info")
    print("0. Run all tests")

    choice = input("\nEnter choice (0-4): ").strip()

    if choice == "1":
        await test_single_file_ingestion()
    elif choice == "2":
        await test_search()
    elif choice == "3":
        confirm = input("This will index all files. Continue? (y/n): ")
        if confirm.lower() == 'y':
            await test_full_ingestion()
    elif choice == "4":
        await test_collection_info()
    elif choice == "0":
        await test_single_file_ingestion()
        await test_search()
        await test_collection_info()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
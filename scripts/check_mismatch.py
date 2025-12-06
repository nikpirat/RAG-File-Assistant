"""Check for mismatches between database and vector store."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.ingestion.pipeline import IngestionPipeline
from src.models.database import FileMetadata
from src.retrieval.vector_store import VectorStore
from src.retrieval.embeddings import EmbeddingService


async def main():
    """Check for file path mismatches."""
    print("=" * 60)
    print("CHECKING FOR DATABASE vs QDRANT MISMATCHES")
    print("=" * 60)
    print()

    pipeline = IngestionPipeline()
    await pipeline.initialize()

    # Get files from database
    async with pipeline.async_session() as session:
        result = await session.execute(
            select(FileMetadata).where(FileMetadata.is_indexed == True)
        )
        db_files = result.scalars().all()

        print(f"📊 Files in DATABASE: {len(db_files)}")
        print()

        db_paths = set()
        for f in db_files:
            normalized = str(Path(f.file_path).as_posix()).lower()
            db_paths.add(normalized)
            print(f"  • {f.file_name}")
            print(f"    Path: {normalized}")
            print()

    # Get files from Qdrant
    print("=" * 60)
    print(f"📊 Files in QDRANT:")
    print()

    vector_store = VectorStore()
    embedding_service = EmbeddingService()

    # Search with generic query to get all results
    query_embedding = await embedding_service.generate_query_embedding("test")
    all_results = await vector_store.search(
        query_embedding=query_embedding,
        top_k=200,
        score_threshold=0.0,
        filters=None,
    )

    # Get unique file paths from Qdrant
    qdrant_paths = set()
    qdrant_files = {}

    for result in all_results:
        path = str(Path(result.metadata.file_path).as_posix()).lower()
        qdrant_paths.add(path)
        if path not in qdrant_files:
            qdrant_files[path] = {
                "name": result.metadata.file_name,
                "count": 0
            }
        qdrant_files[path]["count"] += 1

    for path, info in sorted(qdrant_files.items()):
        print(f"  • {info['name']} ({info['count']} chunks)")
        print(f"    Path: {path}")
        print()

    # Check for mismatches
    print("=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print()

    # Files in DB but not in Qdrant
    missing_in_qdrant = db_paths - qdrant_paths
    if missing_in_qdrant:
        print(f"❌ Files in DATABASE but NOT in QDRANT ({len(missing_in_qdrant)}):")
        for path in sorted(missing_in_qdrant):
            # Find the file name
            for f in db_files:
                if str(Path(f.file_path).as_posix()).lower() == path:
                    print(f"  • {f.file_name}")
                    print(f"    {path}")
                    break
        print()

    # Files in Qdrant but not in DB
    extra_in_qdrant = qdrant_paths - db_paths
    if extra_in_qdrant:
        print(f"⚠️  Files in QDRANT but NOT in DATABASE ({len(extra_in_qdrant)}):")
        for path in sorted(extra_in_qdrant):
            info = qdrant_files[path]
            print(f"  • {info['name']}")
            print(f"    {path}")
        print()

    # Perfect match
    matched = db_paths & qdrant_paths
    if matched:
        print(f"✅ Files in BOTH ({len(matched)}):")
        for path in sorted(matched):
            info = qdrant_files[path]
            print(f"  • {info['name']} ({info['count']} chunks)")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Database files: {len(db_paths)}")
    print(f"Qdrant files: {len(qdrant_paths)}")
    print(f"Matched: {len(matched)}")
    print(f"Missing in Qdrant: {len(missing_in_qdrant)}")
    print(f"Extra in Qdrant: {len(extra_in_qdrant)}")
    print()

    if missing_in_qdrant:
        print("⚠️  RECOMMENDATION:")
        print("Run: python scripts/reindex_all.py")
        print()

    # Special check for MobTechExam
    print("=" * 60)
    print("SPECIAL CHECK: MobTechExam files")
    print("=" * 60)

    mobtech_db = [p for p in db_paths if 'mobtech' in p]
    mobtech_qdrant = [p for p in qdrant_paths if 'mobtech' in p]

    print(f"In DATABASE: {len(mobtech_db)}")
    for p in mobtech_db:
        print(f"  {p}")

    print(f"\nIn QDRANT: {len(mobtech_qdrant)}")
    for p in mobtech_qdrant:
        print(f"  {p}")

    await pipeline.close()
    print()


if __name__ == "__main__":
    asyncio.run(main())
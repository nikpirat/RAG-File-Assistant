"""Fixed re-indexing - handles duplicates correctly."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.config.settings import settings
from src.retrieval.vector_store import VectorStore
from sqlalchemy import select, delete
from src.models.database import FileMetadata, DocumentChunk


async def main():
    """Re-index all files - fixed version."""
    print("=" * 60)
    print("FIXED COMPLETE RE-INDEXING")
    print("=" * 60)
    print()

    # Ask for confirmation
    response = input("⚠️  This will re-index ALL files. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return

    print()
    print("🔄 Starting re-indexing...")
    print()

    pipeline = IngestionPipeline()
    await pipeline.initialize()

    # Step 1: Clear Qdrant completely
    print("Step 1: Clearing Qdrant vector store...")
    vector_store = VectorStore()

    try:
        await vector_store.client.delete_collection(vector_store.collection_name)
        print("  ✅ Old collection deleted")
    except Exception as e:
        print(f"  ℹ️  {e}")

    await vector_store.create_collection()
    print("  ✅ New collection created")
    print()

    # Step 2: Clear database chunks
    print("Step 2: Clearing database chunks...")
    async with pipeline.async_session() as session:
        await session.execute(delete(DocumentChunk))

        # Also reset indexed status
        result = await session.execute(select(FileMetadata))
        all_files = result.scalars().all()
        for f in all_files:
            f.is_indexed = False
            f.chunk_count = 0

        await session.commit()
        print("  ✅ Database cleaned")
    print()

    # Step 3: Scan directory for actual files
    print("Step 3: Scanning directory...")
    print(f"  Path: {settings.files_root_path}")

    # Get unique files only (no duplicates)
    unique_files = {}
    for ext in settings.get_supported_extensions_list():
        for file_path in settings.files_root_path.rglob(f"*{ext}"):
            if file_path.is_file():
                # Use absolute path as key to avoid duplicates
                abs_path = str(file_path.resolve())
                unique_files[abs_path] = file_path

    files = list(unique_files.values())

    print(f"  ✅ Found {len(files)} unique files")
    print()

    print("Files to index:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    print()

    # Step 4: Index each file
    print("Step 4: Indexing files...")
    print("=" * 60)

    success_count = 0
    failed_count = 0

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file_path.name}")
        print(f"  Path: {file_path}")

        async with pipeline.async_session() as session:
            try:
                success = await pipeline.ingest_file(file_path, session)

                if success:
                    # Verify chunks were created
                    result = await session.execute(
                        select(FileMetadata).where(
                            FileMetadata.file_path == str(file_path)
                        )
                    )
                    file_meta = result.scalar_one_or_none()

                    if file_meta and file_meta.chunk_count > 0:
                        print(f"  ✅ Success: {file_meta.chunk_count} chunks")
                        success_count += 1
                    else:
                        print(f"  ⚠️  No chunks created")
                        failed_count += 1
                else:
                    print(f"  ❌ Failed")
                    failed_count += 1

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                failed_count += 1

    print()
    print("=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    print()

    # Check database
    async with pipeline.async_session() as session:
        result = await session.execute(
            select(FileMetadata).where(FileMetadata.is_indexed == True)
        )
        indexed_files = result.scalars().all()

        total_chunks = sum(f.chunk_count for f in indexed_files)

        print(f"📊 Database:")
        print(f"  Indexed files: {len(indexed_files)}")
        print(f"  Total chunks: {total_chunks}")
        print()

        for f in indexed_files:
            print(f"  • {f.file_name}: {f.chunk_count} chunks")

    print()

    # Check Qdrant
    try:
        info = await vector_store.get_collection_info()
        print(f"📊 Qdrant:")
        print(f"  Total vectors: {info['total_points']}")

        if info['total_points'] == total_chunks:
            print(f"  ✅ Match! Database chunks = Qdrant vectors")
        else:
            print(f"  ⚠️  Mismatch! DB has {total_chunks} but Qdrant has {info['total_points']}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files found: {len(files)}")
    print(f"✅ Successfully indexed: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print()

    if success_count == len(files) and info['total_points'] == total_chunks:
        print("✅ PERFECT! Everything indexed correctly.")
        print()
        print("Next: Restart bot and test:")
        print("  python src/main.py")
        print("  'What's inside MobTechExam?'")
    else:
        print("⚠️  Some files failed to index.")
        print("Check the errors above.")

    await pipeline.close()
    print()


if __name__ == "__main__":
    asyncio.run(main())
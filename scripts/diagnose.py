"""Diagnostic script - FIXED."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.ingestion.pipeline import IngestionPipeline
from src.models.database import FileMetadata


async def main():
    """Run diagnostics."""
    print("=" * 60)
    print("RAG FILE ASSISTANT - DIAGNOSTICS")
    print("=" * 60)

    pipeline = IngestionPipeline()
    await pipeline.initialize()

    async with pipeline.async_session() as session:
        # Check total files
        result = await session.execute(
            select(FileMetadata).where(FileMetadata.is_indexed == True)
        )
        files = result.scalars().all()

        print(f"\n📊 Total indexed files: {len(files)}")

        if not files:
            print("\n❌ No files indexed!")
            await pipeline.close()
            return

        print("\n📄 Files in database:")
        total_chunks = 0
        for file in files:
            print(f"\n  • {file.file_name} ({file.file_type})")
            print(f"    Size: {file.file_size_bytes / 1024:.2f} KB")
            print(f"    Chunks: {file.chunk_count}")

            # Check if file exists on disk
            if not Path(file.file_path).exists():
                print(f"    ⚠️  WARNING: File not found on disk!")
            else:
                print(f"    ✅ File exists on disk")

            total_chunks += file.chunk_count

        print(f"\n📊 Summary:")
        print(f"  Files: {len(files)}")
        print(f"  Total chunks: {total_chunks}")

        # Check for CSV files
        result = await session.execute(
            select(FileMetadata).where(
                FileMetadata.file_type == "csv",
                FileMetadata.is_indexed == True
            )
        )
        csv_files = result.scalars().all()

        print(f"\n📊 CSV files: {len(csv_files)}")
        if csv_files:
            for csv_file in csv_files:
                print(f"  • {csv_file.file_name}")

        # Check vector store
        from src.retrieval.vector_store import VectorStore
        vector_store = VectorStore()

        try:
            info = await vector_store.get_collection_info()
            print(f"\n🔢 Vector store (Qdrant):")
            print(f"  Total points: {info['total_points']}")
            print(f"  Status: {info['status']}")

            if info['total_points'] == 0:
                print("  ❌ ERROR: No vectors in Qdrant!")
                print("  Run: python scripts/index_files.py")
            elif info['total_points'] != total_chunks:
                print(f"  ⚠️  Mismatch: DB has {total_chunks} chunks but Qdrant has {info['total_points']} points")
                print("  Consider re-indexing")
            else:
                print("  ✅ Vector count matches database chunks")
        except Exception as e:
            print(f"\n❌ Error accessing vector store: {e}")

        # Check for issues
        print("\n🔍 Checking for issues:")

        issues_found = False

        # Issue 1: Files not indexed
        result = await session.execute(
            select(FileMetadata).where(FileMetadata.is_indexed == False)
        )
        unindexed = result.scalars().all()
        if unindexed:
            issues_found = True
            print(f"  ⚠️  {len(unindexed)} files not indexed:")
            for file in unindexed:
                print(f"     • {file.file_name}: {file.error_message}")

        # Issue 2: Files with no chunks
        result = await session.execute(
            select(FileMetadata).where(
                FileMetadata.is_indexed == True,
                FileMetadata.chunk_count == 0
            )
        )
        no_chunks = result.scalars().all()
        if no_chunks:
            issues_found = True
            print(f"  ❌ {len(no_chunks)} files indexed but have no chunks:")
            for file in no_chunks:
                print(f"     • {file.file_name}")

        # Issue 3: Missing files on disk
        missing = [f for f in files if not Path(f.file_path).exists()]
        if missing:
            issues_found = True
            print(f"  ⚠️  {len(missing)} files missing from disk:")
            for file in missing:
                print(f"     • {file.file_name}")

        if not issues_found:
            print("  ✅ No issues found!")

        print("\n" + "=" * 60)
        print("STATUS:")
        print("=" * 60)

        if info.get('total_points', 0) == 0:
            print("❌ CRITICAL: No vectors in Qdrant - re-index needed")
        elif no_chunks:
            print("❌ Some files have no chunks - re-index needed")
        elif missing:
            print("⚠️  Some files missing - restore or clean database")
        else:
            print("✅ Everything looks good!")
            print("\nYour database is healthy. The bot should work correctly.")
            print("\nTest with:")
            print("  1. List all files")
            print("  2. What's inside MobTechExam?")
            print("  3. Send me MobTechExam")

    await pipeline.close()
    print("\n✅ Diagnostics complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
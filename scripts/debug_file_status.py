"""Debug script to check file status in database vs disk."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.config.logging_config import configure_logging, get_logger
from src.models.database import FileMetadata
from src.ingestion.pipeline import IngestionPipeline
from src.config.settings import settings

logger = get_logger(__name__)


async def check_file_status(filename: str = None):
    """Check status of files in database vs disk."""
    configure_logging()
    logger.info("=== File Status Check ===")

    pipeline = IngestionPipeline()
    await pipeline.initialize()

    async with pipeline.async_session() as session:
        if filename:
            # Check specific file
            result = await session.execute(
                select(FileMetadata).where(
                    FileMetadata.file_name.ilike(f"%{filename}%")
                )
            )
            files = result.scalars().all()
        else:
            # Check all files
            result = await session.execute(
                select(FileMetadata)
            )
            files = result.scalars().all()

        print(f"\n📊 Found {len(files)} file(s) in database:\n")

        for file_meta in files:
            file_path = Path(file_meta.file_path)

            # Check if file exists
            exists = file_path.exists()
            is_file = file_path.is_file() if exists else False

            # Status
            status = "✅ OK" if (exists and is_file and file_meta.is_indexed) else "❌ ISSUE"

            print(f"{status} {file_meta.file_name}")
            print(f"   Path: {file_meta.file_path}")
            print(f"   Type: {file_meta.file_type}")
            print(f"   Size: {file_meta.file_size_bytes / 1024:.1f} KB")
            print(f"   Indexed: {file_meta.is_indexed}")
            print(f"   Chunks: {file_meta.chunk_count}")
            print(f"   Exists on disk: {exists}")
            print(f"   Is file: {is_file}")

            if file_meta.error_message:
                print(f"   ⚠️ Error: {file_meta.error_message}")

            # Issues
            issues = []
            if not exists:
                issues.append("❌ File doesn't exist on disk")
            if exists and not is_file:
                issues.append("❌ Path exists but is not a file")
            if not file_meta.is_indexed:
                issues.append("⚠️ Not indexed")
            if file_meta.chunk_count == 0:
                issues.append("⚠️ No chunks")

            if issues:
                print(f"   Issues found:")
                for issue in issues:
                    print(f"     {issue}")

            print()

    await pipeline.close()


async def check_specific_csv():
    """Check data_sample1.csv specifically."""
    configure_logging()
    print("\n🔍 Checking data_sample1.csv status...\n")

    pipeline = IngestionPipeline()
    await pipeline.initialize()

    # Check in database
    async with pipeline.async_session() as session:
        result = await session.execute(
            select(FileMetadata).where(
                FileMetadata.file_name.ilike("%data_sample1%")
            )
        )
        file_meta = result.scalar_one_or_none()

        if file_meta:
            print("✅ Found in database:")
            print(f"   Name: {file_meta.file_name}")
            print(f"   Path: {file_meta.file_path}")
            print(f"   Indexed: {file_meta.is_indexed}")
            print(f"   Chunks: {file_meta.chunk_count}")

            # Check on disk
            file_path = Path(file_meta.file_path)
            if file_path.exists():
                print(f"   ✅ File exists on disk")
                print(f"   Size: {file_path.stat().st_size} bytes")
            else:
                print(f"   ❌ FILE MISSING ON DISK!")
        else:
            print("❌ NOT found in database")

            # Check if file exists on disk
            possible_paths = list(settings.files_root_path.rglob("data_sample1.csv"))
            if possible_paths:
                print(f"\n⚠️ But found on disk:")
                for p in possible_paths:
                    print(f"   {p}")
                    print(f"   Size: {p.stat().st_size} bytes")
                print("\n💡 File needs to be indexed!")
            else:
                print("\n❌ Not found on disk either")

    await pipeline.close()


async def list_all_csv():
    """List all CSV files."""
    configure_logging()
    print("\n📋 All CSV Files:\n")

    pipeline = IngestionPipeline()
    await pipeline.initialize()

    async with pipeline.async_session() as session:
        result = await session.execute(
            select(FileMetadata).where(
                FileMetadata.file_type == 'csv'
            )
        )
        csv_files = result.scalars().all()

        print(f"Found {len(csv_files)} CSV file(s) in database:\n")

        for f in csv_files:
            exists = Path(f.file_path).exists()
            status = "✅" if (exists and f.is_indexed) else "❌"
            print(f"{status} {f.file_name}")
            print(f"   Indexed: {f.is_indexed}, Chunks: {f.chunk_count}")
            print(f"   On disk: {exists}")
            print()

    await pipeline.close()


async def main():
    """Main menu."""
    print("\n" + "="*60)
    print("File Status Debugger")
    print("="*60)
    print("\nOptions:")
    print("1. Check specific file")
    print("2. Check all files")
    print("3. Check data_sample1.csv")
    print("4. List all CSV files")

    choice = input("\nChoice (1-4): ").strip()

    if choice == "1":
        filename = input("Enter filename (or part of it): ").strip()
        await check_file_status(filename)
    elif choice == "2":
        await check_file_status()
    elif choice == "3":
        await check_specific_csv()
    elif choice == "4":
        await list_all_csv()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
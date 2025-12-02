"""Smart document chunking with semantic awareness."""
import hashlib
from typing import List
from src.config.settings import settings
from src.config.logging_config import get_logger
from src.models.schemas import DocumentChunkSchema, ChunkMetadata

logger = get_logger(__name__)


class DocumentChunker:
    """Chunk documents with overlap and semantic awareness."""

    def __init__(
            self,
            chunk_size: int = settings.chunk_size,
            chunk_overlap: int = settings.chunk_overlap,
            min_chunk_size: int = settings.min_chunk_size,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        logger.info(
            "chunker_initialized",
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_size=min_chunk_size,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text) // 4

    def create_hash(self, content: str) -> str:
        """Create hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()

    def chunk_document(
            self,
            content: str,
            file_metadata: ChunkMetadata,
    ) -> List[DocumentChunkSchema]:
        """Chunk document with overlap."""
        logger.info("chunking_document", file_path=file_metadata.file_path)

        # Split by pages if page markers exist
        if "--- Page" in content:
            chunks = self._chunk_by_pages(content, file_metadata)
        else:
            chunks = self._chunk_by_tokens(content, file_metadata)

        logger.info(
            "chunking_complete",
            file_path=file_metadata.file_path,
            chunk_count=len(chunks),
        )

        return chunks

    def _chunk_by_pages(
            self,
            content: str,
            file_metadata: ChunkMetadata,
    ) -> List[DocumentChunkSchema]:
        """Chunk by pages with overlap."""
        chunks = []
        pages = content.split("--- Page")

        for page_idx, page_content in enumerate(pages):
            if not page_content.strip():
                continue

            # Extract page number
            lines = page_content.split("\n", 1)
            page_num = None
            if len(lines) > 0 and "---" in lines[0]:
                try:
                    page_num = int(lines[0].split()[0])
                    page_content = lines[1] if len(lines) > 1 else ""
                except (ValueError, IndexError):
                    pass

            # Chunk this page
            page_chunks = self._chunk_by_tokens(
                page_content,
                file_metadata,
                page_number=page_num,
            )

            chunks.extend(page_chunks)

        return chunks

    def _chunk_by_tokens(
            self,
            content: str,
            file_metadata: ChunkMetadata,
            page_number: int = None,
    ) -> List[DocumentChunkSchema]:
        """Chunk by token count with overlap."""
        chunks = []

        # Split by paragraphs first for better semantic boundaries
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        current_chunk = []
        current_tokens = 0
        chunk_index = file_metadata.chunk_index if hasattr(file_metadata, 'chunk_index') else 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        file_metadata,
                        chunk_index,
                        page_number,
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                split_chunks = self._split_large_text(para, file_metadata, chunk_index, page_number)
                chunks.extend(split_chunks)
                chunk_index += len(split_chunks)
                continue

            # Check if adding this paragraph exceeds chunk size
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text,
                    file_metadata,
                    chunk_index,
                    page_number,
                ))
                chunk_index += 1

                # Keep overlap: retain last few paragraphs
                overlap_paras = self._get_overlap_paragraphs(current_chunk)
                current_chunk = overlap_paras
                current_tokens = sum(self.count_tokens(p) for p in current_chunk)

            current_chunk.append(para)
            current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_text,
                    file_metadata,
                    chunk_index,
                    page_number,
                ))

        return chunks

    def _split_large_text(
            self,
            text: str,
            file_metadata: ChunkMetadata,
            start_index: int,
            page_number: int = None,
    ) -> List[DocumentChunkSchema]:
        """Split large text by sentences."""
        chunks = []
        sentences = text.split(". ")

        current_chunk = []
        current_tokens = 0
        chunk_index = start_index

        for sentence in sentences:
            sentence = sentence.strip() + "."
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text,
                    file_metadata,
                    chunk_index,
                    page_number,
                ))
                chunk_index += 1
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sent_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text,
                file_metadata,
                chunk_index,
                page_number,
            ))

        return chunks

    def _get_overlap_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Get paragraphs for overlap."""
        overlap_paras = []
        overlap_tokens = 0

        for para in reversed(paragraphs):
            para_tokens = self.count_tokens(para)
            if overlap_tokens + para_tokens > self.chunk_overlap:
                break
            overlap_paras.insert(0, para)
            overlap_tokens += para_tokens

        return overlap_paras

    def _create_chunk(
            self,
            content: str,
            file_metadata: ChunkMetadata,
            chunk_index: int,
            page_number: int = None,
    ) -> DocumentChunkSchema:
        """Create chunk schema."""
        metadata = ChunkMetadata(
            file_path=file_metadata.file_path,
            file_name=file_metadata.file_name,
            file_type=file_metadata.file_type,
            file_size_bytes=file_metadata.file_size_bytes,
            chunk_index=chunk_index,
            chunk_type="text",
            page_number=page_number,
            section_title=None,
            file_modified_at=file_metadata.file_modified_at,
        )

        return DocumentChunkSchema(
            content=content,
            metadata=metadata,
            token_count=self.count_tokens(content),
            chunk_index=chunk_index,
        )
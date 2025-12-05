"""Embedding generation using Google Gemini."""
import asyncio
import google.generativeai as genai
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Google Gemini."""

    def __init__(self):
        """Initialize Gemini embedding service."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in environment!")

        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)

        self.model_name = settings.gemini_embedding_model
        self.dimensions = settings.embedding_dimensions
        self.task_type = "retrieval_document"  # For document embeddings

        logger.info(
            "gemini_embedding_service_initialized",
            model=self.model_name,
            dimensions=self.dimensions,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed
            task_type: Type of task (retrieval_document, retrieval_query, etc.)

        Returns:
            Embedding vector
        """
        try:
            # Truncate if too long (Gemini has limits)
            max_chars = 20000  # Conservative limit
            if len(text) > max_chars:
                logger.warning("text_truncated", original_len=len(text), max_len=max_chars)
                text = text[:max_chars]

            # Generate embedding
            result = await genai.embed_content_async(
                model=self.model_name,
                content=text,
                task_type=task_type,
            )

            embedding = result['embedding']

            logger.debug(
                "embedding_generated",
                text_length=len(text),
                embedding_dim=len(embedding),
            )

            return embedding

        except Exception as e:
            logger.error("gemini_embedding_failed", error=str(e), exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Size of batches (None = use settings default)

        Returns:
            List of embedding vectors
        """
        if batch_size is None:
            batch_size = settings.batch_size

        logger.info("generating_embeddings_batch", total_texts=len(texts))

        all_embeddings = []

        # Process in batches with concurrency control
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Gemini API supports batch embedding (up to 100 texts)
                # But for safety, we'll do concurrent single requests
                tasks = [
                    self.generate_embedding(text, task_type="retrieval_document")
                    for text in batch
                ]

                batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle any errors in the batch
                for idx, result in enumerate(batch_embeddings):
                    if isinstance(result, Exception):
                        logger.error(
                            "batch_embedding_error",
                            batch_idx=i + idx,
                            error=str(result),
                        )
                        # Use a zero vector as fallback
                        all_embeddings.append([0.0] * self.dimensions)
                    else:
                        all_embeddings.append(result)

                logger.info(
                    "batch_processed",
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch),
                )

            except Exception as e:
                logger.error(
                    "batch_embedding_failed",
                    batch_num=i // batch_size + 1,
                    error=str(e),
                )
                # Add zero vectors for failed batch
                for _ in range(len(batch)):
                    all_embeddings.append([0.0] * self.dimensions)

        logger.info("embeddings_batch_complete", total_embeddings=len(all_embeddings))
        return all_embeddings

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        logger.info("generating_query_embedding", query_length=len(query))

        # Use retrieval_query task type for better search performance
        return await self.generate_embedding(query, task_type="retrieval_query")

    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
"""Embedding generation with Ollama."""
from typing import List
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Ollama."""

    def __init__(self):
        self.base_url = settings.ollama_host
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

        logger.info(
            "embedding_service_initialized",
            model=self.model,
            dimensions=self.dimensions,
            host=self.base_url,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text,
                    }
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["embedding"]

                logger.debug(
                    "embedding_generated",
                    text_length=len(text),
                    embedding_dim=len(embedding),
                )

                return embedding

        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = settings.batch_size,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        logger.info("generating_embeddings_batch", total_texts=len(texts))

        all_embeddings = []

        # Ollama doesn't have native batching, so we do concurrent requests
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Create tasks for concurrent processing
                tasks = [self.generate_embedding(text) for text in batch]
                batch_embeddings = await asyncio.gather(*tasks)
                all_embeddings.extend(batch_embeddings)

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
                raise

        logger.info("embeddings_batch_complete", total_embeddings=len(all_embeddings))
        return all_embeddings

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        logger.info("generating_query_embedding", query_length=len(query))
        return await self.generate_embedding(query)
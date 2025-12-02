"""LLM service using Ollama."""
import json
import httpx
from typing import AsyncIterator, Optional, List, Dict
from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class LLMService:
    """Service for interacting with Ollama LLM."""

    def __init__(self):
        self.base_url = settings.ollama_host
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature

        logger.info(
            "llm_service_initialized",
            model=self.model,
            temperature=self.temperature,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion from prompt."""
        logger.info("generating_completion", prompt_length=len(prompt))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature or self.temperature,
                        "stream": False,
                    }
                )
                response.raise_for_status()
                data = response.json()

                content = data["message"]["content"]
                logger.info("completion_generated", length=len(content))

                return content

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Generate completion with streaming."""
        logger.info("generating_completion_stream", prompt_length=len(prompt))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature or self.temperature,
                        "stream": True,
                    }
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                chunk = data["message"]["content"]
                                yield chunk

        except Exception as e:
            logger.error("stream_generation_failed", error=str(e))
            raise

    async def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """Generate with conversation history."""
        logger.info("generating_with_history", message_count=len(messages))

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature or self.temperature,
                        "stream": False,
                    }
                )
                response.raise_for_status()
                data = response.json()

                return data["message"]["content"]

        except Exception as e:
            logger.error("history_generation_failed", error=str(e))
            raise
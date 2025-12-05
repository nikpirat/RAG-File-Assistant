"""LLM service using Google Gemini 2.5 Flash."""
import google.generativeai as genai
from typing import AsyncIterator, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class LLMService:
    """Service for interacting with Google Gemini LLM."""

    def __init__(self):
        """Initialize Gemini service."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in environment!")

        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)

        # Configure generation settings
        self.generation_config = genai.GenerationConfig(
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
            top_p=0.95,
            top_k=40,
        )

        # Safety settings (permissive for file assistant context)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )

        logger.info(
            "gemini_llm_initialized",
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text response
        """
        logger.info("generating_completion", prompt_length=len(prompt))

        try:
            # Build full prompt with system context
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Override config if needed
            if temperature is not None or max_tokens is not None:
                config = genai.GenerationConfig(
                    temperature=temperature or self.generation_config.temperature,
                    max_output_tokens=max_tokens or self.generation_config.max_output_tokens,
                    top_p=0.95,
                    top_k=40,
                )
                model = genai.GenerativeModel(
                    model_name=settings.gemini_model,
                    generation_config=config,
                    safety_settings=self.safety_settings,
                )
            else:
                model = self.model

            # Generate response
            response = await model.generate_content_async(
                full_prompt,
                request_options={"timeout": settings.gemini_timeout}
            )

            # Extract text
            if not response.parts:
                logger.warning("empty_response_from_gemini")
                return "I apologize, but I couldn't generate a response. Please try again."

            content = response.text

            logger.info(
                "completion_generated",
                length=len(content),
                finish_reason=response.candidates[0].finish_reason if response.candidates else None,
            )

            return content

        except Exception as e:
            logger.error("gemini_generation_failed", error=str(e), exc_info=True)
            # Return user-friendly error
            if "quota" in str(e).lower():
                return "API quota exceeded. Please try again later."
            elif "safety" in str(e).lower():
                return "Response blocked by safety filters. Please rephrase your query."
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Generate completion with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Override default temperature

        Yields:
            Text chunks as they're generated
        """
        logger.info("generating_completion_stream", prompt_length=len(prompt))

        try:
            # Build full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Override config if needed
            if temperature is not None:
                config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=self.generation_config.max_output_tokens,
                    top_p=0.95,
                    top_k=40,
                )
                model = genai.GenerativeModel(
                    model_name=settings.gemini_model,
                    generation_config=config,
                    safety_settings=self.safety_settings,
                )
            else:
                model = self.model

            # Generate with streaming
            response = await model.generate_content_async(
                full_prompt,
                stream=True,
                request_options={"timeout": settings.gemini_timeout}
            )

            # Stream chunks
            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error("gemini_stream_failed", error=str(e), exc_info=True)
            yield f"Error: {str(e)}"

    async def generate_with_history(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate with conversation history.

        Args:
            messages: List of {"role": "user"|"model", "content": "..."}
            temperature: Override default temperature

        Returns:
            Generated response
        """
        logger.info("generating_with_history", message_count=len(messages))

        try:
            # Override config if needed
            if temperature is not None:
                config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=self.generation_config.max_output_tokens,
                    top_p=0.95,
                    top_k=40,
                )
                model = genai.GenerativeModel(
                    model_name=settings.gemini_model,
                    generation_config=config,
                    safety_settings=self.safety_settings,
                )
            else:
                model = self.model

            # Start chat with history
            chat = model.start_chat(
                history=[
                    {"role": msg["role"], "parts": [msg["content"]]}
                    for msg in messages[:-1]  # All but last message
                ]
            )

            # Send last message
            last_message = messages[-1]["content"]
            response = await chat.send_message_async(
                last_message,
                request_options={"timeout": settings.gemini_timeout}
            )

            return response.text

        except Exception as e:
            logger.error("gemini_history_generation_failed", error=str(e), exc_info=True)
            raise

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).

        Args:
            text: Input text

        Returns:
            Token count
        """
        try:
            result = await self.model.count_tokens_async(text)
            return result.total_tokens
        except Exception as e:
            logger.warning("token_counting_failed", error=str(e))
            # Fallback: rough estimation (1 token ≈ 4 chars)
            return len(text) // 4
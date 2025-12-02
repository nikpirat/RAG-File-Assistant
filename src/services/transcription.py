"""Voice transcription service using faster-whisper."""
import asyncio
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class TranscriptionService:
    """Service for transcribing audio files."""

    def __init__(self):
        logger.info(
            "initializing_whisper",
            model=settings.whisper_model,
            device=settings.whisper_device,
        )

        # Initialize Whisper model
        self.model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

        logger.info("whisper_initialized")

    async def transcribe_file(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio file to text."""
        logger.info("transcribing_audio", file=str(audio_path))

        try:
            # Run transcription in thread pool (it's CPU-intensive)
            loop = asyncio.get_event_loop()

            def _transcribe():
                segments, info = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=5,
                )
                # Convert generator to list and extract info
                segment_list = list(segments)
                return segment_list, info

            segments, info = await loop.run_in_executor(None, _transcribe)

            # Detected language
            detected_language = info.language
            logger.info(
                "transcription_info",
                language=detected_language,
                probability=info.language_probability,
            )

            # Combine all segments
            text = " ".join([segment.text for segment in segments])

            logger.info(
                "transcription_complete",
                text_length=len(text),
                language=detected_language,
            )

            return text.strip()

        except Exception as e:
            logger.error("transcription_failed", error=str(e), exc_info=True)
            raise

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        temp_dir: Path,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio from bytes."""
        # Save to temp file
        temp_file = temp_dir / "temp_audio.ogg"
        temp_file.write_bytes(audio_bytes)

        try:
            text = await self.transcribe_file(temp_file, language)
            return text
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
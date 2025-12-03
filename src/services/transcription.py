"""Enhanced voice transcription with improved accuracy."""
import asyncio
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class TranscriptionService:
    """Enhanced service for transcribing audio files with better accuracy."""

    def __init__(self):
        logger.info(
            "initializing_whisper_enhanced",
            model=settings.whisper_model,
            device=settings.whisper_device,
        )

        # Initialize Whisper model with optimized settings
        self.model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            # Enhanced settings for better accuracy
            cpu_threads=4,  # Use multiple CPU threads
            num_workers=1,
        )

        logger.info("whisper_initialized_enhanced")

    async def transcribe_file(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio file to text with enhanced accuracy."""
        logger.info("transcribing_audio_enhanced", file=str(audio_path))

        try:
            # Run transcription in thread pool (it's CPU-intensive)
            loop = asyncio.get_event_loop()

            def _transcribe():
                # Enhanced transcription settings
                segments, info = self.model.transcribe(
                    str(audio_path),
                    language=language,

                    # ENHANCED ACCURACY SETTINGS
                    beam_size=5,  # Higher = more accurate but slower
                    best_of=5,  # Number of candidates to consider
                    patience=2.0,  # Patience in beam search

                    # VAD (Voice Activity Detection) for better segmentation
                    vad_filter=True,  # Filter out non-speech
                    vad_parameters={
                        "threshold": 0.5,  # Voice detection sensitivity
                        "min_speech_duration_ms": 250,  # Minimum speech duration
                        "max_speech_duration_s": 30,  # Maximum speech duration
                        "min_silence_duration_ms": 500,  # Minimum silence between segments
                        "speech_pad_ms": 400,  # Padding around speech
                    },

                    # Additional accuracy improvements
                    condition_on_previous_text=True,  # Use context from previous segments
                    temperature=0.0,  # Greedy decoding for better accuracy
                    compression_ratio_threshold=2.4,  # Filter out repetitions
                    log_prob_threshold=-1.0,  # Filter out low confidence
                    no_speech_threshold=0.6,  # Threshold for detecting speech

                    # Word-level timestamps for better precision
                    word_timestamps=True,

                    # Initial prompt to guide transcription (helps with context)
                    initial_prompt="This is a voice message about files, documents, or questions.",
                )

                # Convert generator to list and extract info
                segment_list = list(segments)
                return segment_list, info

            segments, info = await loop.run_in_executor(None, _transcribe)

            # Detected language
            detected_language = info.language
            logger.info(
                "transcription_info_enhanced",
                language=detected_language,
                probability=info.language_probability,
                duration=info.duration,
            )

            # Combine all segments with improved formatting
            text_parts = []
            for segment in segments:
                # Clean up the text
                segment_text = segment.text.strip()

                # Skip very short segments (likely noise)
                if len(segment_text) < 2:
                    continue

                # Skip segments with very low confidence (if available)
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob < -1.0:
                    logger.debug("skipping_low_confidence_segment", text=segment_text[:30])
                    continue

                text_parts.append(segment_text)

            # Join segments with proper spacing
            text = " ".join(text_parts)

            # Post-processing for better readability
            text = self._post_process_text(text)

            logger.info(
                "transcription_complete_enhanced",
                text_length=len(text),
                segments=len(text_parts),
                language=detected_language,
            )

            return text.strip()

        except Exception as e:
            logger.error("transcription_failed", error=str(e), exc_info=True)
            raise

    def _post_process_text(self, text: str) -> str:
        """Post-process transcribed text for better quality."""

        # Remove multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Fix common transcription issues
        replacements = {
            ' i ': ' I ',
            "i'm": "I'm",
            "i'll": "I'll",
            "i'd": "I'd",
            "i've": "I've",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Ensure proper sentence ending
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        temp_dir: Path,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio from bytes with enhanced accuracy."""
        # Save to temp file with proper extension
        temp_file = temp_dir / "temp_audio.ogg"
        temp_file.write_bytes(audio_bytes)

        try:
            # Log file size for debugging
            size_kb = len(audio_bytes) / 1024
            logger.info("transcribing_bytes", size_kb=f"{size_kb:.1f}")

            text = await self.transcribe_file(temp_file, language)
            return text
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
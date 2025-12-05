"""Telegram bot handlers - Optimized with proper resource management."""
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, CommandHandler, filters
from telegram.constants import ChatAction
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.config.settings import settings
from src.config.logging_config import get_logger
from src.services.transcription import TranscriptionService
from src.services.llm import LLMService
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.agent.tools import AgentTools
from src.agent.memory import ConversationMemory
from src.agent.graph import FileAssistantAgent

logger = get_logger(__name__)


class BotHandlers:
    """Telegram bot handlers with proper lifecycle management."""

    def __init__(self):
        """Initialize bot handlers and services."""
        logger.info("initializing_bot_handlers")

        # Initialize services
        self.transcription = TranscriptionService()
        self.llm = LLMService()
        self.embedding = EmbeddingService()
        self.vector_store = VectorStore()

        # Database engine and session factory
        self.engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Check connections before use
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("bot_handlers_initialized")

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        logger.info("cleaning_up_bot_handlers")

        try:
            # Close database engine
            await self.engine.dispose()
            logger.info("database_engine_disposed")
        except Exception as e:
            logger.error("cleanup_error", error=str(e))

    def _check_user_allowed(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot."""
        allowed_users = settings.allowed_user_ids
        if not allowed_users:
            return True  # Allow all if not configured

        return str(user_id) in allowed_users

    async def start_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        user = update.effective_user
        logger.info("start_command", user_id=user.id, username=user.username)

        if not self._check_user_allowed(user.id):
            await update.message.reply_text(
                "❌ Sorry, you're not authorized to use this bot."
            )
            return

        welcome_message = f"""👋 Hello {user.first_name}!

I'm your AI-powered file assistant using **Gemini 2.5 Flash**.

**What I can do:**
📄 Search through your files by content
📋 List files by name or type
📊 Get statistics about your files
🔍 Find specific information (like "question 8")
📤 Send you actual files (up to 50MB)
💬 Remember our conversation
🎤 Understand voice messages

**Commands:**
/start - Show this message
/help - Get detailed help
/stats - Show file statistics
/clear - Clear conversation history

**Quick Examples:**
• "Find documents about AI"
• "What's the answer to question 8?"
• "Send me report.pdf"
• "List all PDF files"

Just send me a message or voice note! 🚀
"""

        await update.message.reply_text(welcome_message)

    async def help_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command."""
        help_text = """**📖 How to Use Me**

**1️⃣ Search Files:**
   • "Find documents about machine learning"
   • "Show me files with sales data"
   • "What files mention Python?"

**2️⃣ Get Specific Content:**
   • "What's question 8 in exam.pdf?"
   • "Show me section 3 from manual"
   • "Find the definition of AI"

**3️⃣ Send Files:**
   • "Send me report.pdf"
   • "Give me the spreadsheet"
   • "Share presentation.pptx"

**4️⃣ List & Browse:**
   • "List all PDF files"
   • "Show files named report"
   • "What files are larger than 10MB?"

**5️⃣ Get Information:**
   • "How many files do you have?"
   • "Show file statistics"
   • "/stats" command

**💡 Pro Tips:**
✓ Speak clearly in voice messages
✓ I remember our conversation context
✓ Be specific for better results
✓ I can send files up to 50MB
✓ Use /clear to start fresh

**🤖 Powered by Google Gemini 2.5 Flash**
"""

        await update.message.reply_text(help_text)

    async def stats_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /stats command."""
        user = update.effective_user

        if not self._check_user_allowed(user.id):
            return

        status_msg = await update.message.reply_text("📊 Fetching statistics...")

        try:
            async with self.async_session() as session:
                tools = AgentTools(session, self.embedding, self.vector_store)
                stats = await tools.get_file_stats()

            await status_msg.edit_text(stats)

        except Exception as e:
            logger.error("stats_command_error", error=str(e), exc_info=True)
            await status_msg.edit_text(
                f"❌ Error fetching statistics: {str(e)[:100]}"
            )

    async def clear_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /clear command to reset conversation."""
        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        if not self._check_user_allowed(user.id):
            return

        try:
            async with self.async_session() as session:
                memory = ConversationMemory(session)
                await memory.initialize()
                await memory.clear_conversation(str(user.id), chat_id)
                await memory.close()

            await update.message.reply_text(
                "✅ Conversation history cleared! Starting fresh."
            )

        except Exception as e:
            logger.error("clear_command_error", error=str(e), exc_info=True)
            await update.message.reply_text("❌ Error clearing history.")

    async def _send_files_to_user(
        self,
        update: Update,
        files_to_send: list,
    ) -> None:
        """Send files to user via Telegram."""
        for file_info in files_to_send:
            try:
                file_path = Path(file_info["path"])

                if not file_path.exists():
                    await update.message.reply_text(
                        f"❌ File not found: {file_info['name']}"
                    )
                    continue

                # Notify user
                await update.message.reply_text(
                    f"📤 Sending: {file_info['name']} ({file_info.get('size_mb', 0):.2f}MB)"
                )

                # Send file
                with open(file_path, 'rb') as f:
                    await update.message.reply_document(
                        document=f,
                        filename=file_info['name'],
                        caption=f"📄 {file_info['name']}"
                    )

                logger.info(
                    "file_sent_successfully",
                    file=file_info['name'],
                    user_id=update.effective_user.id,
                )

            except Exception as e:
                logger.error(
                    "file_send_error",
                    file=file_info.get('name'),
                    error=str(e),
                    exc_info=True,
                )
                await update.message.reply_text(
                    f"❌ Error sending {file_info.get('name')}: {str(e)[:100]}"
                )

    async def handle_text_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle text messages from users."""
        user = update.effective_user
        chat_id = str(update.effective_chat.id)
        message_text = update.message.text

        logger.info(
            "text_message_received",
            user_id=user.id,
            message_preview=message_text[:50],
        )

        if not self._check_user_allowed(user.id):
            await update.message.reply_text("❌ You're not authorized.")
            return

        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)

        try:
            async with self.async_session() as session:
                # Initialize memory
                memory = ConversationMemory(session)
                await memory.initialize()

                # Create tools and agent
                tools = AgentTools(session, self.embedding, self.vector_store)
                agent = FileAssistantAgent(tools, memory, self.llm)

                # Run agent
                result = await agent.run(
                    user_query=message_text,
                    user_id=str(user.id),
                    chat_id=chat_id,
                )

                await memory.close()

            # Extract response and files
            response = result["response"]
            files_to_send = result.get("files_to_send", [])

            # Send response (split if too long)
            max_length = 4000
            if len(response) > max_length:
                chunks = [
                    response[i:i+max_length]
                    for i in range(0, len(response), max_length)
                ]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(response)

            # Send files if any
            if files_to_send:
                logger.info("sending_files_to_user", count=len(files_to_send))
                await self._send_files_to_user(update, files_to_send)

        except Exception as e:
            logger.error(
                "text_message_handler_error",
                error=str(e),
                exc_info=True,
            )
            await update.message.reply_text(
                "❌ Sorry, I encountered an error. Please try again.\n\n"
                "Use /help for guidance or /clear to reset."
            )

    async def handle_voice_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle voice messages with transcription."""
        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        logger.info("voice_message_received", user_id=user.id)

        if not self._check_user_allowed(user.id):
            await update.message.reply_text("❌ You're not authorized.")
            return

        voice = update.message.voice
        size_mb = voice.file_size / (1024 * 1024)
        duration = voice.duration

        logger.info(
            "voice_details",
            size_mb=f"{size_mb:.2f}",
            duration_sec=duration,
        )

        # Check size limit
        if size_mb > settings.telegram_max_voice_size_mb:
            await update.message.reply_text(
                f"❌ Voice message too large ({size_mb:.1f}MB). "
                f"Maximum: {settings.telegram_max_voice_size_mb}MB."
            )
            return

        # Notify user
        status_msg = await update.message.reply_text(
            "🎤 Transcribing your voice message...\n"
            "(This may take a moment)"
        )

        try:
            # Download voice
            file = await voice.get_file()
            voice_bytes = await file.download_as_bytearray()

            # Transcribe
            temp_dir = Path("./temp")
            temp_dir.mkdir(exist_ok=True)

            transcript = await self.transcription.transcribe_bytes(
                bytes(voice_bytes),
                temp_dir,
            )

            logger.info(
                "voice_transcribed",
                transcript_preview=transcript[:100],
                length=len(transcript),
            )

            # Update status
            await status_msg.edit_text(
                f"📝 *Transcribed:* {transcript}\n\n⏳ Processing...",
                parse_mode="Markdown",
            )

            # Process as text
            await update.message.chat.send_action(ChatAction.TYPING)

            async with self.async_session() as session:
                memory = ConversationMemory(session)
                await memory.initialize()

                tools = AgentTools(session, self.embedding, self.vector_store)
                agent = FileAssistantAgent(tools, memory, self.llm)

                result = await agent.run(
                    user_query=transcript,
                    user_id=str(user.id),
                    chat_id=chat_id,
                )

                await memory.close()

            # Send response
            response = result["response"]
            files_to_send = result.get("files_to_send", [])

            await update.message.reply_text(response)

            # Send files if any
            if files_to_send:
                await self._send_files_to_user(update, files_to_send)

        except Exception as e:
            logger.error(
                "voice_handler_error",
                error=str(e),
                exc_info=True,
            )
            await update.message.reply_text(
                "❌ Sorry, couldn't process your voice message.\n"
                "Please try again or send a text message.\n\n"
                f"Error: {str(e)[:100]}"
            )

    def register_handlers(self, application) -> None:
        """Register all command and message handlers."""
        logger.info("registering_telegram_handlers")

        # Command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("clear", self.clear_command))

        # Message handlers
        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_text_message
            )
        )
        application.add_handler(
            MessageHandler(filters.VOICE, self.handle_voice_message)
        )

        logger.info("telegram_handlers_registered_successfully")
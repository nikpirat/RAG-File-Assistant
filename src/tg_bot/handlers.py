"""Telegram bot message handlers."""
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, CommandHandler, filters
from telegram.constants import ChatAction

from src.config.settings import settings
from src.config.logging_config import get_logger
from src.services.transcription import TranscriptionService
from src.services.llm import LLMService
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.agent.tools import AgentTools
from src.agent.memory import ConversationMemory
from src.agent.graph import FileAssistantAgent
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

logger = get_logger(__name__)


class BotHandlers:
    """Telegram bot handlers."""

    def __init__(self):
        # Initialize services
        self.transcription = TranscriptionService()
        self.llm = LLMService()
        self.embedding = EmbeddingService()
        self.vector_store = VectorStore()

        # Database
        self.engine = create_async_engine(settings.database_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("bot_handlers_initialized")

    def _check_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use bot."""
        allowed_users = settings.allowed_user_ids
        if not allowed_users:
            return True  # Allow all if not configured

        return str(user_id) in allowed_users

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        logger.info("start_command", user_id=user.id, username=user.username)

        if not self._check_user_allowed(user.id):
            await update.message.reply_text("Sorry, you're not authorized to use this bot.")
            return

        welcome_message = f"""👋 Hello {user.first_name}!

I'm your intelligent file assistant. I can help you:

📄 Search through your files by content
📋 List files by name or type
📊 Get statistics about your files
🔍 Find specific information in documents
💬 Remember our conversation context

**Commands:**
/start - Show this message
/help - Get help
/stats - Show file statistics
/clear - Clear conversation history

Just send me a message or voice note with your question!
"""

        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """**How to use me:**

1️⃣ **Search files:**
   "Find documents about machine learning"
   "Show me files containing sales data"

2️⃣ **List files:**
   "List all PDF files"
   "Show files named report"

3️⃣ **Get info:**
   "What files do you have?"
   "How many documents are indexed?"

4️⃣ **Ask questions:**
   "What does the contract say about termination?"
   "Summarize the quarterly report"

💡 **Tips:**
- You can send voice messages
- I remember our conversation
- Be specific for better results
- Use /clear to start fresh
"""

        await update.message.reply_text(help_text)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        user = update.effective_user

        if not self._check_user_allowed(user.id):
            return

        await update.message.reply_text("📊 Fetching statistics...")

        async with self.async_session() as session:
            tools = AgentTools(session, self.embedding, self.vector_store)
            stats = await tools.get_file_stats()
            await update.message.reply_text(stats)

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command."""
        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        if not self._check_user_allowed(user.id):
            return

        async with self.async_session() as session:
            memory = ConversationMemory(session)
            await memory.initialize()
            await memory.clear_conversation(str(user.id), chat_id)
            await memory.close()

        await update.message.reply_text("✅ Conversation history cleared!")

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages."""
        user = update.effective_user
        chat_id = str(update.effective_chat.id)
        message_text = update.message.text

        logger.info("text_message", user_id=user.id, message=message_text[:50])

        if not self._check_user_allowed(user.id):
            await update.message.reply_text("Sorry, you're not authorized.")
            return

        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)

        try:
            async with self.async_session() as session:
                # Create agent
                memory = ConversationMemory(session)
                await memory.initialize()

                tools = AgentTools(session, self.embedding, self.vector_store)
                agent = FileAssistantAgent(tools, memory, self.llm)

                # Run agent
                response = await agent.run(
                    user_query=message_text,
                    user_id=str(user.id),
                    chat_id=chat_id,
                )

                await memory.close()

            # Send response (split if too long)
            if len(response) > 4000:
                # Split into chunks
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(response)

        except Exception as e:
            logger.error("message_handling_failed", error=str(e), exc_info=True)
            await update.message.reply_text(
                "Sorry, I encountered an error processing your request. "
                "Please try again or use /help for guidance."
            )

    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages."""
        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        logger.info("voice_message", user_id=user.id)

        if not self._check_user_allowed(user.id):
            await update.message.reply_text("Sorry, you're not authorized.")
            return

        # Check file size
        voice = update.message.voice
        size_mb = voice.file_size / (1024 * 1024)

        if size_mb > settings.telegram_max_voice_size_mb:
            await update.message.reply_text(
                f"Voice message too large ({size_mb:.1f}MB). "
                f"Maximum is {settings.telegram_max_voice_size_mb}MB."
            )
            return

        await update.message.reply_text("🎤 Transcribing voice message...")

        try:
            # Download voice file
            file = await voice.get_file()
            voice_bytes = await file.download_as_bytearray()

            # Transcribe
            temp_dir = Path("C:/Users/yakuz/Downloads/voiceRec")
            temp_dir.mkdir(exist_ok=True)

            transcript = await self.transcription.transcribe_bytes(
                bytes(voice_bytes),
                temp_dir,
            )

            logger.info("voice_transcribed", transcript=transcript[:100])

            # Show transcript
            await update.message.reply_text(f"📝 Transcribed: {transcript}\n\nProcessing...")

            # Process as text message
            async with self.async_session() as session:
                memory = ConversationMemory(session)
                await memory.initialize()

                tools = AgentTools(session, self.embedding, self.vector_store)
                agent = FileAssistantAgent(tools, memory, self.llm)

                response = await agent.run(
                    user_query=transcript,
                    user_id=str(user.id),
                    chat_id=chat_id,
                )

                await memory.close()

            await update.message.reply_text(response)

        except Exception as e:
            logger.error("voice_handling_failed", error=str(e), exc_info=True)
            await update.message.reply_text(
                "Sorry, I couldn't process your voice message. "
                "Please try again or send a text message."
            )

    def register_handlers(self, application):
        """Register all handlers with the application."""

        # Commands
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("clear", self.clear_command))

        # Messages
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message)
        )
        application.add_handler(
            MessageHandler(filters.VOICE, self.handle_voice_message)
        )

        logger.info("handlers_registered")
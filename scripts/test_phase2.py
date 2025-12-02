"""Test script for Phase 2 components."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.services.llm import LLMService
from src.services.transcription import TranscriptionService
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.agent.tools import AgentTools
from src.agent.memory import ConversationMemory
from src.agent.graph import FileAssistantAgent
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


async def test_llm():
    """Test LLM service."""
    logger.info("=== Testing LLM Service ===")

    try:
        llm = LLMService()
        response = await llm.generate(
            prompt="What is machine learning? Explain in one sentence.",
            temperature=0.1,
        )
        logger.info(f"✓ LLM Response: {response}")
        return True
    except Exception as e:
        logger.error(f"✗ LLM test failed: {e}")
        return False


async def test_agent():
    """Test agent with tools."""
    logger.info("\n=== Testing Agent ===")

    try:
        # Setup database
        engine = create_async_engine(settings.database_url, echo=False)
        async_session = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with async_session() as session:
            # Initialize services
            llm = LLMService()
            embedding = EmbeddingService()
            vector_store = VectorStore()

            # Create tools and memory
            tools = AgentTools(session, embedding, vector_store)
            memory = ConversationMemory(session)
            await memory.initialize()

            # Create agent
            agent = FileAssistantAgent(tools, memory, llm)

            # Test query
            test_query = "Show me file statistics"
            logger.info(f"Test Query: {test_query}")

            response = await agent.run(
                user_query=test_query,
                user_id="test_user",
                chat_id="test_chat",
            )

            logger.info(f"✓ Agent Response:\n{response}")

            await memory.close()
            await engine.dispose()

        return True

    except Exception as e:
        logger.error(f"✗ Agent test failed: {e}", exc_info=True)
        return False


async def test_memory():
    """Test conversation memory."""
    logger.info("\n=== Testing Memory ===")

    try:
        engine = create_async_engine(settings.database_url, echo=False)
        async_session = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            memory = ConversationMemory(session)
            await memory.initialize()

            # Add test messages
            await memory.add_message(
                user_id="test_user",
                chat_id="test_chat",
                role="user",
                content="Hello, what files do you have?",
            )

            await memory.add_message(
                user_id="test_user",
                chat_id="test_chat",
                role="assistant",
                content="I have several PDF and TXT files indexed.",
            )

            # Retrieve messages
            messages = await memory.get_recent_messages(
                user_id="test_user",
                chat_id="test_chat",
                limit=10,
            )

            logger.info(f"✓ Retrieved {len(messages)} messages")
            for msg in messages:
                logger.info(f"  {msg['role']}: {msg['content'][:50]}...")

            # Clear conversation
            await memory.clear_conversation("test_user", "test_chat")
            logger.info("✓ Conversation cleared")

            await memory.close()
            await engine.dispose()

        return True

    except Exception as e:
        logger.error(f"✗ Memory test failed: {e}", exc_info=True)
        return False


async def test_transcription():
    """Test voice transcription (if you have a test audio file)."""
    logger.info("\n=== Testing Transcription ===")

    # Check if test audio exists
    test_audio = Path("/tmp/test_audio.ogg")

    if not test_audio.exists():
        logger.info("⊘ Skipping transcription test (no test audio file)")
        logger.info("  Create /tmp/test_audio.ogg to test this feature")
        return True

    try:
        transcription = TranscriptionService()
        text = await transcription.transcribe_file(test_audio)
        logger.info(f"✓ Transcription: {text}")
        return True
    except Exception as e:
        logger.error(f"✗ Transcription test failed: {e}")
        return False


async def main():
    """Run all tests."""
    configure_logging()
    logger.info("=== Phase 2 Component Tests ===\n")

    results = {}

    # Test 1: LLM
    results["llm"] = await test_llm()

    # Test 2: Memory
    results["memory"] = await test_memory()

    # Test 3: Agent
    results["agent"] = await test_agent()

    # Test 4: Transcription (optional)
    results["transcription"] = await test_transcription()

    # Summary
    logger.info("\n=== Test Summary ===")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ All tests passed!")
    else:
        logger.info("\n✗ Some tests failed")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
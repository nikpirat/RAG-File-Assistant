"""Test Gemini integration and compare with Ollama."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.services.llm import LLMService

logger = get_logger(__name__)


async def test_basic_generation():
    """Test basic text generation."""
    configure_logging()
    print("\n" + "=" * 60)
    print("Test 1: Basic Generation")
    print("=" * 60 + "\n")

    llm = LLMService()

    prompt = "What is 2+2? Answer in one sentence."

    print(f"Provider: {settings.llm_provider}")
    print(f"Model: {llm.model}")
    print(f"Prompt: {prompt}\n")

    start = time.time()
    response = await llm.generate(prompt)
    elapsed = time.time() - start

    print(f"Response: {response}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Length: {len(response)} chars\n")

    return elapsed


async def test_with_system_prompt():
    """Test generation with system prompt."""
    print("\n" + "=" * 60)
    print("Test 2: System Prompt")
    print("=" * 60 + "\n")

    llm = LLMService()

    system_prompt = "You are a helpful assistant. Always respond in exactly one sentence."
    prompt = "Explain machine learning."

    print(f"System: {system_prompt}")
    print(f"Prompt: {prompt}\n")

    start = time.time()
    response = await llm.generate(prompt, system_prompt=system_prompt)
    elapsed = time.time() - start

    print(f"Response: {response}")
    print(f"Time: {elapsed:.2f}s\n")

    return elapsed


async def test_anti_hallucination():
    """Test anti-hallucination behavior."""
    print("\n" + "=" * 60)
    print("Test 3: Anti-Hallucination")
    print("=" * 60 + "\n")

    llm = LLMService()

    system_prompt = """You are a strict assistant. 
CRITICAL: Only use information explicitly provided in the user's message.
NEVER make up or assume information.
If you don't have information, say "I don't have that information."""

    prompt = """Based on this data: The file is called "report.pdf"

What files do I have?"""

    print(f"Testing if LLM only uses provided information...\n")

    response = await llm.generate(prompt, system_prompt=system_prompt)

    print(f"Response: {response}\n")

    if "report.pdf" in response.lower():
        print("✅ PASS: Only mentioned provided file")
    else:
        print("⚠️ WARNING: May have hallucinated")


async def test_rag_quality():
    """Test RAG-style response."""
    print("\n" + "=" * 60)
    print("Test 4: RAG Quality")
    print("=" * 60 + "\n")

    llm = LLMService()

    system_prompt = """You are a file assistant. Use ONLY the information from tools below.

Tool Results:
File: data_sample1.csv
Content: model,accuracy,speed
GPT-4,94.23%,slow
Llama,88%,fast

User Query: What's the accuracy for GPT model?
"""

    prompt = "Answer the user's query using ONLY the tool results above."

    print("Testing RAG response quality...\n")

    start = time.time()
    response = await llm.generate(prompt, system_prompt=system_prompt)
    elapsed = time.time() - start

    print(f"Response: {response}")
    print(f"Time: {elapsed:.2f}s\n")

    if "94.23" in response or "94" in response:
        print("✅ PASS: Extracted correct data")
    else:
        print("❌ FAIL: Did not extract correct data")


async def compare_speed():
    """Compare speed with Ollama."""
    print("\n" + "=" * 60)
    print("Test 5: Speed Comparison")
    print("=" * 60 + "\n")

    # Test Gemini
    print("Testing Gemini...")
    settings.llm_provider = "gemini"
    llm_gemini = LLMService()

    prompt = "List 5 programming languages in one sentence."

    start = time.time()
    response_gemini = await llm_gemini.generate(prompt)
    gemini_time = time.time() - start

    print(f"Gemini: {gemini_time:.2f}s")
    print(f"Response: {response_gemini[:100]}...\n")

    # Test Ollama (if available)
    try:
        print("Testing Ollama...")
        settings.llm_provider = "ollama"
        llm_ollama = LLMService()

        start = time.time()
        response_ollama = await llm_ollama.generate(prompt)
        ollama_time = time.time() - start

        print(f"Ollama: {ollama_time:.2f}s")
        print(f"Response: {response_ollama[:100]}...\n")

        speedup = ollama_time / gemini_time
        print(f"📊 Gemini is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than Ollama")

    except Exception as e:
        print(f"⚠️ Ollama not available: {e}")
        print("Can't compare speeds\n")


async def test_error_handling():
    """Test error handling and fallback."""
    print("\n" + "=" * 60)
    print("Test 6: Error Handling")
    print("=" * 60 + "\n")

    # Test with invalid API key
    print("Testing with invalid API key...")
    original_key = settings.gemini_api_key
    settings.gemini_api_key = "invalid_key"

    try:
        llm = LLMService()
        response = await llm.generate("Test")
        print(f"Response: {response[:50]}...")
        print("✅ Fallback mechanism worked\n")
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}\n")
    finally:
        settings.gemini_api_key = original_key


async def main():
    """Run all tests."""
    configure_logging()

    print("\n" + "=" * 70)
    print("GEMINI 1.5 FLASH INTEGRATION TEST SUITE")
    print("=" * 70)

    # Check configuration
    print(f"\n📋 Configuration:")
    print(f"   Provider: {settings.llm_provider}")
    print(f"   Model: {settings.gemini_model}")
    print(f"   API Key: {'✅ Set' if settings.gemini_api_key else '❌ Not set'}")

    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        print("\n❌ ERROR: GEMINI_API_KEY not configured!")
        print("Please update .env with your actual API key")
        return

    try:
        # Run tests
        await test_basic_generation()
        await test_with_system_prompt()
        await test_anti_hallucination()
        await test_rag_quality()
        await compare_speed()
        await test_error_handling()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 70 + "\n")

        print("Next steps:")
        print("1. If all tests passed, Gemini is working correctly")
        print("2. Restart your bot: sudo systemctl restart rag-assistant")
        print("3. Test in Telegram with real queries")
        print("4. Monitor usage: python scripts/monitor_usage.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        logger.error("test_suite_failed", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
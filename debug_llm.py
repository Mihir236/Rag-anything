import os
import sys
import asyncio
from dotenv import load_dotenv

# Load env vars
load_dotenv(override=True)

# Add current directory to path
sys.path.append(os.getcwd())

from lightrag.llm.openai import openai_complete_if_cache, openai_embed

async def test_llm():
    api_key = os.environ.get("LLM_BINDING_API_KEY")
    base_url = os.environ.get("LLM_BINDING_HOST")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    print(f"Testing LLM: {model}")
    print(f"Base URL: {base_url}")
    print(f"API Key present: {bool(api_key)}")

    try:
        response = await openai_complete_if_cache(
            model,
            "What is 2+2?",
            api_key=api_key,
            base_url=base_url
        )
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"LLM Error: {e}")

async def test_embedding():
    api_key = os.environ.get("LLM_BINDING_API_KEY")
    base_url = os.environ.get("LLM_BINDING_HOST")
    model = os.environ.get("EMBEDDING_MODEL", "nvidia/nv-embed-v1")

    print(f"\nTesting Embedding: {model}")
    try:
        embedding = await openai_embed(
            ["Hello world"],
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        print(f"Embedding success. Shape/Len: {len(embedding)}")
        if len(embedding) > 0:
            print(f"First vector dim: {len(embedding[0])}")
    except Exception as e:
        print(f"Embedding Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm())
    asyncio.run(test_embedding())

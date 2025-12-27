import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

async def test_raw_embedding():
    api_key = os.environ.get("LLM_BINDING_API_KEY")
    base_url = os.environ.get("LLM_BINDING_HOST")
    model = os.environ.get("EMBEDDING_MODEL", "nvidia/nv-embed-v1")

    print(f"Testing Raw Embedding: {model}")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = await client.embeddings.create(
            input="Hello world",
            model=model
        )
        emb = response.data[0].embedding
        print(f"Success! Embedding len: {len(emb)}")
        return len(emb)
    except Exception as e:
        print(f"Raw Embedding Error: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_raw_embedding())

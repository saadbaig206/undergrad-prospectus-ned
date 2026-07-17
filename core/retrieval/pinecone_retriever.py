import asyncio
import os
import threading

from pinecone import AsyncPinecone


pinecone_client = None
pinecone_index = None
_pinecone_lock = threading.Lock()

PINECONE_INDEX_NAME = "rag-chatbot-index"
PINECONE_TOP_K = int(os.getenv("PINECONE_TOP_K", "12"))

def get_pinecone_index():

    global pinecone_client
    global pinecone_index

    if pinecone_index is None:

        print("[PINECONE] Initializing Pinecone Index client...")

        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not configured.")

        if pinecone_client is None:
            pinecone_client = AsyncPinecone(api_key=api_key)

        host = os.getenv("PINECONE_INDEX_HOST")

        if host:
            print(f"[PINECONE] Initializing Index client using direct host: {host}")
            pinecone_index = pinecone_client.IndexAsyncio(host=host)
        else:
            print("[PINECONE] PINECONE_INDEX_HOST not set. Index host will be resolved via control plane API call.")
            pinecone_index = pinecone_client.IndexAsyncio(name=PINECONE_INDEX_NAME)

    else:
        print("[PINECONE] Reusing existing Pinecone Index client.")

    return pinecone_index


async def retrieve_from_pinecone(
    query_vector,
    top_k,
    metadata_filter=None,
):
    index = get_pinecone_index()

    response = await index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )

    return response.matches
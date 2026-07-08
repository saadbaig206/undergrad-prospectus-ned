import os
import sys
from functools import lru_cache
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from core.intent_detector import IntentDetector
import asyncio
import httpx

import pydantic
import pydantic.v1 as pydantic_v1

try:
    import langchain_core.pydantic_v1
except ImportError:
    class MockPydanticV1:
        BaseModel = pydantic_v1.BaseModel
        Field = pydantic_v1.Field
        root_validator = pydantic_v1.root_validator
        SecretStr = pydantic.SecretStr
    sys.modules["langchain_core.pydantic_v1"] = MockPydanticV1

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.embeddings import embed_query

load_dotenv()

PINECONE_INDEX_NAME = "rag-chatbot-index"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")
SEAT_DIST_FILE_LINK = os.getenv(
    "SEAT_DIST_FILE_LINK",
    f"{API_BASE_URL}/seat_distribution.pdf"
)

groq_client = None
langchain_llm = None

pinecone_client = None
pinecone_index = None

_global_httpx_client = None

def get_global_httpx_client():
    global _global_httpx_client
    if _global_httpx_client is None:
        _global_httpx_client = httpx.AsyncClient(timeout=10.0)
    return _global_httpx_client

intent_detector = IntentDetector()

@lru_cache(maxsize=1024)
def cached_embedding(text: str):
    return embed_query(text)

def get_llm():

    global langchain_llm

    if langchain_llm is None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY not configured.")

        langchain_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=api_key,
            streaming=True,
        )

    return langchain_llm

langchain_fast_llm = None

def get_fast_llm():
    global langchain_fast_llm

    if langchain_fast_llm is None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY not configured.")

        langchain_fast_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=api_key,
            streaming=False,
        )

    return langchain_fast_llm

def get_pinecone_index():
    global pinecone_client
    global pinecone_index

    if pinecone_index is None:
        print("🔍 [PINECONE] Initializing Pinecone Index client...")
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not configured.")

        if pinecone_client is None:
            pinecone_client = Pinecone(api_key=api_key)

        host = os.getenv("PINECONE_INDEX_HOST")
        if host:
            print(f"🔍 [PINECONE] Initializing Index client using direct host: {host}")
            pinecone_index = pinecone_client.Index(
                PINECONE_INDEX_NAME,
                host=host
            )
        else:
            print("⚠️ [PINECONE] PINECONE_INDEX_HOST not set. Index host will be resolved via control plane API call.")
            pinecone_index = pinecone_client.Index(
                PINECONE_INDEX_NAME
            )
    else:
        print("🔍 [PINECONE] Reusing existing Pinecone Index client.")

    return pinecone_index

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert conversation summarizer.

Summarize the conversation while preserving:

- User preferences
- Important facts
- Previously discussed university topics
- Follow-up context
- Unresolved questions

Return ONLY the summary.
""",
        ),
        (
            "human",
            "{history}",
        ),
    ]
)

summary_chain = (
    summary_prompt
    | get_fast_llm()
    | StrOutputParser()
)


def clean_history_for_llm(chat_history: list) -> list:
    """
    Cleans and truncates assistant responses in the chat history
    to avoid exceeding LLM context and rate limits (e.g. Groq TPM limits).
    """
    cleaned = []
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            if len(content) > 250:
                content = content[:250] + "... [truncated]"
        cleaned.append({"role": role, "content": content})
    return cleaned

async def summarize_history(history_to_summarize: list) -> str:

    if not history_to_summarize:
        return ""

    history = "\n".join(
        f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
        for msg in history_to_summarize
    )

    try:
        res = await summary_chain.ainvoke(
            {
                "history": history
            }
        )
        return res.strip()

    except Exception as e:
        print(f"History summarization failed: {e}")
        return ""
    

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You rewrite follow-up user questions into standalone search queries.

Rules:
- Preserve the user's intent.
- Resolve references such as:
  - it
  - this
  - that
  - they
  - same
  - above
  - previous
- Never answer the question.
- Never invent information.
- Return ONLY the rewritten standalone query.
""",
        ),
        (
            "human",
            """
Conversation:

{history}

Latest User Message:

{query}
""",
        ),
    ]
)

rewrite_chain = (
    rewrite_prompt
    | get_fast_llm()
    | StrOutputParser()
)

async def condense_query(chat_history: list, user_query: str) -> str:
    # Only keep the last 6 messages of history for resolving conversational references (prevents rate limits)
    chat_history = clean_history_for_llm(chat_history[-6:])

    if not chat_history:
        return user_query

    try:
        history = "\n".join(
            f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history
        )

        rewritten = await rewrite_chain.ainvoke(
            {
                "history": history,
                "query": user_query,
            }
        )

        return rewritten.strip() or user_query

    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return user_query
        

async def route_chat_stream(user_query: str, chat_history: list = None):
    """Routes the query dynamically based on content type and streams the response."""
    import time
    chat_history = clean_history_for_llm(chat_history or [])
    
    t0 = time.time()
    if chat_history and needs_query_rewrite(user_query):
        standalone_query = await condense_query(chat_history, user_query)
        print(f"⏱️ [LATENCY] Query Rewrite: {time.time() - t0:.4f}s")
    else:
        standalone_query = user_query
        print("⏱️ [LATENCY] Query Rewrite: Skipped")
    
    t1 = time.time()
    try:
        query_vector = cached_embedding(standalone_query)
        print(f"⏱️ [LATENCY] Embedding Generation: {time.time() - t1:.4f}s")
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        query_vector = None
        
    t2 = time.time()
    intent = intent_detector.classify(standalone_query, query_vector=query_vector)
    print(f"⏱️ [LATENCY] Intent Classification: {time.time() - t2:.4f}s")
    
    if intent == "SEAT":
        yield f"For the complete and accurate Seat Distribution Matrix, please refer to the official document: [Seat Distribution PDF]({SEAT_DIST_FILE_LINK})."
        return
        
    if intent == "GENERAL":
        system_base = "You are a professional, helpful university admission assistant. Answer the user's query directly and helpfully."
    else: # RAG
        t3 = time.time()
        try:
            print(f"🔍 [DEBUG] query_vector is None: {query_vector is None}")
            if query_vector is None:
                t_embed = time.time()
                query_vector = cached_embedding(standalone_query)
                print(f"⏱️ [DEBUG] Late-binding embedding generation: {time.time() - t_embed:.4f}s")
            
            t_query = time.time()
            api_key = os.getenv("PINECONE_API_KEY")
            host = os.getenv("PINECONE_INDEX_HOST")
            if not host:
                raise ValueError("PINECONE_INDEX_HOST is not configured.")
            url = f"{host}/query" if host.startswith("https://") else f"https://{host}/query"
            
            client = get_global_httpx_client()
            response = await client.post(
                url,
                json={
                    "vector": query_vector,
                    "topK": 4,
                    "includeMetadata": True,
                    "includeValues": False
                },
                headers={
                    "Api-Key": api_key,
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            matches = data.get("matches", [])
            print(f"⏱️ [DEBUG] Raw HTTPX REST query execution: {time.time() - t_query:.4f}s")
            
            context_chunks = []
            for m in matches:
                metadata = m.get("metadata")
                if metadata and "text" in metadata:
                    context_chunks.append(metadata["text"])
            context = "\n---\n".join(context_chunks)
            print(f"⏱️ [LATENCY] Pinecone Retrieval: {time.time() - t3:.4f}s")
        except Exception as e:
            print(f"Pinecone retrieval failed: {e}")
            context = "No context available."

        system_base = f"""You are a highly capable academic academic assistant. Your goal is to answer the user's question as thoroughly and helpful as possible using the retrieved context from the official Undergraduate Prospectus.
        
Retrieved Context:
{context}

Instructions:
1. Provide a direct, detailed, and clear answer.
2. Structure your response with bullet points and bold text for readability.
3. If the context does not contain a specific detail (like exact location, dimensions, or amenities), do NOT state that the prospectus or document does not contain this information, lacks detail, or fails to provide it. Avoid negative disclaimers entirely (e.g., do NOT write "However, the prospectus does not provide detailed information...").
4. Always maintain a professional, encouraging, and authoritative tone. Focus exclusively on presenting the positive information that IS available. If a requested detail is missing, simply explain what the document states generally about the topic and guide the user to check official university channels (website/admissions office) for specific updates, without pointing out what is missing from the prospectus.
5. Note that university departments (e.g., Computer Science, Software Engineering, Civil Engineering, Polymer Engineering) do not have individual deans; they fall under larger Faculties led by Deans. If the user asks for the dean of a specific department, identify which Faculty that department belongs to in the context, and name the Dean of that Faculty (e.g., Prof. Dr. Saad Ahmed Qazi is the Dean of the Faculty of ECE which includes Computer Systems, Software Engineering, etc.).
6. Keep your responses completely evergreen. Do NOT mention specific years (such as '2025' or '2025-26') in your answers. Refer to the source document generally as 'the Undergraduate Prospectus' or 'the prospectus'."""

    # Limit active history to the last 10 messages (5 turns) to stay well within Groq TPM limits
    recent_history = chat_history[-10:]
    
    messages = [SystemMessage(content=system_base)]
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=user_query))

    # 5. Stream the response from ChatGroq LLM
    try:
        response = get_llm().astream(messages)
        async for chunk in response:
            content = chunk.content
            if content:
                yield content
    except Exception as e:
        err_msg = str(e).lower()
        if "connection" in err_msg or "timeout" in err_msg or "max retries" in err_msg or "unreachable" in err_msg or "resolv" in err_msg or "failed to establish" in err_msg:
            yield "⚠️ **Internet Connection Error:** Could not connect to the AI service. Please check your internet connection and try again."
        else:
            yield f"⚠️ **Chatbot Error:** {e}"

import re

def needs_query_rewrite(query: str) -> bool:
    """
    Returns True only if the user's message depends on previous context.
    """
    query = query.lower().strip()

    contextual_words = {
        "it", "its", "they", "them", "that", "those", "this", "these",
        "he", "she", "his", "her", "same", "above", "previous",
        "earlier", "former", "latter", "him", "himself", "herself", "themselves"
    }

    # Single-word questions that require context (e.g. "fees", "eligibility")
    continuation_keywords = {
        "fee", "fees", "eligibility", "criteria", "requirements", "dean", 
        "syllabus", "courses", "duration", "seats", "seat", "apply", 
        "admission", "admissions", "cutoff", "merit"
    }

    tokens = re.findall(r"\w+", query)
    if not tokens:
        return False

    # 1. Contains explicit contextual/pronoun words
    if any(word in contextual_words for word in tokens):
        return True

    # 2. Is a single word that is a known continuation query
    if len(tokens) == 1 and tokens[0] in continuation_keywords:
        return True

    # 3. Otherwise, do not rewrite. For example, "what is CS?" or "eligibility of CS"
    # are standalone and do not require conversational context to search.
    return False

if __name__ == "__main__":
    import asyncio
    async def test():
        print("Testing Route Chat Stream...")
        history = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
        print("\n--- Test 1: General Query ---")
        for chunk in route_chat_stream("Who are you?", history):
            print(chunk, end="", flush=True)
            
        print("\n\n--- Test 2: RAG Query ---")
        for chunk in route_chat_stream("What is the eligibility criteria for software engineering?"):
            print(chunk, end="", flush=True)
            
        print("\n\n--- Test 3: Seat Matrix ---")
        for chunk in route_chat_stream("Give me the seat distribution matrix please."):
            print(chunk, end="", flush=True)
        print()

    # PyMuPDF/Pinecone might run in event loop, let's run test
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())

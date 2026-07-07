import os
import sys
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from core.intent_detector import IntentDetector

import pydantic
import pydantic.v1 as pydantic_v1

# Monkeypatch langchain_core.pydantic_v1 for compatibility with langchain-groq under Pydantic V2
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


# Load environment variables
load_dotenv()

# --- Configuration ---
PINECONE_INDEX_NAME = "rag-chatbot-index"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
).rstrip("/")
SEAT_DIST_FILE_LINK = f"{API_BASE_URL}/seat_distribution.pdf"

groq_client = None
langchain_llm = None

pinecone_client = None
pinecone_index = None

intent_detector = IntentDetector()

@lru_cache(maxsize=1024)
def cached_embedding(text: str):
    return embed_query(text)

def get_llm():
    """
    LangChain ChatGroq instance.
    Used for:
    - Query rewriting
    - Conversation summarization
    - Final answer generation
    """

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
    """
    LangChain ChatGroq instance using a faster model.
    Used for query rewriting and conversation summarization to reduce latency.
    """
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

        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not configured.")

        if pinecone_client is None:
            pinecone_client = Pinecone(api_key=api_key)

        pinecone_index = pinecone_client.Index(
            PINECONE_INDEX_NAME
        )

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


def summarize_history(history_to_summarize: list) -> str:
    """
    Summarize older conversation history.
    Called only for very long conversations.
    """

    if not history_to_summarize:
        return ""

    history = "\n".join(
        f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
        for msg in history_to_summarize
    )

    try:
        return summary_chain.invoke(
            {
                "history": history
            }
        ).strip()

    except Exception as e:
        print(f"History summarization failed: {e}")
        return 
    

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

def condense_query(chat_history: list, user_query: str) -> str:
    """
    Rewrite a follow-up question into a standalone query.
    """

    if not chat_history:
        return user_query

    try:

        if len(chat_history) <= 20:

            history = "\n".join(
                f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
                for msg in chat_history
            )

        elif len(chat_history) <= 40:

            history = "\n".join(
                f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
                for msg in chat_history[-12:]
            )

        else:

            summary = summarize_history(chat_history[:-12])

            recent = "\n".join(
                f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
                for msg in chat_history[-12:]
            )

            history = (
                f"Conversation Summary:\n{summary}\n\n"
                f"Recent Conversation:\n{recent}"
            )

        rewritten = rewrite_chain.invoke(
            {
                "history": history,
                "query": user_query,
            }
        ).strip()

        return rewritten or user_query

    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return user_query
        

def route_chat_stream(user_query: str, chat_history: list = None):
    """Routes the query dynamically based on content type and streams the response."""
    import time
    chat_history = chat_history or []
    
    t0 = time.time()
    # 1. Condense query to a standalone search query if history exists
    if chat_history and needs_query_rewrite(user_query):
        standalone_query = condense_query(chat_history, user_query)
        print(f"⏱️ [LATENCY] Query Rewrite: {time.time() - t0:.4f}s")
    else:
        standalone_query = user_query
        print("⏱️ [LATENCY] Query Rewrite: Skipped")
    
    t1 = time.time()
    # Precompute query embedding once to reduce latency
    try:
        query_vector = cached_embedding(standalone_query)
        print(f"⏱️ [LATENCY] Embedding Generation: {time.time() - t1:.4f}s")
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        query_vector = None
        
    t2 = time.time()
    # 2. Classify the intent of the standalone query (passing precomputed embedding)
    intent = intent_detector.classify(standalone_query, query_vector=query_vector)
    print(f"⏱️ [LATENCY] Intent Classification: {time.time() - t2:.4f}s")
    
    if intent == "SEAT":
        yield f"For the complete and accurate Seat Distribution Matrix, please refer to the official document: [Seat Distribution PDF]({SEAT_DIST_FILE_LINK})."
        return
        
    # 3. Setup context-management system message
    if intent == "GENERAL":
        system_base = "You are a professional, helpful university admission assistant. Answer the user's query directly and helpfully."
    else: # RAG
        # Embed and retrieve from Pinecone
        t3 = time.time()
        try:
            if query_vector is None:
                query_vector = cached_embedding(standalone_query)
            results = get_pinecone_index().query(vector=query_vector, top_k=5, include_metadata=True)
            context_chunks = [match['metadata']['text'] for match in results['matches']]
            context = "\n---\n".join(context_chunks)
            print(f"⏱️ [LATENCY] Pinecone Retrieval: {time.time() - t3:.4f}s")
        except Exception as e:
            print(f"Pinecone retrieval failed: {e}")
            context = "No context available."

        system_base = f"""You are a highly capable academic academic assistant. Your goal is to answer the user's question as thoroughly and helpful as possible using the retrieved context from the official Undergraduate Prospectus 2025.
        
Retrieved Context:
{context}

Instructions:
1. Provide a direct, detailed, and clear answer.
2. Structure your response with bullet points and bold text for readability.
3. If the context does not contain the exact specific detail (like a date or specific value), do NOT say 'I don't know' or 'I don't have the information'. Instead, explain what the document states generally about the topic, cite the relevant section name/number from the context, and guide the user on how they can find the official update (e.g., checking the university's official website or admission office as mentioned in section X).
4. Always maintain a professional, encouraging, and authoritative tone. Avoid negative phrasing like 'I cannot answer' or 'this is not mentioned'."""

    # 4. Build the final messages array based on short vs long chat context rules
    messages = []
    
    if len(chat_history) <= 20:
        messages.append(SystemMessage(content=system_base))
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    elif len(chat_history) <= 40:
        messages.append(SystemMessage(content=system_base))
        for msg in chat_history[-12:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    else:
        summary = summarize_history(chat_history[:-12])
        system_prompt_with_summary = f"{system_base}\n\nSummary of earlier conversation:\n{summary}"
        messages.append(SystemMessage(content=system_prompt_with_summary))
        for msg in chat_history[-12:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
                
    messages.append(HumanMessage(content=user_query))

    # 5. Stream the response from ChatGroq LLM
    try:
        response = get_llm().stream(messages)
        for chunk in response:
            content = chunk.content
            if content:
                yield content
    except Exception as e:
        yield f"⚠️ Chatbot logic error occurred: {e}"
        



import re

def needs_query_rewrite(query: str) -> bool:
    """
    Returns True only if the user's message depends on previous context.
    """

    query = query.lower().strip()

    contextual_words = {

        "it",
        "its",
        "they",
        "them",
        "that",
        "those",
        "this",
        "these",
        "he",
        "she",
        "his",
        "her",
        "there",
        "same",
        "above",
        "previous",
        "earlier",
        "former",
        "latter"

    }

    tokens = re.findall(r"\w+", query)

    if any(word in contextual_words for word in tokens):
        return True

    if len(tokens) <= 3:
        return True

    return False


# --- Test Interface ---
if __name__ == "__main__":
    import asyncio
    # Simple test run
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

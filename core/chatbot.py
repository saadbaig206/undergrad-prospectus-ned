import os
import sys
from functools import lru_cache
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from core.intent_detector import IntentDetector
import asyncio
import httpx
import threading

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

def expand_query_abbreviations(query: str) -> str:
    if not query:
        return query
    
    # Common NED department abbreviations to full names
    abbreviations = {
        "eld": "Electronic Engineering",
        "el": "Electronic Engineering",
        "cis": "Computer & Information Systems Engineering",
        "cid": "Computer & Information Systems Engineering",
        "cs": "Computer Science",
        "se": "Software Engineering",
        "me": "Mechanical Engineering",
        "ee": "Electrical Engineering",
        "ce": "Civil Engineering",
        "pe": "Petroleum Engineering",
        "te": "Telecommunications Engineering",
        "be": "Biomedical Engineering",
        "bme": "Biomedical Engineering",
        "im": "Industrial & Manufacturing Engineering",
        "tx": "Textile Engineering",
        "ar": "Architecture",
        "arch": "Architecture",
    }
    
    import re
    expanded = query
    for abbrev, full_name in abbreviations.items():
        # Match only full words (case-insensitive)
        pattern = r"\b" + re.escape(abbrev) + r"\b"
        expanded = re.sub(pattern, full_name, expanded, flags=re.IGNORECASE)
    return expanded


groq_client = None
langchain_llm = None

pinecone_client = None
pinecone_index = None

_global_httpx_client = None

# ---------------------------------------------------------------------------
# In-memory response cache  (avoids repeated Pinecone + LLM round-trips for
# identical questions)
# ---------------------------------------------------------------------------
from functools import lru_cache as _lru_cache
import hashlib as _hashlib

_response_cache: dict = {}          # key → cached answer string
_CACHE_MAX = 512                    # max distinct cached entries

def _cache_key(query: str, history_len: int) -> str:
    """Cheap cache key: hash of (lowercased query, conversation depth)."""
    raw = f"{query.strip().lower()}|{history_len}"
    return _hashlib.md5(raw.encode()).hexdigest()

def _cache_get(query: str, history_len: int):
    return _response_cache.get(_cache_key(query, history_len))

def _cache_set(query: str, history_len: int, answer: str):
    if len(_response_cache) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = next(iter(_response_cache))
        _response_cache.pop(oldest, None)
    _response_cache[_cache_key(query, history_len)] = answer
# ---------------------------------------------------------------------------
_pinecone_index = None
_pinecone_lock = threading.Lock()

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                api_key = os.getenv("PINECONE_API_KEY")
                pc = Pinecone(api_key=api_key)
                _pinecone_index = pc.Index("rag-chatbot-index")
    return _pinecone_index

def get_global_httpx_client():
    global _global_httpx_client
    if _global_httpx_client is None:
        # Persistent connection pool: keeps TCP connections to Pinecone alive
        # so every request reuses an existing socket instead of re-doing TCP+TLS handshake.
        # keepalive_expiry=300 keeps connections pooled for 5 minutes of inactivity.
        # max_keepalive_connections=10 handles concurrent requests without reconnecting.
        _global_httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=25.0, write=10.0, pool=5.0),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=300.0,
            ),
        )
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

        model = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")
        langchain_llm = ChatGroq(
            model=model,
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

        model = os.getenv("FAST_LLM_MODEL_NAME", "llama-3.1-8b-instant")
        langchain_fast_llm = ChatGroq(
            model=model,
            temperature=0.1,
            groq_api_key=api_key,
            streaming=False,
        )

    return langchain_fast_llm

def get_pinecone_index():
    global pinecone_client
    global pinecone_index

    if pinecone_index is None:
        print("[PINECONE] Initializing Pinecone Index client...")
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not configured.")

        if pinecone_client is None:
            pinecone_client = Pinecone(api_key=api_key)

        host = os.getenv("PINECONE_INDEX_HOST")
        if host:
            print(f"[PINECONE] Initializing Index client using direct host: {host}")
            pinecone_index = pinecone_client.Index(
                PINECONE_INDEX_NAME,
                host=host
            )
        else:
            print("[PINECONE] PINECONE_INDEX_HOST not set. Index host will be resolved via control plane API call.")
            pinecone_index = pinecone_client.Index(
                PINECONE_INDEX_NAME
            )
    else:
        print("[PINECONE] Reusing existing Pinecone Index client.")

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
        ) if chat_history else "No previous conversation history."

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
    is_seat_query = False

    # -----------------------------------------------------------------------
    # Cache hit: serve identical question instantly without Pinecone / LLM
    # -----------------------------------------------------------------------
    cached = _cache_get(user_query, len(chat_history))
    if cached:
        print(f"[CACHE HIT] Serving cached response for: {user_query[:60]}")
        yield cached
        return
    # -----------------------------------------------------------------------
    
    t0 = time.time()
    expanded_user_query = expand_query_abbreviations(user_query)
    if chat_history and needs_query_rewrite(expanded_user_query):
        standalone_query = await condense_query(chat_history, expanded_user_query)
        print(f"[LATENCY] Query Rewrite: {time.time() - t0:.4f}s")
    else:
        standalone_query = expanded_user_query
        print("[LATENCY] Query Rewrite: Skipped")
    
    t1 = time.time()
    try:
        query_vector = cached_embedding(standalone_query)
        print(f"[LATENCY] Embedding Generation: {time.time() - t1:.4f}s")
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        query_vector = None
        
    t2 = time.time()
    intent = intent_detector.classify(standalone_query, query_vector=query_vector)
    print(f"[LATENCY] Intent Classification: {time.time() - t2:.4f}s")
    
    if intent == "SEAT":
        is_seat_query = True
        import re
        query_lower = standalone_query.lower()
        pg_patterns = [
            r"\bpostgrad\b", r"\bpostgraduate\b", r"\bmasters?\b", r"\bphd\b", r"\bph\.d\b",
            r"\bms\b", r"\bm\.e\.\b", r"\bm\.s\.\b", r"\bm\.c\.s\.\b", r"\bdoctorate\b",
            r"\bm\.engg\b", r"\bmem\b", r"\bm\.arch\b", r"\bmurp\b"
        ]
        ug_patterns = [
            r"\bundergrad\b", r"\bundergraduate\b", r"\bbs\b", r"\bb\.e\.\b", 
            r"\bdae\b", r"\bintermediate\b", r"\bhsc\b", r"\bmatric\b"
        ]
        is_pg = any(re.search(pat, query_lower) for pat in pg_patterns)
        is_ug = any(re.search(pat, query_lower) for pat in ug_patterns)
        
        if is_pg and not is_ug:
            yield "There is no fixed seat distribution matrix published in the Postgraduate Prospectus. Admissions to postgraduate Master's and Ph.D. programmes at NED University are determined based on departmental capacity, eligibility criteria, and academic/faculty resources rather than a predefined category-wise seat matrix. For details on specific programmes, please consult the respective department sections in the Postgraduate Prospectus.\n"
            return
        elif is_ug and not is_pg:
            yield f"For the complete and accurate Undergraduate Seat Distribution Matrix, please refer to the official document: [Undergraduate Seat Distribution PDF]({SEAT_DIST_FILE_LINK}).\n\n"
            return
        else:
            # Ambiguous or both: show undergraduate PDF first, then explain the postgraduate policy
            yield f"### Undergraduate\nFor the complete and accurate Undergraduate Seat Distribution Matrix, please refer to the official document: [Undergraduate Seat Distribution PDF]({SEAT_DIST_FILE_LINK}).\n\n"
            yield "### Postgraduate / Masters\nThere is no fixed seat distribution matrix published in the Postgraduate Prospectus. Admissions to postgraduate Master's and Ph.D. programmes at NED University are determined based on departmental capacity, eligibility criteria, and academic/faculty resources rather than a predefined category-wise seat matrix. For details on specific programmes, please consult the respective department sections in the Postgraduate Prospectus.\n"
            return
        
    if intent == "GENERAL":
        system_base = "You are Prospectus AI, the official academic assistant for NED University. Answer the user's query directly and helpfully. Always identify yourself as Prospectus AI."
    else: # RAG
        t3 = time.time()
        try:
            import re
            query_lower = standalone_query.lower()
            pg_patterns = [
                r"\bpostgrad", r"\bpostgraduate\b", r"\bmasters?\b", r"\bphd\b", r"\bph\.d\b",
                r"\bms\b", r"\bm\.e\.\b", r"\bm\.s\.\b", r"\bm\.c\.s\.\b", r"\bdoctorate\b",
                r"\bm\.engg\b", r"\bmem\b", r"\bm\.arch\b", r"\bmurp\b"
            ]
            ug_patterns = [
                r"\bundergrad", r"\bbs\b", r"\bb\.e\.\b", r"\bdae\b", r"\bintermediate\b", 
                r"\bhsc\b", r"\bmatric\b"
            ]
            
            is_pg = any(re.search(pat, query_lower) for pat in pg_patterns)
            is_ug = any(re.search(pat, query_lower) for pat in ug_patterns)
            
            if not is_pg and not is_ug and chat_history:
                for msg in reversed(chat_history[-4:]):
                    msg_content = msg.get("content", "").lower()
                    if any(re.search(pat, msg_content) for pat in pg_patterns):
                        is_pg = True
                        break
                    if any(re.search(pat, msg_content) for pat in ug_patterns):
                        is_ug = True
                        break
            
            filter_dict = {}
            if is_pg and not is_ug:
                filter_dict = {"academic_level": "postgraduate"}
                print("[DEBUG] Filtering Pinecone by: postgraduate")
            elif is_ug and not is_pg:
                filter_dict = {"academic_level": "undergraduate"}
                print("[DEBUG] Filtering Pinecone by: undergraduate")
            else:
                print("[DEBUG] Query level is ambiguous. Querying both levels.")

            print(f"[DEBUG] query_vector is None: {query_vector is None}")
            if query_vector is None:
                t_embed = time.time()
                query_vector = cached_embedding(standalone_query)
                print(f"[DEBUG] Late-binding embedding generation: {time.time() - t_embed:.4f}s")
            
            t_query = time.time()
            index = get_pinecone_index()
            response = await asyncio.to_thread(
                index.query,
                vector=query_vector,
                top_k=3,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            matches = response.matches
            print(f"[DEBUG] Raw gRPC query execution: {time.time() - t_query:.4f}s")
            
            context_chunks = []
            for m in matches:
                metadata = m.metadata
                if metadata and "text" in metadata:
                    page = metadata.get("source_page", "Unknown")
                    try:
                        if isinstance(page, (int, float)):
                            page = int(page)
                    except Exception:
                        pass
                    context_chunks.append(f"[Page {page}]\n{metadata['text']}")
            context = "\n---\n\n".join(context_chunks)
            print(f"[LATENCY] Pinecone Retrieval: {time.time() - t3:.4f}s")
        except Exception as e:
            print(f"Pinecone retrieval failed: {e}")
            context = "No context available."

        system_base = f"""You are Prospectus AI, the official academic assistant for NED University.
 
Your task is to answer the user's question accurately, politely, and clearly using ONLY the retrieved prospectus context.
 
Retrieved Context:
{context}
 
Instructions:
 
1. **Tone & Style**: Maintain a warm, welcoming, professional, and helpful tone. Speak directly as the voice of the university. Answer the user's question directly and naturally. Do NOT begin with robotic introductory phrases like "Based on the retrieved context...", "The prospectus says...", or "According to page 25...". Instead, write directly: "The Chairperson of the Department of Electronic Engineering is Prof. Dr. Sadia Muniza Faraz (Page 25)."
 
2. **Citations**: Every factual statement, name, seat count, or eligibility rule must be immediately cited with its page number (e.g. "(Page 20)" or "on Page 20") directly at the end of the sentence or clause.
 
3. **Handling Missing Details / Out-of-Scope Queries**:
   - If the context does not contain enough details, warmly guide the user to the right contact or website: *"For the most up-to-date details on this topic, I recommend visiting the official NED University website at neduet.edu.pk or contacting the department directly — the team there will be happy to assist you!"*
   - Avoid negative, robotic, or dry phrases like "I couldn't find...", "No information available", "Not mentioned", or similar denials.
   - If the query is completely unrelated to NED University, admissions, or academics, politely say: *"I am your NED University Academic Assistant. I can help you with admissions, seat distribution, department chairpersons, and general prospectus details. Please let me know how I can assist you with university inquiries!"* and do NOT answer the unrelated question. Under no circumstances should you generate code, write general programming solutions, or provide answers to non-university topics. Stop immediately after the polite redirection.
 
4. **Accuracy**: Copy all numerical values, percentages, and names exactly. Never make assumptions, estimate, or invent facts.
 
5. **Formatting**: Use clean markdown structure, bold text for headers/names, and bullet points for lists to make answers highly readable and visually professional. If the context has a table, preserve it as a markdown table.
 
6. **Academic Levels**: If context contains details for both undergraduate and postgraduate levels, you MUST separate your answer into distinct sections with clear headings (e.g. "### Undergraduate" and "### Postgraduate"). Never blend or mix details from different levels into a single list or sentence.
 
7. **Conversation History**: Use conversation history to resolve references such as "it", "that programme", or "its eligibility".
 
8. **Ambiguity**: If the question is ambiguous, ask a brief clarification politely instead of guessing.
 
9. **Evergreen & Professional**: Keep responses professional, concise, and evergreen. Speak about programs and rules directly rather than referencing the source document. However, you must still explicitly append the page citations (e.g. "(Page 20)") as shown in Instruction 1.
 
10. **General University Facts**: The current Vice-Chancellor of NED University of Engineering and Technology is Prof. Dr. Muhammad Tufail. The main campus is located on University Road, Karachi - 75270. The Registrar Main Campus contact details are registrar@neduet.edu.pk, Tel: +92-21-99261261-8, Fax: +92-21-99261255."""
        
        if is_seat_query:
            system_base += "\n\n11. Focus exclusively on seat distribution and the number of seats. Do NOT mention, list, or summarize any university fees, tuition fees, admission fees, security deposits, or document verification charges, even if they appear in the retrieved context."

    # Limit active history to the last 10 messages (5 turns) to stay well within Groq TPM limits
    recent_history = chat_history[-10:]
    
    messages = [SystemMessage(content=system_base)]
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=standalone_query))

    # 5. Stream the response from ChatGroq LLM (with rate-limit retry + response caching)
    MAX_RETRIES = 3
    full_response_parts = []
    for attempt in range(MAX_RETRIES):
        try:
            full_response_parts = []
            async for chunk in get_llm().astream(messages):
                content = chunk.content
                if content:
                    full_response_parts.append(content)
                    yield content
            # Cache the full assembled answer for future identical queries
            if full_response_parts:
                _cache_set(user_query, len(chat_history), "".join(full_response_parts))
            break  # Success — exit retry loop
        except Exception as e:
            err_msg = str(e)
            err_lower = err_msg.lower()
            # --- Rate-limit: parse suggested wait time and back off ---
            if "429" in err_msg or "rate limit" in err_lower:
                import re as _re
                wait_match = _re.search(r"try again in ([\d.]+)s", err_lower)
                wait_sec = float(wait_match.group(1)) if wait_match else (2 ** attempt * 5)
                wait_sec = min(wait_sec + 1, 30)  # cap at 30 s
                if attempt < MAX_RETRIES - 1:
                    print(f"[RATE-LIMIT] Groq 429 — waiting {wait_sec:.1f}s before retry {attempt + 2}/{MAX_RETRIES}")
                    # Yield a soft notice so the frontend doesn't time out
                    yield f"\n*(Rate limit reached — retrying in {wait_sec:.0f}s…)*\n"
                    await asyncio.sleep(wait_sec)
                    full_response_parts = []  # reset for retry
                    continue
                else:
                    yield "\n⚠️ **Rate limit exceeded.** Please wait a moment and try again."
            elif "connection" in err_lower or "timeout" in err_lower or "max retries" in err_lower or "unreachable" in err_lower or "resolv" in err_lower or "failed to establish" in err_lower:
                yield "⚠️ **Internet Connection Error:** Could not connect to the AI service. Please check your internet connection and try again."
                break
            else:
                yield f"⚠️ **Chatbot Error:** {e}"
                break

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

    # 3. Starts with common follow-up/continuation phrases
    followup_prefixes = ("what about", "how about", "what of", "how of", "what for", "how for", "and ")
    if query.startswith(followup_prefixes):
        return True

    # 4. Very short queries (1-3 words) that end with a question mark
    if query.endswith("?") and len(tokens) <= 3:
        return True

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

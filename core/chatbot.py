import os
import sys
from functools import lru_cache
from dotenv import load_dotenv
from pinecone import Pinecone
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
from core.embeddings import embed_query, embed_documents
import math
import json
import re

def normalize_token(token: str) -> str:
    t = token.lower().strip()
    synonyms = {
        "chairman": "chairperson",
        "chairperson": "chairperson",
        "chairwoman": "chairperson",
        "chair": "chairperson",
        
        "program": "program",
        "programme": "program",
        
        "lab": "lab",
        "laboratory": "lab",
        
        "dept": "department",
        "department": "department",
        
        "postgrad": "postgraduate",
        "postgraduate": "postgraduate",
        "pg": "postgraduate",
        
        "undergrad": "undergraduate",
        "undergraduate": "undergraduate",
        "ug": "undergraduate"
    }
    return synonyms.get(t, t)

class LocalBM25:
    def __init__(self):
        self.chunks = []
        self.doc_freqs = {}
        self.avg_doc_len = 0
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.loaded = False
        self.academic_level = None
        
    def load_chunks(self, academic_level: str):
        path = os.path.join("output_chunks", f"{'UG' if academic_level == 'undergraduate' else 'PG'}Prospectus_compiled_knowledge.json")
        if not os.path.exists(path):
            return False
            
        with open(path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        doc_lens = []
        self.doc_freqs = {}
        for chunk in self.chunks:
            chunk["academic_level"] = academic_level
            tokens = self.tokenize(chunk["text"])
            chunk["tokens"] = tokens
            chunk["term_freqs"] = {}
            for t in tokens:
                chunk["term_freqs"][t] = chunk["term_freqs"].get(t, 0) + 1
                
            for t in set(tokens):
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
                
            doc_lens.append(len(tokens))
            
        num_docs = len(self.chunks)
        self.avg_doc_len = sum(doc_lens) / num_docs if num_docs > 0 else 0
        
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
        self.loaded = True
        self.academic_level = academic_level
        return True
        
    def tokenize(self, text: str) -> list[str]:
        raw_tokens = re.findall(r"\w+", text.lower())
        return [normalize_token(t) for t in raw_tokens]
        
    def score(self, query: str, top_k: int = 12) -> list:
        if not self.loaded:
            return []
            
        query_tokens = self.tokenize(query)
        scores = []
        
        for chunk in self.chunks:
            score = 0.0
            term_freqs = chunk["term_freqs"]
            doc_len = len(chunk["tokens"])
            
            for token in query_tokens:
                if token in term_freqs:
                    tf = term_freqs[token]
                    idf_val = self.idf.get(token, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf_val * (numerator / denominator)
                    
            if score > 0:
                scores.append((score, chunk))
                
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]

_bm25_instances = {}

def get_bm25_instance(academic_level: str):
    global _bm25_instances
    if academic_level not in _bm25_instances:
        bm25 = LocalBM25()
        success = bm25.load_chunks(academic_level)
        if success:
            _bm25_instances[academic_level] = bm25
        else:
            return None
    return _bm25_instances.get(academic_level)


# ---------------------------------------------------------------------------

load_dotenv()

PINECONE_INDEX_NAME = "rag-chatbot-index"
PINECONE_TOP_K = int(os.getenv("PINECONE_TOP_K", "12"))

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
    
    # Generic version: no hardcoded abbreviations
    abbreviations = {}
    
    import re
    expanded = query
    for abbrev, full_name in abbreviations.items():
        # Match only full words (case-insensitive)
        # Add negative lookahead to prevent expanding course codes (e.g., CS-301, CS 301)
        pattern = r"\b" + re.escape(abbrev) + r"\b(?!\s*-?\s*\d)"
        expanded = re.sub(pattern, full_name, expanded, flags=re.IGNORECASE)
    return expanded


groq_client = None
langchain_llm = None

pinecone_client = None
pinecone_index = None

_global_httpx_client = None

# ---------------------------------------------------------------------------
# Semantic & Exact Cache
# ---------------------------------------------------------------------------
import numpy as np
import hashlib as _hashlib

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_chunks")
_RESPONSE_CACHE_PATH = os.path.join(_CACHE_DIR, "response_cache.json")
_SEMANTIC_CACHE_PATH = os.path.join(_CACHE_DIR, "semantic_cache.json")

_response_cache: dict = {}          # key → cached answer string
_semantic_cache = []                # list of {"query": str, "vector": list, "history_len": int, "answer": str}
_CACHE_MAX = 512                    # max distinct cached entries

def _load_persistent_caches():
    global _response_cache, _semantic_cache
    try:
        if os.path.exists(_RESPONSE_CACHE_PATH):
            with open(_RESPONSE_CACHE_PATH, "r", encoding="utf-8") as f:
                _response_cache = json.load(f)
        if os.path.exists(_SEMANTIC_CACHE_PATH):
            with open(_SEMANTIC_CACHE_PATH, "r", encoding="utf-8") as f:
                _semantic_cache = json.load(f)
    except Exception as e:
        print(f"[CACHE] Error loading persistent cache: {e}")

def _save_response_cache():
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_RESPONSE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_response_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[CACHE] Error saving response cache: {e}")

def _save_semantic_cache():
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_SEMANTIC_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_semantic_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[CACHE] Error saving semantic cache: {e}")

# Load persistent caches immediately
_load_persistent_caches()

def _cache_key(query: str, chat_history: list) -> str:
    """Secure cache key: hash of query and serialized conversation history contents."""
    history_serialized = ""
    if chat_history:
        history_serialized = "|".join(f"{msg['role']}:{msg['content']}" for msg in chat_history)
    raw = f"{query.strip().lower()}|{history_serialized}"
    return _hashlib.md5(raw.encode()).hexdigest()

def _cache_get(query: str, chat_history: list):
    return _response_cache.get(_cache_key(query, chat_history))

def _cache_set(query: str, chat_history: list, answer: str):
    if len(_response_cache) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = next(iter(_response_cache))
        _response_cache.pop(oldest, None)
    _response_cache[_cache_key(query, chat_history)] = answer
    _save_response_cache()

def _cache_get_semantic(query: str, query_vector: list[float], chat_history: list, threshold: float = 0.88) -> str:
    """
    Looks up response using semantic similarity.
    Since embeddings are L2 normalized, cosine similarity is just the dot product.
    """
    if not query_vector or not _semantic_cache:
        return None
        
    q_vec = np.array(query_vector)
    
    # Normalize to be 100% safe
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm
        
    history_serialized = ""
    if chat_history:
        history_serialized = "|".join(f"{msg['role']}:{msg['content']}" for msg in chat_history)
        
    best_score = -1.0
    best_answer = None
    
    for item in _semantic_cache:
        if item["history_serialized"] == history_serialized:
            item_vec = np.array(item["vector"])
            item_norm = np.linalg.norm(item_vec)
            if item_norm > 0:
                item_vec = item_vec / item_norm
                
            score = float(np.dot(q_vec, item_vec))
            if score > best_score:
                best_score = score
                best_answer = item["answer"]
                
    if best_score >= threshold:
        print(f"[SEMANTIC CACHE HIT] Score: {best_score:.4f} for similar query: '{query[:40]}'")
        return best_answer
        
    return None

def _cache_set_semantic(query: str, query_vector: list[float], chat_history: list, answer: str):
    if not query_vector or not answer:
        return
    if len(_semantic_cache) >= _CACHE_MAX:
        _semantic_cache.pop(0)
        
    history_serialized = ""
    if chat_history:
        history_serialized = "|".join(f"{msg['role']}:{msg['content']}" for msg in chat_history)
        
    _semantic_cache.append({
        "query": query.strip().lower(),
        "vector": query_vector,
        "history_serialized": history_serialized,
        "answer": answer
    })
    _save_semantic_cache()
# ---------------------------------------------------------------------------

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

_llm_instances = {}

def get_llm_for_attempt(attempt: int = 0):
    """
    Returns a ChatGroq instance, rotating models on retry to bypass rate limits (429).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured.")
        
    primary_model = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")
    models_str = os.getenv("LLM_FAILOVER_MODELS", "llama-3.1-8b-instant,llama-3.3-70b-versatile,llama3-8b-8192,mixtral-8x7b-32768")
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    
    # Ensure primary_model is the first model in the rotation list
    if primary_model in models:
        models.remove(primary_model)
    models.insert(0, primary_model)
    
    model_name = models[attempt % len(models)]
    
    if model_name not in _llm_instances:
        _llm_instances[model_name] = ChatGroq(
            model=model_name,
            temperature=0.3,
            groq_api_key=api_key,
            streaming=True,
            max_retries=0,
        )
    return _llm_instances[model_name]

def get_llm():
    return get_llm_for_attempt(0)

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
    from core.retrieval.pinecone_retriever import get_pinecone_index as _get
    return _get()

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
You are an expert query rewriter. Your task is to rewrite a follow-up user question into a single, self-contained standalone search query that preserves the conversation's context.

Rules:
- Resolve all references (e.g., "it", "this", "that", "same", "above", "previous").
- If the latest message is a continuation (like a department name, e.g., "electronic engineering" or "what about software"), rewrite it to reflect the original question's intent (e.g., "Who is the chairperson of Electronic Engineering?").
- Do NOT answer the question.
- Do NOT add search syntax, preamble, explanations, or quotes.
- Return ONLY the rewritten query.

Examples:
1.
Conversation:
User: Who is the chairman of software engineering?
Assistant: Prof. Dr. Shehnila Zardari is the chairperson of Software Engineering.
Latest User Message: What about Computer Science?
Rewritten Query: Who is the chairperson of the Department of Computer Science?

2.
Conversation:
User: What is the eligibility criteria for postgraduate masters?
Assistant: The candidate must have a relevant undergraduate degree with at least 50% marks.
Latest User Message: fees?
Rewritten Query: What is the fee structure for postgraduate masters programs?

3.
Conversation:
User: Who is the dean of Civil engineering?
Assistant: Prof. Dr. S.F.A. Rafeeqi.
Latest User Message: electronic
Rewritten Query: Who is the chairperson of the Department of Electronic Engineering?
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
    # Exact Cache hit: serve identical question instantly without Pinecone / LLM
    # -----------------------------------------------------------------------
    cached = _cache_get(user_query, chat_history)
    if cached:
        print(f"[CACHE HIT] Serving exact cached response for: {user_query[:60]}")
        yield cached
        return
    # -----------------------------------------------------------------------
    
    t0 = time.time()
    expanded_user_query = expand_query_abbreviations(user_query)
    
    condense_task = None
    if chat_history and needs_query_rewrite(expanded_user_query):
        condense_task = asyncio.create_task(condense_query(chat_history, expanded_user_query))
        
    t1 = time.time()
    try:
        query_vector_raw = await asyncio.to_thread(cached_embedding, expanded_user_query)
        orig_query_vector = query_vector_raw
    except Exception as e:
        print(f"Embedding failed: {e}")
        query_vector_raw = None
        orig_query_vector = None
        
    t2 = time.time()
    try:
        intent = intent_detector.classify(expanded_user_query, query_vector=query_vector_raw)
    except Exception as e:
        intent = "RAG"
    print(f"[LATENCY] Parallel Embed + Intent: {time.time() - t1:.4f}s")
    
    # Semantic Cache hit: check if there's a semantically similar cached query
    if query_vector_raw:
        semantic_cached = _cache_get_semantic(user_query, query_vector_raw, chat_history)
        if semantic_cached:
            if condense_task:
                condense_task.cancel()
            yield semantic_cached
            return
            
    if intent == "SEAT":
        if condense_task:
            condense_task.cancel()
        is_seat_query = True
        import re
        query_lower = expanded_user_query.lower()
        if chat_history:
            for msg in reversed(chat_history[-4:]):
                query_lower += " " + msg.get("content", "").lower()

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
            yield "There is no fixed seat distribution matrix published in the Postgraduate Prospectus. Admissions to postgraduate Master's and Ph.D. programmes are determined based on departmental capacity, eligibility criteria, and academic/faculty resources rather than a predefined category-wise seat matrix. For details on specific programmes, please consult the respective department sections in the Postgraduate Prospectus.\n"
            return
        elif is_ug and not is_pg:
            yield f"For the complete and accurate Undergraduate Seat Distribution Matrix, please refer to the official document: [Undergraduate Seat Distribution PDF]({SEAT_DIST_FILE_LINK}).\n\n"
            return
        else:
            yield f"### Undergraduate\nFor the complete and accurate Undergraduate Seat Distribution Matrix, please refer to the official document: [Undergraduate Seat Distribution PDF]({SEAT_DIST_FILE_LINK}).\n\n"
            yield "### Postgraduate / Masters\nThere is no fixed seat distribution matrix published in the Postgraduate Prospectus. Admissions to postgraduate Master's and Ph.D. programmes are determined based on departmental capacity, eligibility criteria, and academic/faculty resources rather than a predefined category-wise seat matrix. For details on specific programmes, please consult the respective department sections in the Postgraduate Prospectus.\n"
            return
        
    if intent == "GENERAL":
        if condense_task:
            condense_task.cancel()
        system_base = "You are Prospectus AI, the official academic assistant for the University. Answer the user's query directly and helpfully. Always identify yourself as Prospectus AI. Do NOT include page citations, references, or source details. Do NOT answer specific academic, admission, course, or department queries using external knowledge. If asked about any other university facts, history, programs, or details, politely state that you can only answer from the official records and guide the user to ask their specific question so you can look it up in the prospectus. CRITICAL: Never explicitly mention any specific university name (such as 'NED University', 'NED', 'NEDUET') in your responses. Always use the generic term 'the University' instead."
        standalone_query = expanded_user_query
    else: # RAG
        if condense_task:
            try:
                standalone_query = await condense_task
                print(f"[LATENCY] Query Rewrite finished: {time.time() - t0:.4f}s")
            except Exception as e:
                print(f"Query rewrite failed: {e}")
                standalone_query = expanded_user_query
                
            try:
                query_vector = await asyncio.to_thread(cached_embedding, standalone_query)
            except Exception:
                query_vector = None
        else:
            standalone_query = expanded_user_query
            query_vector = query_vector_raw
        t3 = time.time()
        try:
            import re
            query_lower = standalone_query.lower()
            
            search_query = standalone_query
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
                query_vector = cached_embedding(search_query)
                print(f"[DEBUG] Late-binding embedding generation: {time.time() - t_embed:.4f}s")
            
            # --- HYBRID RETRIEVAL & RERANKING via core.retrieval.retriever ---
            t_query = time.time()
            from core.retrieval.retriever import retrieve
            
            print(f"[DEBUG] Invoking advanced retrieval engine...")
            retrieval_res = await retrieve(
                query=standalone_query,
                query_vector=query_vector,
                academic_level_filter=filter_dict if filter_dict else None,
                is_ug=is_ug,
                is_pg=is_pg
            )
            context = retrieval_res.get("context", "No context available.")
            print(f"[LATENCY] Advanced Retrieval Pipeline: {time.time() - t_query:.4f}s")
        except Exception as e:
            print(f"Pinecone retrieval failed: {e}")
            context = "No context available."

        system_base = f"""You are Prospectus AI, the official academic assistant for the University.
 
Your task is to answer the user's question accurately, politely, and clearly using ONLY the retrieved prospectus context. Do NOT use any pre-trained or external knowledge about the University.
 
Retrieved Context:
{context}
 
Instructions:
 
1. **Tone & Style**: Maintain a warm, welcoming, professional, and helpful tone. Speak directly as the voice of the university. Answer the user's question directly and naturally. Do NOT begin with robotic introductory phrases like "Based on the retrieved context...", "The prospectus says...", or "According to the document...". Instead, write directly: "The Chairperson of the Department of Electronic Engineering is Prof. Dr. Sadia Muniza Faraz."
 
2. **Proper Citations**: You must cite your sources seamlessly. For general text, append `(UG Prospectus, Page X)` at the end of the relevant sentence. For data presented in tables or lists, place the citation clearly in a dedicated "Source" column or at the bottom of the table to avoid repetitive clutter. Never invent page numbers.

3. **Handling Missing Details / Out-of-Scope Queries (Strict Knowledge Limitation)**:
   - You must answer using ONLY the provided Retrieved Context. Do NOT use any pre-trained or external knowledge about the University. Rely strictly on the retrieved context for details like programs, courses, fees, administration, faculty, locations, admission criteria, etc.
   - If the context does not contain enough details, warmly guide the user to the right contact or website: *"For detailed and up-to-date information on this topic, I recommend visiting the official University website or contacting the department directly — the team there will be happy to assist you!"* Do NOT try to answer using your external knowledge or make up facts.
   - Avoid negative, robotic, or dry phrases like "I couldn't find...", "No information available", "Not mentioned", or similar denials.
   - If the query is completely unrelated to the University, admissions, or academics, politely say: *"I am your University Academic Assistant. I can help you with admissions, seat distribution, department chairpersons, and general prospectus details. Please let me know how I can assist you with university inquiries!"* and do NOT answer the unrelated question. Under no circumstances should you generate code, write general programming solutions, or provide answers to non-university topics. Stop immediately after the polite redirection.
 
4. **Accuracy**: Copy all numerical values, percentages, and names exactly. Never make assumptions, estimate, or invent facts.
 
5. **Formatting**: Your answers must be incredibly clean and visually professional.
   - If the retrieved context contains structured data (like course codes, credit hours, seat distributions, or fee structures), **you MUST present it as a Markdown table** (e.g., `| Course Code | Title | Credit Hours | Source |`). 
   - Never output repetitive, dense, or raw lists of courses/fees.
   - Use bold text for key names, headers, and emphasis.
 
6. **Academic Levels**: If context contains details for both undergraduate and postgraduate levels, you MUST separate your answer into distinct sections with clear headings (e.g. "### Undergraduate" and "### Postgraduate"). Never blend or mix details from different levels into a single list or sentence.
 
7. **Conversation History**: Use conversation history to resolve references such as "it", "that programme", or "its eligibility".
 
8. **Ambiguity**: If the question is ambiguous, ask a brief clarification politely instead of guessing.
 
9. **Evergreen & Professional**: Keep responses professional, concise, and evergreen. Speak about programs and rules directly rather than referencing the source document as a paper file, but still include the required parenthetical source citations (e.g., `(UG Prospectus, Page 20)`).

10. **Generic Naming**: NEVER explicitly mention any specific university name (such as "NED University", "NED", "NEDUET", etc.) in your responses. Always use the generic term "the University" instead."""
        
        if is_seat_query:
            system_base += "\n\n11. Focus exclusively on seat distribution and the number of seats. Do NOT mention, list, or summarize any university fees, tuition fees, admission fees, security deposits, or document verification charges, even if they appear in the retrieved context."

        system_base += "\n\n12. **Department Specificity**: If the query asks about a specific department (e.g. 'Electronic Engineering' or 'Computer Science'), you MUST verify that your response corresponds EXACTLY to that department. Do NOT mix or combine names, faculty, or criteria from different departments present in the context."
        system_base += "\n\n13. **Co-Chairpersons**: If the retrieved context lists both a Chairperson and a Co-Chairperson for the department being queried, you must explicitly state both names in your response (e.g., 'The Chairperson is [Name] and the Co-Chairperson is [Name]'), rather than omitting the Co-Chairperson."

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
    MAX_RETRIES = 6
    full_response_parts = []
    for attempt in range(MAX_RETRIES):
        try:
            full_response_parts = []
            async for chunk in get_llm_for_attempt(attempt).astream(messages):
                content = chunk.content
                if content:
                    full_response_parts.append(content)
                    # Force a buttery smooth typewriter effect even if Groq returns massive chunks
                    chunk_size = 2
                    for i in range(0, len(content), chunk_size):
                        yield content[i:i+chunk_size]
                        await asyncio.sleep(0.01)
            # Cache the full assembled answer for future queries
            if full_response_parts:
                answer_str = "".join(full_response_parts)
                _cache_set(user_query, chat_history, answer_str)
                if orig_query_vector:
                    _cache_set_semantic(user_query, orig_query_vector, chat_history, answer_str)
            break  # Success — exit retry loop
        except Exception as e:
            err_msg = str(e)
            err_lower = err_msg.lower()
            if "429" in err_msg or "rate limit" in err_lower or "400" in err_msg or "decommission" in err_lower or "not support" in err_lower or "unsupported" in err_lower:
                if attempt < MAX_RETRIES - 1:
                    print(f"[RATE-LIMIT/FAILOVER] Groq error on attempt {attempt + 1} — switching models and retrying immediately...")
                    full_response_parts = []  # reset for retry
                    continue
                else:
                    yield "\n⚠️ **Service unavailable.** Please wait a moment and try again."
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

    # Single-word questions that require context (e.g. "fees", "eligibility") or department terms
    continuation_keywords = {
        "fee", "fees", "eligibility", "criteria", "requirements", "dean", 
        "syllabus", "courses", "duration", "seats", "seat", "apply", 
        "admission", "admissions", "cutoff", "merit",
        "electronic", "electronics", "computer", "software", "mechanical",
        "electrical", "civil", "petroleum", "telecommunication", "telecommunications",
        "biomedical", "industrial", "manufacturing", "textile", "architecture",
        "automotive", "marine", "materials", "metallurgical", "chemical", "polymer",
        "math", "mathematics", "physics", "chemistry", "english", "linguistics",
        "economics", "finance", "environmental", "earthquake", "coastal", "urban",
        "chairperson", "chairman", "chairwoman", "chair"
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
    followup_prefixes = ("what about", "how about", "what of", "how of", "what for", "how for", "and ", "also ", "then ", "but ")
    if query.startswith(followup_prefixes):
        return True

    # 4. Very short queries (1-3 words) that end with a question mark
    if query.endswith("?") and len(tokens) <= 3:
        return True

    # 5. Short queries (1-3 words) containing continuation/department keywords
    if len(tokens) <= 3 and any(t in continuation_keywords for t in tokens):
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
        async for chunk in route_chat_stream("Who are you?", history):
            print(chunk, end="", flush=True)
            
        print("\n\n--- Test 2: RAG Query ---")
        async for chunk in route_chat_stream("What is the eligibility criteria for software engineering?"):
            print(chunk, end="", flush=True)
            
        print("\n\n--- Test 3: Seat Matrix ---")
        async for chunk in route_chat_stream("Give me the seat distribution matrix please."):
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

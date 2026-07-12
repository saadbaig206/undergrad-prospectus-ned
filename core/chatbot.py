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

def hybrid_search_rrf(semantic_results: list, keyword_results: list, top_k: int = 15) -> list:
    """
    Applies Reciprocal Rank Fusion (RRF) to merge semantic (Pinecone matches) 
    and keyword (BM25 dict chunks) search results.
    """
    rrf_scores = {}
    
    # 1. Parse Pinecone semantic matches
    for rank, match in enumerate(semantic_results):
        page = match.metadata.get("source_page", "Unknown")
        text = match.metadata.get("text", "")
        level = match.metadata.get("academic_level", "unknown")
        key = (page, text[:120])
        
        if key not in rrf_scores:
            rrf_scores[key] = {
                "rank_sem": rank, 
                "rank_key": None, 
                "source_page": page, 
                "text": text,
                "academic_level": level
            }
        else:
            rrf_scores[key]["rank_sem"] = rank
            
    # 2. Parse BM25 keyword matches
    for rank, chunk in enumerate(keyword_results):
        page = chunk.get("source_page", "Unknown")
        text = chunk.get("text", "")
        level = chunk.get("academic_level", "unknown")
        key = (page, text[:120])
        
        if key not in rrf_scores:
            rrf_scores[key] = {
                "rank_sem": None, 
                "rank_key": rank, 
                "source_page": page, 
                "text": text,
                "academic_level": level
            }
        else:
            rrf_scores[key]["rank_key"] = rank
            
    # 3. Compute RRF Scores
    fused_results = []
    for item in rrf_scores.values():
        score = 0.0
        if item["rank_sem"] is not None:
            score += 1.0 / (60.0 + item["rank_sem"])
        if item["rank_key"] is not None:
            score += 1.0 / (60.0 + item["rank_key"])
            
        item["rrf_score"] = score
        fused_results.append((score, item))
        
    fused_results.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in fused_results[:top_k]]

def rerank_chunks(query: str, query_vector: list[float], candidate_chunks: list, top_k: int = 5) -> list:
    """
    Reranks candidate chunks instantly on CPU using RRF rank fusion scores, 
    token overlap (Jaccard similarity), and alphanumeric code matching boosts,
    avoiding slow CPU-bound ONNX model document embeddings.
    """
    if not candidate_chunks:
        return []
        
    q_tokens = set(normalize_token(t) for t in re.findall(r"\w+", query.lower()))
    
    # Alphanumeric uppercase terms in query (like course codes or department names)
    special_terms = re.findall(r"\b[A-Z0-9\-]{2,10}\b", query)
    
    reranked = []
    for chunk in candidate_chunks:
        text = chunk["text"]
        
        # 1. Base score from RRF Rank Fusion
        rrf_score = chunk.get("rrf_score", 0.0)
        
        # 2. Token overlap score (Jaccard similarity)
        c_tokens = set(normalize_token(t) for t in re.findall(r"\w+", text.lower()))
        jaccard_score = 0.0
        if q_tokens and c_tokens:
            intersection = q_tokens.intersection(c_tokens)
            union = q_tokens.union(c_tokens)
            jaccard_score = len(intersection) / len(union) if union else 0.0
            
        # 3. Special keyword boost (exact match on codes/acronyms)
        boost = 1.0
        for term in special_terms:
            if re.search(r"\b" + re.escape(term.lower()) + r"\b", text.lower()):
                boost = 1.3
                break
                
        # 4. Department name boost (if query specifies department name and chunk matches it exactly)
        depts = ["electronic engineering", "computer science", "software engineering", "civil engineering", "mechanical engineering", "electrical engineering", "biomedical engineering", "telecommunications engineering"]
        for dept in depts:
            if dept in query.lower() and dept in text.lower():
                boost *= 1.5
                break
                
        # Combined reranking score
        final_score = (0.7 * rrf_score + 0.3 * jaccard_score) * boost
        reranked.append((final_score, chunk))
        
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in reranked[:top_k]]
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
# Semantic & Exact Cache
# ---------------------------------------------------------------------------
import numpy as np
import hashlib as _hashlib

_response_cache: dict = {}          # key → cached answer string
_semantic_cache = []                # list of {"query": str, "vector": list, "history_len": int, "answer": str}
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

def _cache_get_semantic(query: str, query_vector: list[float], history_len: int, threshold: float = 0.88) -> str:
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
        
    best_score = -1.0
    best_answer = None
    
    for item in _semantic_cache:
        if item["history_len"] == history_len:
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

def _cache_set_semantic(query: str, query_vector: list[float], history_len: int, answer: str):
    if not query_vector or not answer:
        return
    if len(_semantic_cache) >= _CACHE_MAX:
        _semantic_cache.pop(0)
    _semantic_cache.append({
        "query": query.strip().lower(),
        "vector": query_vector,
        "history_len": history_len,
        "answer": answer
    })
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
    # Exact Cache hit: serve identical question instantly without Pinecone / LLM
    # -----------------------------------------------------------------------
    cached = _cache_get(user_query, len(chat_history))
    if cached:
        print(f"[CACHE HIT] Serving exact cached response for: {user_query[:60]}")
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
        orig_query_vector = query_vector
        print(f"[LATENCY] Embedding Generation: {time.time() - t1:.4f}s")
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        query_vector = None
        orig_query_vector = None
        
    # -----------------------------------------------------------------------
    # Semantic Cache hit: check if there's a semantically similar cached query
    # -----------------------------------------------------------------------
    if query_vector:
        semantic_cached = _cache_get_semantic(user_query, query_vector, len(chat_history))
        if semantic_cached:
            yield semantic_cached
            return
    # -----------------------------------------------------------------------
        
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
        system_base = "You are Prospectus AI, the official academic assistant for NED University. Answer the user's query directly and helpfully. Always identify yourself as Prospectus AI. Do NOT include page citations, references, or source details. Do NOT answer specific academic, admission, course, or department queries using external knowledge. For general info, know that: (1) The current Vice-Chancellor is Prof. Dr. Muhammad Tufail, (2) The main campus is on University Road, Karachi - 75270. If asked about any other university facts, history, programs, or details, politely state that you can only answer from the official records and guide the user to ask their specific question so you can look it up in the prospectus."
    else: # RAG
        t3 = time.time()
        try:
            import re
            query_lower = standalone_query.lower()
            
            # Check if this RAG query is asking what NED stands for or about its history
            stands_for_patterns = [
                r"\bned\b.*\bstands?\b",
                r"\bstands?\b.*\bned\b",
                r"\bfull\b.*\bform\b.*\bned\b",
                r"\bned\b.*\bfull\b.*\bform\b",
                r"\bwhat\b.*\bned\b.*\bmean\b",
                r"\bwhat\b.*\bdoes\b.*\bned\b.*\bstand\b",
                r"\bmeaning\b.*\bned\b",
                r"\bname\b.*\bned\b.*\bstands?\b",
                r"\borigin\b.*\bned\b.*\bname\b",
                r"\bwhy\b.*\bcalled\b.*\bned\b",
                r"\bwho\b.*\bned\b.*\bnamed\b",
                r"\bhistory\b",
                r"\bhistorical\b",
                r"\borigin\b",
                r"\bestablished\b",
                r"\bfounded\b",
                r"\bbackground\b",
                r"\babbreviation\b",
                r"\bstands?\b"
            ]
            if any(re.search(pat, query_lower) for pat in stands_for_patterns):
                search_query = "historical background of NED University"
                print(f"[DEBUG] Query asks what NED stands for / history. Rewriting search query to: {search_query}")
                query_vector = cached_embedding(search_query)
            else:
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
            
            # --- HYBRID RETRIEVAL & RERANKING ---
            t_query = time.time()
            
            # 1. Semantic query to Pinecone (retrieve top PINECONE_TOP_K matches)
            index = get_pinecone_index()
            response = await asyncio.to_thread(
                index.query,
                vector=query_vector,
                top_k=PINECONE_TOP_K,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            semantic_matches = response.matches
            print(f"[DEBUG] Raw gRPC Pinecone query execution: {time.time() - t_query:.4f}s")
            
            # 2. Keyword query to Local BM25 (retrieve top PINECONE_TOP_K matches)
            keyword_matches = []
            if is_pg or (not is_pg and not is_ug):
                bm25_pg = get_bm25_instance("postgraduate")
                if bm25_pg:
                    keyword_matches.extend(bm25_pg.score(standalone_query, top_k=PINECONE_TOP_K))
            if is_ug or (not is_pg and not is_ug):
                bm25_ug = get_bm25_instance("undergraduate")
                if bm25_ug:
                    keyword_matches.extend(bm25_ug.score(standalone_query, top_k=PINECONE_TOP_K))
                    
            # 3. Fuse lists using Reciprocal Rank Fusion (RRF) (candidate list of 15 items)
            fused_candidates = hybrid_search_rrf(semantic_matches, keyword_matches, top_k=15)
            print(f"[DEBUG] Hybrid Search fused {len(fused_candidates)} candidate chunks.")
            
            # 4. Apply Hybrid Scorer Reranking (rerank down to top 5 chunks)
            top_reranked_chunks = rerank_chunks(standalone_query, query_vector, fused_candidates, top_k=5)
            print(f"[DEBUG] Cross-Encoder Reranking completed in {time.time() - t_query:.4f}s.")
            
            # 5. Build final context block
            context_chunks = []
            for chunk in top_reranked_chunks:
                page = chunk.get("source_page", "Unknown")
                try:
                    if isinstance(page, (int, float)):
                        page = int(page)
                except Exception:
                    pass
                level = chunk.get("academic_level", "unknown")
                doc_name = "UG Prospectus" if level == "undergraduate" else "PG Prospectus"
                context_chunks.append(f"[Source: {doc_name}, Page {page}]\n{chunk['text']}")
                
            context = "\n---\n\n".join(context_chunks)
            print(f"[LATENCY] Pinecone Retrieval, Hybrid Search & Reranking: {time.time() - t3:.4f}s")
        except Exception as e:
            print(f"Pinecone retrieval failed: {e}")
            context = "No context available."

        system_base = f"""You are Prospectus AI, the official academic assistant for NED University.
 
Your task is to answer the user's question accurately, politely, and clearly using ONLY the retrieved prospectus context. Do NOT use any pre-trained or external knowledge about NED University.
 
Retrieved Context:
{context}
 
Instructions:
 
1. **Tone & Style**: Maintain a warm, welcoming, professional, and helpful tone. Speak directly as the voice of the university. Answer the user's question directly and naturally. Do NOT begin with robotic introductory phrases like "Based on the retrieved context...", "The prospectus says...", or "According to the document...". Instead, write directly: "The Chairperson of the Department of Electronic Engineering is Prof. Dr. Sadia Muniza Faraz."
 
2. **Proper Citations**: Every statement of fact, number, criterion, name, or seat count derived from the context must be immediately followed by a proper citation in parentheses specifying the source document and page number (e.g., `(UG Prospectus, Page 20)` or `(PG Prospectus, Page 15)`). Place the citation directly at the end of the sentence or clause before punctuation. Never invent or make up page numbers; use exactly the source and page provided in the chunk headers. Keep citations clean, standardized, and professional.
 
3. **Handling Missing Details / Out-of-Scope Queries (Strict Knowledge Limitation)**:
   - You must answer using ONLY the provided Retrieved Context and the specific General University Facts listed in Instruction 10. Do NOT use any pre-trained or external knowledge about NED University, its history, founder, name origin, programs, courses, fees, administration, faculty, locations, admission criteria, or any other university-related detail.
   - If the context does not contain enough details (and the answer is not in General University Facts), warmly guide the user to the right contact or website: *"For detailed and up-to-date information on this topic, I recommend visiting the official NED University website at neduet.edu.pk or contacting the department directly — the team there will be happy to assist you!"* Do NOT try to answer using your external knowledge or make up facts.
   - Avoid negative, robotic, or dry phrases like "I couldn't find...", "No information available", "Not mentioned", or similar denials.
   - If the query is completely unrelated to NED University, admissions, or academics, politely say: *"I am your NED University Academic Assistant. I can help you with admissions, seat distribution, department chairpersons, and general prospectus details. Please let me know how I can assist you with university inquiries!"* and do NOT answer the unrelated question. Under no circumstances should you generate code, write general programming solutions, or provide answers to non-university topics. Stop immediately after the polite redirection.
 
4. **Accuracy**: Copy all numerical values, percentages, and names exactly. Never make assumptions, estimate, or invent facts.
 
5. **Formatting**: Use clean markdown structure, bold text for headers/names, and bullet points for lists to make answers highly readable and visually professional. If the context has a table, preserve it as a markdown table.
 
6. **Academic Levels**: If context contains details for both undergraduate and postgraduate levels, you MUST separate your answer into distinct sections with clear headings (e.g. "### Undergraduate" and "### Postgraduate"). Never blend or mix details from different levels into a single list or sentence.
 
7. **Conversation History**: Use conversation history to resolve references such as "it", "that programme", or "its eligibility".
 
8. **Ambiguity**: If the question is ambiguous, ask a brief clarification politely instead of guessing.
 
9. **Evergreen & Professional**: Keep responses professional, concise, and evergreen. Speak about programs and rules directly rather than referencing the source document as a paper file, but still include the required parenthetical source citations (e.g., `(UG Prospectus, Page 20)`).
 
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
    MAX_RETRIES = 6
    full_response_parts = []
    for attempt in range(MAX_RETRIES):
        try:
            full_response_parts = []
            async for chunk in get_llm_for_attempt(attempt).astream(messages):
                content = chunk.content
                if content:
                    full_response_parts.append(content)
                    yield content
            # Cache the full assembled answer for future queries
            if full_response_parts:
                answer_str = "".join(full_response_parts)
                _cache_set(user_query, len(chat_history), answer_str)
                if orig_query_vector:
                    _cache_set_semantic(user_query, orig_query_vector, len(chat_history), answer_str)
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

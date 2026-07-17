import threading
import hashlib as _hashlib
import numpy as np

_CACHE_MAX = 512                    
_semantic_cache = []  
_semantic_cache_lock = threading.Lock()

def _cache_get_semantic(query: str, query_vector: list[float], chat_history: list, threshold: float = 0.82) -> str:
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

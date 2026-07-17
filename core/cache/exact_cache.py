import hashlib as _hashlib
import threading

_CACHE_MAX = 512 
_response_cache: dict = {} 
_cache_lock = threading.Lock()  

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
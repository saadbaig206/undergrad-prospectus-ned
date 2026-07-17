import asyncio
from core.llm.llm_manager import get_llm_for_attempt
from core.cache.exact_cache import _cache_set
from core.cache.semantic_cache import _cache_set_semantic

async def stream_response(messages, user_query, chat_history, orig_query_vector):
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
                _cache_set(user_query, chat_history, answer_str)
                if orig_query_vector:
                    _cache_set_semantic(user_query, orig_query_vector, chat_history, answer_str)
            break  # Success — exit retry loop
        except Exception as e:
            err_msg = str(e)
            err_lower = err_msg.lower()
            if "429" in err_msg or "rate limit" in err_lower or "400" in err_msg or "decommission" in err_lower or "not support" in err_lower or "unsupported" in err_lower:
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s — gives rate-limit quota time to recover
                    backoff = min(2 ** attempt, 16)
                    print(f"[RATE-LIMIT/FAILOVER] Groq error on attempt {attempt + 1} — switching models, retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
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

from langchain_groq import ChatGroq
import os

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
            max_tokens=400,
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
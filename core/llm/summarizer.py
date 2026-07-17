from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_manager import get_fast_llm

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
    

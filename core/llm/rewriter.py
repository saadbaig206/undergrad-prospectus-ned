import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm.llm_manager import get_fast_llm
from core.llm.summarizer import clean_history_for_llm

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

# In rewriter.py, update condense_query:

# In rewriter.py

async def condense_query(chat_history: list, user_query: str) -> str:
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

        # Strip symbols, formatting quotes, and prefix structures
        cleaned_rewritten = rewritten.strip().replace('"', '').replace("'", "")
        cleaned_rewritten = re.sub(r"^rewritten query:\s*", "", cleaned_rewritten, flags=re.I)
        
        return cleaned_rewritten if cleaned_rewritten else user_query

    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return user_query
        
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

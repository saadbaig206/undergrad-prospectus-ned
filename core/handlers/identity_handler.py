import re

_identity_patterns = [
    r"\bwho\b.*\bare\s+you\b",
    r"\bwhat\s+are\s+you\b",
    r"\bwhat\s+is\s+your\s+name\b",
    r"\bintroduce\s+yourself\b",
    r"\btell\s+me\s+about\s+yourself\b",
]


async def handle_identity_query(query):
    if any(re.search(pat, query.lower()) for pat in _identity_patterns):
        yield (
            "I am **Prospectus AI**, the official academic assistant of **the University**. "
            "I can help you with information about undergraduate and postgraduate programmes, admission eligibility, "
            "department details, chairpersons, research facilities, scholarships, and much more — "
            "all sourced directly from the official prospectuses. Feel free to ask me anything!"
        )
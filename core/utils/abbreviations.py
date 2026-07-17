def normalize_token(token: str) -> str:
    """Normalize common synonyms to canonical forms."""
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
        "ug": "undergraduate",
    }
    return synonyms.get(t, t)

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


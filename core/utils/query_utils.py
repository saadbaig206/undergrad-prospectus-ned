import re

pg_patterns = [
    r"\bpostgrad\b", r"\bpostgraduate\b", r"\bmasters?\b", r"\bphd\b", r"\bph\.d\b",
    r"\bms\b", r"\bm\.e\.\b", r"\bm\.s\.\b", r"\bm\.c\.s\.\b", r"\bdoctorate\b",
    r"\bm\.engg\b", r"\bmem\b", r"\bm\.arch\b", r"\bmurp\b"
]
ug_patterns = [
    r"\bundergrad\b", r"\bundergraduate\b", r"\bbs\b", r"\bb\.e\.\b", 
    r"\bdae\b", r"\bintermediate\b", r"\bhsc\b", r"\bmatric\b"
]

def detect_academic_level(query: str):
    query = query.lower()

    is_pg = any(re.search(p, query) for p in pg_patterns)
    is_ug = any(re.search(p, query) for p in ug_patterns)

    return is_pg, is_ug
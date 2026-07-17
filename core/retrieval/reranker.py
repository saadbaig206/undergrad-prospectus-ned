import re
from core.utils.abbreviations import normalize_token
from core.retrieval.query_metadata import build_metadata

# In reranker.py, rewrite the loop fields extraction
# In reranker.py, prepend this helper and use it inside the loop:

def get_field(chunk: dict, key: str, default=""):
    """Safely handles nested Pinecone payloads and flat BM25 dictionaries."""
    val = chunk.get(key)
    if val is None and "metadata" in chunk:
        val = chunk.get("metadata", {}).get(key)
    return val or default

def rerank_chunks(
    query: str,
    query_vector: list[float],
    candidate_chunks: list,
    top_k: int = 5,
    max_candidates: int = 12,
) -> list:
    if not candidate_chunks:
        return []

    candidate_chunks = candidate_chunks[:max_candidates]
    metadata = build_metadata(query)

    q_tokens = {normalize_token(t) for t in re.findall(r"\w+", query.lower())}
    # Match uppercase terms/acronyms OR course codes like CS-301, cs301, ee 101 case-insensitively
    special_terms = re.findall(r"\b[A-Z0-9\-]{2,15}\b", query) + re.findall(r"\b[a-zA-Z]{2,5}\s*-?\s*\d{2,4}\b", query)

    # Pre-compute token sets to avoid performance bottlenecks
    chunk_token_sets = []
    for chunk in candidate_chunks:
        if "tokens" in chunk:
            tokens_set = {normalize_token(t) for t in chunk["tokens"]}
        elif "metadata" in chunk and "tokens" in chunk["metadata"]:
            tokens_set = {normalize_token(t) for t in chunk["metadata"]["tokens"]}
        else:
            text = chunk.get("text", chunk.get("metadata", {}).get("text", ""))
            tokens_set = {normalize_token(t) for t in re.findall(r"\w+", text.lower())}
        chunk_token_sets.append(tokens_set)

    reranked = []

    for i, chunk in enumerate(candidate_chunks):
        text = chunk.get("text", chunk.get("metadata", {}).get("text", ""))
        rrf_score = chunk.get("rrf_score", 0.0)
        c_tokens = chunk_token_sets[i]

        # Jaccard calculation
        if q_tokens and c_tokens:
            intersection = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            jaccard = intersection / union if union else 0.0
        else:
            jaccard = 0.0

        score = (0.55 * rrf_score) + (0.25 * jaccard)

        # Exact keyword boosts
        text_lower = text.lower()
        for term in special_terms:
            if term.lower() in text_lower:
                score += 0.10
                break

        # Safe Metadata Boosts (Reduced magnitude to prevent overriding text relevance)
        chunk_dept = get_field(chunk, "department")
        if metadata.department and chunk_dept:
            if metadata.department.lower() in chunk_dept.lower() or chunk_dept.lower() in metadata.department.lower():
                score += 0.08

        chunk_program = get_field(chunk, "program")
        if hasattr(metadata, "program") and metadata.program and chunk_program:
            if metadata.program.lower() in chunk_program.lower() or chunk_program.lower() in metadata.program.lower():
                score += 0.08

        chunk_section = get_field(chunk, "section")
        if metadata.section and chunk_section:
            if metadata.section.lower() in chunk_section.lower() or chunk_section.lower() in metadata.section.lower():
                score += 0.10

        heading = get_field(chunk, "heading_path") or get_field(chunk, "heading")
        if heading:
            if metadata.department and metadata.department.lower() in heading.lower():
                score += 0.05
            if metadata.section and metadata.section.lower() in heading.lower():
                score += 0.05
            if metadata.entity and metadata.entity.lower() in heading.lower():
                score += 0.05
                
        if hasattr(metadata, "year_level") and metadata.year_level:
            year_syns = {
                "first": ["first", "1st"], "1st": ["first", "1st"],
                "second": ["second", "2nd"], "2nd": ["second", "2nd"],
                "third": ["third", "3rd"], "3rd": ["third", "3rd"],
                "fourth": ["fourth", "4th", "final"], "4th": ["fourth", "4th", "final"],
                "final": ["fourth", "4th", "final"]
            }.get(metadata.year_level, [metadata.year_level])
            
            for syn in year_syns:
                if heading and syn in heading.lower():
                    score += 0.10
                    break

        if hasattr(metadata, "entity") and metadata.entity:
            if metadata.entity.lower() in text.lower():
                score += 0.10

        reranked.append((score, chunk))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in reranked[:top_k]]
from core.config import MAX_CONTEXT_CHARS, MAX_PER_SECTION
from typing import List, Dict

def build_context(top_reranked_chunks: List[Dict]) -> str:

    # ---------------------------------------
    # Remove duplicate chunks
    # ---------------------------------------
    seen = set()
    unique_chunks = []

    for chunk in top_reranked_chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        unique_chunks.append(chunk)

    # ---------------------------------------
    # Section diversity
    # ---------------------------------------
    section_counts = {}
    filtered_chunks = []

    for chunk in unique_chunks:
        section = (
            chunk.get("section")
            or chunk.get("metadata", {}).get("section")
            or "unknown"
        )

        count = section_counts.get(section, 0)

        if count >= MAX_PER_SECTION:
            continue

        section_counts[section] = count + 1
        filtered_chunks.append(chunk)

    # ---------------------------------------
    # Build final context
    # ---------------------------------------
    context_chunks = []
    current_chars = 0

    for chunk in filtered_chunks:

        text = chunk.get("text", "").strip()

        if not text:
            continue

        # Dynamic context limit
        if current_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        current_chars += len(text)

        page = chunk.get("source_page", "Unknown")

        try:
            if isinstance(page, (int, float)):
                page = int(page)
        except Exception:
            pass

        level = chunk.get("academic_level", "unknown")

        doc_name = (
            "UG Prospectus"
            if level == "undergraduate"
            else "PG Prospectus"
        )

        context_chunks.append(
            f"[Source: {doc_name}, Page {page}]\n{text}"
        )

    return "\n---\n\n".join(context_chunks)
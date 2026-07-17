from collections import defaultdict
from typing import Dict, List

from core.config import MAX_CONTEXT_CHARS, MAX_PER_SECTION


def build_context(chunks: List[Dict]) -> str:

    if not chunks:
        return ""

    # ---------------------------------------
    # Remove duplicates
    # ---------------------------------------

    seen = set()
    unique = []

    for chunk in chunks:

        text = chunk.get("text", "").strip()

        if not text:
            continue

        key = (
            chunk.get("source_page"),
            text,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(chunk)

    # ---------------------------------------
    # Group by section
    # ---------------------------------------

    grouped = defaultdict(list)

    for chunk in unique:

        section = (
            chunk.get("section")
            or chunk.get("metadata", {}).get("section")
            or "general"
        )

        grouped[section].append(chunk)

    # ---------------------------------------
    # Order of importance
    # ---------------------------------------

    preferred_order = [

        "faculty",

        "fees",

        "curriculum",

        "eligibility",

        "admission",

        "laboratory",

        "scholarship",

        "research",

        "general",

    ]

    ordered_sections = []

    for sec in preferred_order:

        if sec in grouped:
            ordered_sections.append(sec)

    for sec in grouped:

        if sec not in ordered_sections:
            ordered_sections.append(sec)

    # ---------------------------------------
    # Build context
    # ---------------------------------------

    context = []

    total_chars = 0

    for section in ordered_sections:

        section_chunks = grouped[section]

        section_chunks.sort(
            key=lambda c: (
                c.get("source_page", 9999),
                c.get("chunk_index", 0),
            )
        )

        count = 0

        for chunk in section_chunks:

            if count >= MAX_PER_SECTION:
                break

            text = chunk.get("text", "").strip()

            page = (
                chunk.get("source_page")
                or chunk.get("page")
                or "Unknown"
            )

            level = chunk.get(
                "academic_level",
                "unknown",
            )

            heading = (
                chunk.get("heading_path")
                or chunk.get("heading")
                or ""
            )

            doc = (
                "UG Prospectus"
                if level == "undergraduate"
                else "PG Prospectus"
            )

            block = f"""[Source: {doc}, Page {page}]
                        Section:{section}
                        {heading}
                        {text}
                        """

            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                return "\n---\n".join(context)

            context.append(block)

            total_chars += len(block)

            count += 1

    return "\n---\n".join(context)
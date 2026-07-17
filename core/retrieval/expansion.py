from collections import defaultdict


def get_metadata_field(chunk, key, default=None):
    val = chunk.get(key)
    if val is None and isinstance(chunk, dict) and "metadata" in chunk:
        val = chunk.get("metadata", {}).get(key)
    elif val is None and hasattr(chunk, "metadata"):
        val = chunk.metadata.get(key)
    return val if val is not None else default

def expand_section(top_chunks, candidates):
    expanded = list(top_chunks)
    seen = {get_metadata_field(c, "text", "") for c in top_chunks if get_metadata_field(c, "text", "")}
    
    groups = defaultdict(list)
    for chunk in candidates:
        dept = get_metadata_field(chunk, "department")
        sect = get_metadata_field(chunk, "section")
        if dept or sect:
            key = (dept, sect)
            groups[key].append(chunk)

    for chunk in top_chunks:
        dept = get_metadata_field(chunk, "department")
        sect = get_metadata_field(chunk, "section")
        if not dept and not sect:
            continue
        key = (dept, sect)
        for c in groups.get(key, []):
            text = get_metadata_field(c, "text", "")
            if not text or text in seen:
                continue
            seen.add(text)
            expanded.append(c)

    return expanded

def expand_page(top_chunks, candidates):
    expanded = list(top_chunks)
    seen = {get_metadata_field(c, "text", "") for c in top_chunks if get_metadata_field(c, "text", "")}

    pages = {
        get_metadata_field(c, "page") or get_metadata_field(c, "source_page")
        for c in top_chunks
    }
    pages = {p for p in pages if p is not None}

    for chunk in candidates:
        chunk_page = get_metadata_field(chunk, "page") or get_metadata_field(chunk, "source_page")
        if not chunk_page or chunk_page not in pages:
            continue

        text = get_metadata_field(chunk, "text", "")
        if not text or text in seen:
            continue

        seen.add(text)
        expanded.append(chunk)

    return expanded

def expand_table(top_chunks, candidates):
    expanded = list(top_chunks)
    seen = {get_metadata_field(c, "text", "") for c in top_chunks if get_metadata_field(c, "text", "")}

    headings = {
        get_metadata_field(c, "heading_path")
        for c in top_chunks
    }
    headings = {h for h in headings if h is not None}

    for chunk in candidates:
        chunk_heading = get_metadata_field(chunk, "heading_path")
        if not chunk_heading or chunk_heading not in headings:
            continue

        text = get_metadata_field(chunk, "text", "")
        if not text or text in seen:
            continue

        seen.add(text)
        expanded.append(chunk)

    return expanded
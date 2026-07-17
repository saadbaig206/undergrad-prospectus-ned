from copy import deepcopy
from typing import List
import re
from langchain_core.documents import Document
from langchain_text_splitters import (MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter,)

# 1. Identify standard Markdown heading structures
HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS,
    strip_headers=False,
)

# 2. Text splitter optimized for standard paragraph prose only
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def _merge_metadata(parent: dict, child: dict) -> dict:
    """
    Stitches parent context and sub-heading levels into a unified 
    breadcrumb path to prevent context loss in isolated chunks.
    """
    meta = deepcopy(parent)
    meta.update({k: v for k, v in child.items() if v not in ("", None)})

    # Build structural breadcrumb paths (e.g., "Software Engineering > Chairperson")
    headings = []
    for key in ("h1", "h2", "h3", "h4"):
        if meta.get(key):
            headings.append(meta[key])
    if headings:
        meta["heading_path"] = " > ".join(headings)

    return meta


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Hierarchical Chunker with Dynamic Block-Shielding for Prospectuses.
    Guarantees that tables are never broken down or fragmented.
    """
    output: List[Document] = []

    for doc in documents:
        raw_content = doc.page_content.strip()
        if not raw_content:
            continue

        # CRITICAL: Detect if this segment contains a table structure
        # (Checks for LlamaParse content classification tags or raw Markdown pipe arrays)
        is_table_block = (
            doc.metadata.get("content_type") in ("curriculum_table", "fee_table", "table")
            or "|" in raw_content
        )

        # SOLUTION 1 & 2: Complete Block-Shielding for Tables
        # Throw the entire table intact into its own chunk so rows & columns stay aligned
        if is_table_block:
            # Inherit and preserve any existing global page/department markers
            table_meta = deepcopy(doc.metadata)
            table_meta["chunk_index"] = 0
            table_meta["chunk_size"] = len(raw_content)
            
            output.append(
                Document(
                    page_content=raw_content,
                    metadata=table_meta
                )
            )
            continue

        # 3. For non-tabular prose text, split gracefully using Markdown headers
        md_sections = _header_splitter.split_text(raw_content)

        if not md_sections:
            md_sections = [Document(page_content=raw_content, metadata={})]

        for section in md_sections:
            merged_meta = _merge_metadata(doc.metadata, section.metadata)
            section_text = section.page_content.strip()

            # Double-check safeguard for nested tables inside heading splits
            if "|" in section_text:
                nested_table_meta = deepcopy(merged_meta)
                nested_table_meta["chunk_index"] = 0
                nested_table_meta["chunk_size"] = len(section_text)
                
                output.append(
                    Document(
                        page_content=section_text,
                        metadata=nested_table_meta
                    )
                )
                continue

            # Break large paragraph prose into manageable sliding token pieces
            pieces = _text_splitter.split_text(section_text)

            for idx, piece in enumerate(pieces):
                chunk_meta = deepcopy(merged_meta)
                chunk_meta["chunk_index"] = idx
                chunk_meta["chunk_size"] = len(piece)

                output.append(
                    Document(
                        page_content=piece.strip(),
                        metadata=chunk_meta
                    )
                )

    return output
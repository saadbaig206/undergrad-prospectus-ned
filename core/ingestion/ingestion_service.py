from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from core.ingestion.parser import (
    split_pdf,
    parse_pdf_to_markdown_async,
    cleanup_temp_file,
)
from core.ingestion.metadata_extractor import extract_documents
from core.ingestion.semantic_chunker import chunk_documents
from core.ingestion.pinecone_store import PineconeStore


@dataclass
class IngestionResult:
    success: bool
    message: str
    markdown_pages: int
    documents: int
    chunks: int
    elapsed_seconds: float


async def ingest_prospectus(
    pdf_path: str,
    academic_level: str,
    year: int,
    seat_distribution_output: str = "seat_distribution.pdf",
    excluded_pages: list[int] = None
) -> IngestionResult:
    start = perf_counter()
    temp_pdf = None
    try:
        print(f"[INGESTION] Starting ingestion for {academic_level.upper()} prospectus...")
        temp_pdf = split_pdf(
            pdf_path,
            excluded_pages=excluded_pages or [],
            output_seat_path=seat_distribution_output,
        )
        print(f"[INGESTION] PDF Split complete. Temporary file: {temp_pdf}")

        print(f"[INGESTION] Phase 1/4: Parsing PDF into markdown layout...")
        markdown = await parse_pdf_to_markdown_async(temp_pdf)
        print(f"[INGESTION] Phase 1/4 Complete. Extracted {markdown.count('## Page')} pages of markdown.")

        print(f"[INGESTION] Phase 2/4: Extracting documents and hierarchy...")
        docs = extract_documents(
            markdown=markdown,
            academic_level=academic_level,
            year=year,
        )
        print(f"[INGESTION] Phase 2/4 Complete. Extracted {len(docs)} structural documents.")

        print(f"[INGESTION] Phase 3/4: Semantic chunking of documents...")
        chunks = chunk_documents(docs)
        print(f"[INGESTION] Phase 3/4 Complete. Created {len(chunks)} retrieval chunks.")

        print(f"[INGESTION] Phase 4/4: Uploading chunks to Pinecone vector store...")
        store = PineconeStore()
        store.replace_documents(
            documents=chunks,
            academic_level=academic_level,
        )
        print(f"[INGESTION] Phase 4/4 Complete. Successfully updated Pinecone.")

        import json
        import os
        os.makedirs("output_chunks", exist_ok=True)
        prefix = "UG" if academic_level == "undergraduate" else "PG"
        json_path = os.path.join("output_chunks", f"{prefix}Prospectus_compiled_knowledge.json")
        
        json_data = []
        rag_keywords = set()
        for doc in chunks:
            item = dict(doc.metadata)
            item["text"] = doc.page_content
            json_data.append(item)
            
            dept = item.get("department")
            if dept:
                dept_clean = str(dept).lower().replace("department of ", "").replace("dept of ", "").strip()
                for w in dept_clean.split():
                    w = ''.join(c for c in w if c.isalpha())
                    if len(w) > 3:
                        rag_keywords.add(w)
            
            designation = item.get("designation")
            if designation:
                desig_clean = str(designation).lower().strip()
                for w in desig_clean.split():
                    w = ''.join(c for c in w if c.isalpha())
                    if len(w) > 3:
                        rag_keywords.add(w)
            
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)
            
        dept_path = os.path.join("output_chunks", "departments.json")
        existing_keywords = []
        if os.path.exists(dept_path):
            try:
                with open(dept_path, "r", encoding="utf-8") as f:
                    existing_keywords = json.load(f)
            except Exception:
                pass
        
        all_keywords = list(set(existing_keywords) | rag_keywords)
        with open(dept_path, "w", encoding="utf-8") as f:
            json.dump(all_keywords, f, indent=4)

        return IngestionResult(
            success=True,
            message="Prospectus ingested successfully.",
            markdown_pages=markdown.count("## Page"),
            documents=len(docs),
            chunks=len(chunks),
            elapsed_seconds=round(perf_counter()-start,2),
        )

    except Exception as exc:
        return IngestionResult(
            success=False,
            message=str(exc),
            markdown_pages=0,
            documents=0,
            chunks=0,
            elapsed_seconds=round(perf_counter()-start,2),
        )
    finally:
        if temp_pdf:
            cleanup_temp_file(temp_pdf)


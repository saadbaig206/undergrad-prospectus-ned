# 
import os
import json
import time
import asyncio
from pathlib import Path
import pypdf
from llama_parse import LlamaParse
from dotenv import load_dotenv
from core.admin_process import EXCLUDED_PAGES

load_dotenv()

# Initialize global shared cloud parsers
text_parser = LlamaParse(
    result_type="markdown", 
    api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
    system_prompt="""This document contains university undergraduate prospectus pages.
Many pages contain course curriculum tables. For each course table:
- Extract the course code, course title, and credit hours (Theory, Practical, and Total).
- Format the credit hours clearly as 'Th-Pr-Total' (e.g. 3-1-4 or 3-0-3 or NC-NC-NC).
- Do not merge adjacent columns or numbers.
- Keep the tables intact as markdown tables."""
)
data_parser = LlamaParse(
    result_type="json", 
    mode="premium", 
    parse_page_as_managed_table=True,
    parsing_instruction="Extract this matrix geometrically. Pin multi-tier headers like CE (UE) and CE (CN) directly to their numerical column coordinates."
)

# Limit concurrent API calls to prevent rate-limiting (HTTP 429)
SEMAPHORE = asyncio.Semaphore(5)

async def parse_single_page_async(page_num: int, page_bytes: bytes, is_matrix_page: bool):
    """Worker task that handles a single page API call concurrently with rate limit handling"""
    async with SEMAPHORE:
        temp_filename = f"temp_async_p_{page_num}.pdf"
        
        # Write page bytes to disk temporarily for the API wrapper
        with open(temp_filename, "wb") as f:
            f.write(page_bytes)
            
        try:
            # LANE A: Parallel Data Processing
            if is_matrix_page:
                # Trigger asynchronous JSON structural extraction via thread executor
                json_result = await asyncio.to_thread(data_parser.get_json_result, temp_filename)
                chunks = []
                if json_result and "pages" in json_result[0]:
                    tables = json_result[0]["pages"][0].get("tables", [])
                    for table in tables:
                        cols = table.get("columns", [])
                        for row in table.get("rows", {}):
                            category = row.get("Category", "Unknown Quota")
                            row_text = f"Quota Category {category} seat distribution: "
                            details = [f"{col} has {row.get(col, '0')} seats" for col in cols if col != "Category" and row.get(col)]
                            row_text += ", ".join(details) + "."
                            chunks.append({
                                "text": row_text,
                                "source_page": page_num,
                                "content_type": "structured_table_row"
                            })
                return chunks

            # LANE B: Parallel Text Processing
            else:
                # Trigger asynchronous Markdown parsing
                text_result = await text_parser.aload_data(temp_filename)
                page_markdown = "\n\n".join([doc.text for doc in text_result])
                return [{
                    "text": page_markdown,
                    "source_page": page_num,
                    "content_type": "markdown_prose"
                }]
                
        except Exception as e:
            print(f"❌ Async error on page {page_num}: {e}")
            return []
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

def index_chunks_to_pinecone(chunks, index_name="rag-chatbot-index"):
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Pinecone API key is not set. Skipping vector database indexing.")
        return

    print("Loading local embedding model for indexing...")
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone, ServerlessSpec
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pc = Pinecone(api_key=api_key)
    
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    
    try:
        print("Clearing existing index...")
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Could not clear index: {e}")
        
    print(f"Embedding and uploading {len(chunks)} chunks to Pinecone...")
    vectors = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        embedding = model.encode(text).tolist()
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": text,
                "source_page": chunk["source_page"],
                "content_type": chunk["content_type"]
            }
        })
        
        if len(vectors) == 100 or i == len(chunks) - 1:
            index.upsert(vectors=vectors)
            vectors = []
            
    print("Pinecone indexing complete!")

async def run_parallel_pipeline(pdf_path: str, seat_matrix_pages: list, output_dir: str):
    start_time = time.time()
    pdf_file = Path(pdf_path)
    reader = pypdf.PdfReader(pdf_file)
    total_pages = len(reader.pages)
    
    print(f"Launching Async Parallel Pipeline for {total_pages} pages...")
    
    tasks = []
    for page_num in range(1, total_pages + 1):
        # Skip seat matrix pages entirely as per user request to ensure no text is extracted from them
        if page_num in seat_matrix_pages:
            print(f"Page {page_num} is a split page. Skipping text extraction.")
            continue
            
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[page_num - 1])
        
        import io
        mem_buf = io.BytesIO()
        writer.write(mem_buf)
        page_bytes = mem_buf.getvalue()
        
        tasks.append(parse_single_page_async(page_num, page_bytes, is_matrix_page=False))
        
    print(f"Dispatching parallel network workers...")
    results = await asyncio.gather(*tasks)
    
    final_rag_chunks = [chunk for page_result in results for chunk in page_result]
    final_rag_chunks.sort(key=lambda x: x["source_page"])
    
    os.makedirs(output_dir, exist_ok=True)
    output_payload_path = Path(output_dir) / f"{pdf_file.stem}_compiled_knowledge.json"
    with open(output_payload_path, "w", encoding="utf-8") as f:
        json.dump(final_rag_chunks, f, indent=4, ensure_ascii=False)
        
    print(f"\nParallel pipeline finished in: {time.time() - start_time:.2f} seconds!")
    print(f"Saved to: {output_payload_path}")

    # Index the chunks to Pinecone database
    index_chunks_to_pinecone(final_rag_chunks)

def main():
    # Dynamically read 1-based page numbers from admin_process EXCLUDED_PAGES
    seat_pages = [p + 1 for p in EXCLUDED_PAGES]
    print(f"Seat matrix pages (1-based) to exclude: {seat_pages}")
    asyncio.run(run_parallel_pipeline("UGProspectus2025.pdf", seat_matrix_pages=seat_pages, output_dir="output_chunks"))

if __name__ == "__main__":
    main()
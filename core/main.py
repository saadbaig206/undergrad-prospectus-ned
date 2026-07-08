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
Extract all content on the page completely and accurately, preserving headings, lists, paragraphs, and tables in markdown.
For pages with course curriculum tables:
- Extract the course code, course title, and credit hours (Theory, Practical, and Total).
- Format the credit hours clearly as 'Th-Pr-Total' (e.g. 3-1-4 or 3-0-3).
- Keep tables intact as markdown tables."""
)
# Limit concurrent API calls to prevent rate-limiting (HTTP 429)
SEMAPHORE = asyncio.Semaphore(5)

async def parse_single_page_async(page_num: int, page_bytes: bytes):
    """Worker task that handles a single page API call concurrently with rate limit handling"""
    async with SEMAPHORE:
        temp_filename = f"temp_async_p_{page_num}.pdf"
        
        # Write page bytes to disk temporarily for the API wrapper
        with open(temp_filename, "wb") as f:
            f.write(page_bytes)
            
        try:
            # Trigger asynchronous Markdown parsing
            text_result = await text_parser.aload_data(temp_filename)
            page_markdown = "\n\n".join([doc.text for doc in text_result])
            
            # Split page markdown into smaller overlapping chunks
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            split_texts = text_splitter.split_text(page_markdown)
            return [{
                "text": chunk_text,
                "source_page": page_num,
                "content_type": "markdown_prose"
            } for chunk_text in split_texts]
                
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

    print("Loading local Model2Vec embedding model for indexing...")
    from core.embeddings import embed_documents
    from pinecone import Pinecone, ServerlessSpec
    
    pc = Pinecone(api_key=api_key)
    
    # Robust index name listing across different pinecone SDK versions
    existing_indexes = pc.list_indexes()
    index_names = []
    for idx in existing_indexes:
        if isinstance(idx, str):
            index_names.append(idx)
        elif hasattr(idx, 'name'):
            index_names.append(idx.name)
        elif isinstance(idx, dict) and 'name' in idx:
            index_names.append(idx['name'])
            
    should_create = True
    if index_name in index_names:
        try:
            desc = pc.describe_index(index_name)
            dim = desc.dimension if hasattr(desc, 'dimension') else desc.get('dimension')
            if dim != 256:
                print(f"Index {index_name} exists but with dimension {dim}. Recreating with dimension 256...")
                pc.delete_index(index_name)
                import time
                time.sleep(3)
            else:
                should_create = False
        except Exception as e:
            print(f"Error checking index description: {e}. Recreating index...")
            try:
                pc.delete_index(index_name)
            except Exception:
                pass
            import time
            time.sleep(3)
            
    if should_create:
        print(f"Creating Pinecone index: {index_name} with dimension 256...")
        pc.create_index(
            name=index_name,
            dimension=256,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    
    try:
        print("Clearing existing index...")
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Could not clear index: {e}")
        
    print(f"Embedding {len(chunks)} chunks using Model2Vec...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_documents(texts)
    
    print(f"Uploading vectors to Pinecone...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": f"chunk_{i}",
            "values": embeddings[i],
            "metadata": {
                "text": chunk["text"],
                "source_page": chunk["source_page"],
                "content_type": chunk["content_type"]
            }
        })
        
    # Batch upsert to prevent payload size issues
    batch_size = 100
    for idx in range(0, len(vectors), batch_size):
        batch = vectors[idx:idx + batch_size]
        index.upsert(vectors=batch)
            
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
        
        tasks.append(parse_single_page_async(page_num, page_bytes))
        
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

    index_chunks_to_pinecone(final_rag_chunks)

def main():
    seat_pages = EXCLUDED_PAGES
    print(f"Seat matrix pages (1-based) to exclude: {seat_pages}")
    asyncio.run(run_parallel_pipeline("UGProspectus.pdf", seat_matrix_pages=seat_pages, output_dir="output_chunks"))

if __name__ == "__main__":
    main()
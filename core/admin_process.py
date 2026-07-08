import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import fitz  
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.embeddings import embed_documents

nest_asyncio.apply()
load_dotenv()

PDF_PATH = "UGProspectus.pdf"
SEAT_DIST_PDF_PATH = "seat_distribution.pdf"
MARKDOWN_OUTPUT_PATH = "extracted_content.md"
PINECONE_INDEX_NAME = "rag-chatbot-index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Pages the admin wants to exclude from text parsing (1-indexed)
EXCLUDED_PAGES = [79, 80, 81] 

# --- 1. Split the PDF ---
def split_pdf(input_path, excluded_pages, output_seat_path):
    """
    Split the PDF using PyMuPDF instead of PyPDF2.
    PyMuPDF preserves the original PDF structure much better,
    making it compatible with LlamaParse.
    """

    doc = fitz.open(input_path)

    main_doc = fitz.open()
    seat_doc = fitz.open()

    main_pdf_path = "temp_main_extracted.pdf"

    for page_no in range(doc.page_count):

        if page_no in excluded_pages:
            seat_doc.insert_pdf(
                doc,
                from_page=page_no,
                to_page=page_no
            )
        else:
            main_doc.insert_pdf(
                doc,
                from_page=page_no,
                to_page=page_no
            )

    # Save without aggressive compression
    main_doc.save(
        main_pdf_path,
        garbage=0,
        clean=False,
        deflate=False
    )

    seat_doc.save(
        output_seat_path,
        garbage=0,
        clean=False,
        deflate=False
    )

    main_doc.close()
    seat_doc.close()
    doc.close()

    print(f" Saved main PDF to: {main_pdf_path}")
    print(f" Saved seat distribution PDF to: {output_seat_path}")

    return main_pdf_path

async def parse_pdf_to_markdown_async(pdf_path):
    print("Parsing PDF content...")
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")
    
    # Check if the key is empty or a placeholder
    is_placeholder = False
    if api_key:
        if api_key.startswith("llm_") or "placeholder" in api_key.lower() or "your_" in api_key.lower():
            is_placeholder = True

    if api_key and not is_placeholder:
        try:
            print("Using LlamaParse to generate markdown data (chunked for stability)...")
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            print(f"Total pages to parse: {total_pages}")
            
            chunk_size = 10
            tasks = []
            temp_files = []
            
            parser = LlamaParse(
                result_type="markdown", 
                api_key=api_key,
                system_prompt="""This document contains university undergraduate prospectus pages.
Many pages contain course curriculum tables. For each course table:
- Extract the course code, course title, and credit hours (Theory, Practical, and Total).
- Format the credit hours clearly as 'Th-Pr-Total' (e.g. 3-1-4 or 3-0-3 or NC-NC-NC).
- Do not merge adjacent columns or numbers.
- Keep the tables intact as markdown tables."""
            )
            semaphore = asyncio.Semaphore(2)
            
            async def parse_chunk_with_sem(chunk_path, chunk_idx):
                async with semaphore:
                    print(f"Uploading and parsing chunk {chunk_idx + 1}... (path: {chunk_path})")
                    await asyncio.sleep(chunk_idx * 0.5)
                    try:
                        docs = await parser.aload_data(chunk_path)
                        text = "\n\n".join([doc.text for doc in docs if getattr(doc, "text", None)])
                        print(f"Chunk {chunk_idx + 1} parsed successfully.")
                        return text
                    except Exception as e:
                        print(f"Error parsing chunk {chunk_idx + 1}: {e}")
                        raise e
            
            for i in range(0, total_pages, chunk_size):
                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(doc, from_page=i, to_page=min(i + chunk_size - 1, total_pages - 1))
                
                temp_chunk_path = f"temp_chunk_{i}.pdf"
                chunk_doc.save(temp_chunk_path)
                chunk_doc.close()
                temp_files.append(temp_chunk_path)
                
                tasks.append(parse_chunk_with_sem(temp_chunk_path, len(tasks)))
                
            try:
                results = await asyncio.gather(*tasks)
                full_text = "\n\n".join(results)
                if full_text.strip():
                    print("LlamaParse extraction successful.")
                    return full_text
            finally:
                doc.close()
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass
        except Exception as exc:
            print(f"LlamaParse failed: {exc}. Falling back to local PDF extraction.")
    elif is_placeholder:
        print("Llama Cloud API key appears to be a placeholder. Skipping LlamaParse...")
    else:
        print("Llama Cloud API key is not set. Skipping LlamaParse...")

    print("Falling back to local PDF text extraction...")
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text() or ""
            cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if cleaned_text.strip():
                pages_text.append(f"## Page {page_num + 1}\n{cleaned_text}")

        if pages_text:
            doc.close()
            return "\n\n".join(pages_text)
        else:
            print("No text could be extracted locally. Checking if PDF contains images...")
            has_images = False
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                if len(page.get_images()) > 0:
                    has_images = True
                    break
            doc.close()
            
            if has_images:
                error_msg = (
                    "# Warning: Scanned PDF Detected\n\n"
                    "This PDF contains scanned images and no embedded text. "
                    "Extracting text from this document requires a valid Llama Cloud API key (LLAMA_CLOUD_API_KEY) for OCR.\n\n"
                    "Please set a valid key in your `.env` file and re-run `admin_process.py`."
                )
                print("\n" + "="*80)
                print("WARNING: Scanned PDF detected, but no valid Llama Cloud API key is configured.")
                print("Text extraction cannot proceed. Please update your .env file.")
                print("="*80 + "\n")
                return error_msg
    except Exception as exc:
        print(f"Local PDF extraction failed: {exc}")

    return "# Extracted content\n\nNo readable text could be extracted from the PDF."

# --- 3. Chunk and Embed locally, then Upsert to Pinecone ---
def cleanup_temp_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except (PermissionError, OSError) as exc:
            print(f"Could not remove temporary file {file_path}: {exc}")

def index_to_pinecone(text, index_name):
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Pinecone API key is not set. Skipping vector pipeline index.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # Check if we have anything to upload to avoid Pinecone payload faults
    if not chunks or (len(chunks) == 1 and not chunks[0].strip()):
        print("Warning: Aborting vector database upload because text contents are empty.")
        return
        
    pc = Pinecone(api_key=api_key)
    
    # Recreate index if dimension has changed to 1024
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            desc = pc.describe_index(index_name)
            if desc.dimension != 1024:
                print(f"Index {index_name} has incorrect dimension {desc.dimension}. Recreating it with 1024 dimensions...")
                pc.delete_index(index_name)
    except Exception as e:
        print(f"Warning checking/deleting index: {e}")
    
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index: {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
    except Exception as e:
        print(f"Failed to create index: {e}")
        raise e
    
    index = pc.Index(index_name)
    
    # Sweep the index clean to prevent stale or duplicate vectors from previous uploads
    try:
        print("Clearing existing index vectors...")
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Could not clear index: {e}")
        
    print(f"Embedding and preparing {len(chunks)} chunks to Pinecone...")
    vectors = []
    batch_size = 96
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        try:
            embeddings = embed_documents(batch_chunks)
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                global_idx = i + j
                vectors.append({
                    "id": f"chunk_{global_idx}",
                    "values": embedding,
                    "metadata": {"text": chunk}
                })
        except Exception as e:
            print(f"Failed to generate embeddings for batch {i // batch_size + 1}: {e}")
            raise e
            
    print(f"Uploading {len(vectors)} vectors to Pinecone...")
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i : i + 100])
            
    print("Indexing complete!")

async def main():
    # 1. Isolate exclusions 
    cleaned_pdf = split_pdf(PDF_PATH, EXCLUDED_PAGES, SEAT_DIST_PDF_PATH)
    
    # 2. Complete extraction safely
    markdown_text = await parse_pdf_to_markdown_async(cleaned_pdf)
    
    # 3. Output structural save
    print(f"Saving markdown text to local file: {MARKDOWN_OUTPUT_PATH}...")
    with open(MARKDOWN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print("Markdown file saved.")
    
    # 4. Generate embeddings pipeline
    index_to_pinecone(markdown_text, PINECONE_INDEX_NAME)
    
    # 5. Drop temp components
    cleanup_temp_file(cleaned_pdf)

if __name__ == "__main__":
    asyncio.run(main())
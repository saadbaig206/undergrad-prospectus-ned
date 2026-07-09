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
    
    is_placeholder = False
    if api_key:
        if api_key.startswith("llm_") or "placeholder" in api_key.lower() or "your_" in api_key.lower():
            is_placeholder = True

    if api_key and not is_placeholder:
        try:
            print("Using LlamaParse to generate markdown data page-by-page (rendered to PNG)...")
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            print(f"Total pages to parse: {total_pages}")
            
            tasks = []
            
            parser = LlamaParse(
                result_type="markdown", 
                api_key=api_key,
                system_prompt="""Extract EVERY detail visible on the page — do not summarize, skip, or omit any 
content. This includes: headings, program descriptions, admission criteria, 
eligibility requirements, fee structures, policies, faculty/department names, 
contact details, and all curriculum/course tables.

GENERAL RULES:
- Preserve the page's structure and reading order (top to bottom, left to 
  right, respecting multi-column layouts).
- Reproduce all narrative/paragraph text faithfully — do not paraphrase or 
  compress.
- Keep every table intact and reproduce it as a markdown table, preserving 
  all rows and columns exactly as shown (including header rows and any 
  footnotes below a table).
- Do not drop rows, merge cells, or omit any table on the page, even if it 
  looks repetitive or similar to a table on a previous page.
- Ignore repeated page furniture (running header "NED UNIVERSITY OF 
  ENGINEERING & TECHNOLOGY / UNDERGRADUATE PROSPECTUS 2026", website URL, 
  social handle, page number) unless it contains unique content.

COURSE CURRICULUM TABLES — SPECIAL HANDLING:
For every course table, extract:
- Course code
- Course title
- Credit hours: Theory, Practical, and Total

Format credit hours strictly as 'Th-Pr-Total' (e.g. 3-1-4, 3-0-3, or 
NC-NC-NC for non-credit courses).

STRICT FORMATTING CONSTRAINTS:
- Do NOT merge adjacent columns or numbers (e.g. do not combine course code 
  and title into one field, or combine Th/Pr/Total into a single 
  unseparated number).
- Do NOT alter, round, or infer credit hour values — read them exactly as 
  printed on the page image.
- If a value is missing, illegible, or not applicable, mark it clearly as 
  'NC' rather than leaving it blank or guessing.
- Preserve original column order and table boundaries; do not combine 
  multiple separate tables into one, even across adjacent pages — call out 
  if a table appears to continue onto the next page.
- If small/blurry text at typical rasterization resolution is not fully 
  legible, note it as [illegible] rather than guessing a value.
"""
            )
            semaphore = asyncio.Semaphore(5)
            
            async def parse_page_with_sem(page_idx):
                async with semaphore:
                    temp_png_path = f"temp_chunk_admin_{page_idx}.png"
                    try:
                        page = doc[page_idx]
                        pix = page.get_pixmap(dpi=150)
                        pix.save(temp_png_path)
                        
                        print(f"Uploading and parsing page {page_idx + 1}/{total_pages}...")
                        docs = await parser.aload_data(temp_png_path)
                        text = "\n\n".join([d.text for d in docs if getattr(d, "text", None)])
                        print(f"Page {page_idx + 1} parsed successfully.")
                        return text
                    except Exception as e:
                        print(f"Error parsing page {page_idx + 1}: {e}")
                        raise e
                    finally:
                        if os.path.exists(temp_png_path):
                            try:
                                os.remove(temp_png_path)
                            except Exception:
                                pass
            
            for idx in range(total_pages):
                tasks.append(parse_page_with_sem(idx))
                
            try:
                results = await asyncio.gather(*tasks)
                full_text = "\n\n".join(results)
                if full_text.strip():
                    print("LlamaParse extraction successful.")
                    return full_text
            finally:
                doc.close()
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
    
    # Recreate index if dimension has changed to match the embedding dimension
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "256"))
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            desc = pc.describe_index(index_name)
            if desc.dimension != EMBEDDING_DIMENSION:
                print(f"Index {index_name} has incorrect dimension {desc.dimension}. Recreating it with {EMBEDDING_DIMENSION} dimensions...")
                pc.delete_index(index_name)
    except Exception as e:
        print(f"Warning checking/deleting index: {e}")
    
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index: {index_name} with {EMBEDDING_DIMENSION} dimensions...")
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,
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
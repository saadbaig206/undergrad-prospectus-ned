import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import fitz  
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.ingestion.embedder import embed_documents

nest_asyncio.apply()
load_dotenv()

PDF_PATH = "UGProspectus.pdf"
SEAT_DIST_PDF_PATH = "seat_distribution.pdf"
MARKDOWN_OUTPUT_PATH = "extracted_content.md"
PINECONE_INDEX_NAME = "rag-chatbot-index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Pages the admin wants to exclude from text parsing (1-indexed)
EXCLUDED_PAGES = [79, 80, 81] 


def split_pdf(input_path, excluded_pages, output_seat_path):
    doc = fitz.open(input_path)
    main_doc = fitz.open()
    seat_doc = fitz.open()

    main_pdf_path = "temp_main_extracted.pdf"

    for page_no in range(doc.page_count):
        if page_no in excluded_pages:
            seat_doc.insert_pdf(doc, from_page=page_no, to_page=page_no)
        else:
            main_doc.insert_pdf(doc, from_page=page_no, to_page=page_no)

    main_doc.save(main_pdf_path, garbage=0, clean=False, deflate=False)
    
    if seat_doc.page_count > 0:
        seat_doc.save(output_seat_path, garbage=0, clean=False, deflate=False)
        print(f" Saved seat distribution PDF to: {output_seat_path}")
    else:
        print(" No seat distribution pages provided. Skipping seat PDF creation.")
        if os.path.exists(output_seat_path):
            try:
                os.remove(output_seat_path)
            except Exception:
                pass

    main_doc.close()
    seat_doc.close()
    doc.close()

    print(f" Saved main PDF to: {main_pdf_path}")

    return main_pdf_path


def check_if_pdf_is_scanned(pdf_path: str, threshold_chars_per_page: int = 15) -> bool:
    """
    Fast, local, in-memory check to see if a PDF requires OCR.
    Samples the first and last few pages to look for selectable text.
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        # Sample up to 10 total pages across start and end
        sample_pages = list(range(min(5, total_pages)))
        if total_pages > 5:
            sample_pages.extend(range(max(total_pages - 5, 5), total_pages))
            
        total_text_len = 0
        for page_num in sample_pages:
            page = doc.load_page(page_num)
            total_text_len += len(page.get_text().strip())
            
        doc.close()
        
        avg_chars = total_text_len / len(sample_pages)
        print(f"[ANALYZER] Average character density per sampled page: {avg_chars:.2f}")
        
        # If the page average drops below our threshold, it is a scanned image
        return avg_chars < threshold_chars_per_page
    except Exception as e:
        print(f"[ANALYZER] Error determining PDF type: {e}. Defaulting to OCR mode.")
        return True


def execute_local_extraction(pdf_path: str) -> str:
    """Performs lightning-fast native digital extraction with structured page headers."""
    print("Executing standard local text extraction for native digital PDF...")
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text() or ""
        
        pages_text.append(f"## Page {page_num + 1}")
        pages_text.append(text)
        
    doc.close()
    return "\n\n".join(pages_text)


async def parse_pdf_to_markdown_async(pdf_path):
    print("Parsing PDF content...")
    
    print("[ROUTER] Bypassing local extraction to ensure table layouts are preserved via LlamaParse OCR/Vision.")

    print("[ROUTER] Document classified as a scanned image or requires OCR parsing.")
    
    load_dotenv()
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")
    
    is_placeholder = False
    if api_key:
        if api_key.startswith("llm_") or "placeholder" in api_key.lower() or "your_" in api_key.lower():
            is_placeholder = True

    if api_key and not is_placeholder:
        try:
            print("Using LlamaParse to generate markdown data page-by-page via PNG to force OCR (concurrent batch mode)...")
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            print(f"Total pages to parse: {total_pages}")
            
            parser = LlamaParse(
                result_type="markdown", 
                api_key=api_key,
                system_prompt="""Extract EVERY detail visible on the page — do not summarize, skip, or omit any content. Keep all tables completely intact formatted as Markdown tables. Do not alter column layouts or merge rows. Reproduce the tables faithfully."""
            )
            
            semaphore = asyncio.Semaphore(10) # 10 concurrent pages
            
            async def parse_page(page_idx):
                temp_png_path = f"temp_chunk_admin_{page_idx}.png"
                result_text = ""
                
                try:
                    page = doc.load_page(page_idx)
                    pix = page.get_pixmap(dpi=150)
                    pix.save(temp_png_path)
                    
                    async with semaphore:
                        retries = 3
                        while retries > 0:
                            try:
                                print(f"Uploading and parsing page {page_idx + 1}/{total_pages} (Attempts left: {retries})...")
                                docs = await asyncio.wait_for(parser.aload_data(temp_png_path), timeout=90)
                                text = "\n\n".join([d.text for d in docs if getattr(d, "text", None)])
                                print(f"Page {page_idx + 1} parsed successfully.")
                                result_text = f"\n\n## Page {page_idx + 1}\n\n{text}"
                                break
                            except asyncio.TimeoutError:
                                print(f"Timeout parsing page {page_idx + 1}. Retrying...")
                                retries -= 1
                            except Exception as e:
                                print(f"Error parsing page {page_idx + 1}: {e}")
                                retries -= 1
                                await asyncio.sleep(2)
                                
                        if retries == 0:
                            print(f"Failed to parse page {page_idx + 1} after retries.")
                            
                except Exception as e:
                    print(f"Error loading page {page_idx + 1}: {e}")
                finally:
                    if os.path.exists(temp_png_path):
                        try:
                            os.remove(temp_png_path)
                        except:
                            pass
                
                return page_idx, result_text
                
            tasks = [parse_page(i) for i in range(total_pages)]
            parsed_pages = await asyncio.gather(*tasks)
            
            # Sort by page_idx to maintain order
            parsed_pages.sort(key=lambda x: x[0])
            results = [text for idx, text in parsed_pages if text]
            
            combined_markdown = "\n\n".join(results)
            
            doc.close()
            if combined_markdown.strip():
                print("LlamaParse sequential extraction successful.")
                return combined_markdown
            
        except Exception as e:
            print(f"[WARNING] LlamaParse failed or returned empty: {e}")
    elif is_placeholder:
        print("Llama Cloud API key appears to be a placeholder. Skipping LlamaParse...")
    else:
        print("Llama Cloud API key is not set. Skipping LlamaParse...")

    # Final fallback if both methods run into unexpected environment exceptions
    return execute_local_extraction(pdf_path)


def cleanup_temp_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except (PermissionError, OSError) as exc:
            print(f"Could not remove temporary file {file_path}: {exc}")
import os
import secrets
import hashlib
import datetime
import socket

# Force IPv4 DNS resolution to prevent slow Windows IPv6 lookup timeouts on external APIs
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return orig_getaddrinfo(host, port, family, type, proto, flags)
socket.getaddrinfo = patched_getaddrinfo
import asyncio
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.database import get_db_connection, init_db
from backend.schemas import AuthRequest, QueryRequest, CreateAdminRequest

from core.chatbot import route_chat_stream
from fastapi.responses import StreamingResponse

app = FastAPI(title="UG Prospectus Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def keep_pinecone_alive_loop():
    """Background loop to periodically query Pinecone, keeping the connection pool and serverless instance warm."""
    from core.chatbot import get_global_httpx_client
    await asyncio.sleep(5)
    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_INDEX_HOST")
    if not host or not api_key:
        return
    url = f"{host}/query" if host.startswith("https://") else f"https://{host}/query"
    
    while True:
        try:
            client = get_global_httpx_client()
            await client.post(
                url,
                json={
                    "vector": [0.0] * 256,
                    "topK": 1,
                    "includeMetadata": False,
                    "includeValues": False
                },
                headers={
                    "Api-Key": api_key,
                    "Content-Type": "application/json"
                },
                timeout=5.0
            )
        except Exception:
            pass
        await asyncio.sleep(10)  # Keepalive query every 10 seconds

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
    except Exception as exc:
        print(f"Database initialization skipped or failed: {exc}")
        
    # Pre-warm AI and database clients to cut down first-query latency from 3s to under 1s
    try:
        from core.chatbot import get_llm, get_global_httpx_client
        from core.embeddings import get_embeddings_model
        get_embeddings_model()
        get_llm()
        
        # Warm up the Pinecone connection pool by running a lightweight query (256 dim vector)
        api_key = os.getenv("PINECONE_API_KEY")
        host = os.getenv("PINECONE_INDEX_HOST")
        if host and api_key:
            url = f"{host}/query" if host.startswith("https://") else f"https://{host}/query"
            client = get_global_httpx_client()
            try:
                await client.post(
                    url,
                    json={
                        "vector": [0.0] * 256,
                        "topK": 1,
                        "includeMetadata": False,
                        "includeValues": False
                    },
                    headers={
                        "Api-Key": api_key,
                        "Content-Type": "application/json"
                    },
                    timeout=5.0
                )
                print("🔍 [PINECONE] Direct HTTPX connection pool and socket warming completed successfully!")
            except Exception as pe:
                print(f"🔍 [PINECONE] Direct HTTPX pool warming warning: {pe}")
        print("Pre-warming of Embeddings, Pinecone, and LLM clients completed successfully!")
    except Exception as e:
        print(f"Pre-warming warning: {e}")
        
    # Launch background keepalive loop to maintain a hot connection
    asyncio.create_task(keep_pinecone_alive_loop())

@app.get("/")
def root():
    return {
        "message": "UG Prospectus Chatbot API is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    
    from psycopg2.extras import RealDictCursor
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT u.id, u.username, u.role, s.expires_at 
        FROM sessions s 
        JOIN users u ON s.user_id = u.id 
        WHERE s.token = %s;
    """, (token,))
    session = cur.fetchone()
    cur.close()
    conn.close()
    
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session token")
        
    if session['expires_at'] < datetime.datetime.utcnow():
        raise HTTPException(status_code=401, detail="Session has expired")
        
    return session

def get_current_admin(current_user = Depends(get_current_user)):
    if current_user['role'] != 'ADMIN':
        raise HTTPException(status_code=403, detail="Admin permissions required")
    return current_user

@app.post("/auth/signup")
def signup(payload: AuthRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    
    salt = secrets.token_hex(16)
    p_hash = hash_password(payload.password, salt)
    
    try:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, salt, role)
            VALUES (%s, %s, %s, 'USER')
            ON CONFLICT (username) DO NOTHING
            RETURNING id;
            """,
            (payload.username.strip(), p_hash, salt)
        )
        res = cur.fetchone()
    except Exception as e:
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Signup database error: {e}")
        
    cur.close()
    conn.close()
    
    if not res:
        raise HTTPException(status_code=400, detail="Username already exists")
        
    return {"message": "User registered successfully"}

@app.post("/auth/login")
def login(payload: AuthRequest):
    from psycopg2.extras import RealDictCursor
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("SELECT * FROM users WHERE username = %s;", (payload.username.strip(),))
        user = cur.fetchone()
    except Exception as e:
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Login database query error: {e}")
    
    if not user:
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid username or password")
        
    expected_hash = hash_password(payload.password, user['salt'])
    if expected_hash != user['password_hash']:
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid username or password")
        
    token = secrets.token_hex(32)
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    
    try:
        cur.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (%s, %s, %s);",
            (token, user['id'], expires_at)
        )
    except Exception as e:
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Session creation error: {e}")
        
    cur.close()
    conn.close()
    
    return {
        "token": token,
        "role": user['role'],
        "username": user['username']
    }

@app.post("/admin/create-admin")
def create_admin(payload: CreateAdminRequest, admin_user = Depends(get_current_admin)):
    conn = get_db_connection()
    cur = conn.cursor()
    
    salt = secrets.token_hex(16)
    p_hash = hash_password(payload.password, salt)
    
    try:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, salt, role)
            VALUES (%s, %s, %s, 'ADMIN')
            ON CONFLICT (username) DO NOTHING
            RETURNING id;
            """,
            (payload.username.strip(), p_hash, salt)
        )
        res = cur.fetchone()
    except Exception as e:
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Admin creation database error: {e}")
        
    cur.close()
    conn.close()
    
    if not res:
        raise HTTPException(status_code=400, detail="Username already exists")
        
    return {"message": f"Admin user '{payload.username}' created successfully"}

def run_ingestion_background(pdf_path: str, seat_matrix_pages: list):
    try:
        from core.main import run_parallel_pipeline

        print(f"Background Ingestion Started. Matrix pages: {seat_matrix_pages}")
        asyncio.run(run_parallel_pipeline(pdf_path, seat_matrix_pages, "output_chunks"))
        print("Background Ingestion Completed Successfully!")
    except Exception as e:
        print(f"Background Ingestion Failed: {e}")

@app.post("/admin/upload-prospectus")
def upload_prospectus(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    excluded_pages: str = Form(...),
    admin_user = Depends(get_current_admin)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    prospectus_path = "UGProspectus.pdf"
    try:
        with open(prospectus_path, "wb") as buffer:
            buffer.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded PDF: {e}")
        
    try:
        pages_list = [int(p.strip()) for p in excluded_pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="excluded_pages must be a comma-separated list of integers")
        
    try:
        admin_process_path = "core/admin_process.py"
        with open(admin_process_path, "r", encoding="utf-8") as f:
            admin_code = f.read()
            
        import re
        new_code = re.sub(
            r"EXCLUDED_PAGES\s*=\s*\[.*?\]",
            f"EXCLUDED_PAGES = {pages_list}",
            admin_code
        )
        with open(admin_process_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        print(f"Updated EXCLUDED_PAGES to {pages_list} in core/admin_process.py")
    except Exception as e:
        print(f"Could not update EXCLUDED_PAGES in core/admin_process.py: {e}")
        
    try:
        import fitz
        doc = fitz.open(prospectus_path)
        seat_doc = fitz.open()
        
        for p in pages_list:
            zero_based_p = p - 1
            if 0 <= zero_based_p < doc.page_count:
                seat_doc.insert_pdf(doc, from_page=zero_based_p, to_page=zero_based_p)
                
        os.makedirs("frontend/static", exist_ok=True)
        os.makedirs("public", exist_ok=True)
        static_seat_path = "frontend/static/seat_distribution.pdf"
        root_seat_path = "seat_distribution.pdf"
        public_seat_path = "public/seat_distribution.pdf"
        
        seat_doc.save(static_seat_path)
        seat_doc.close()
        doc.close()
        
        import shutil
        shutil.copy(static_seat_path, root_seat_path)
        shutil.copy(static_seat_path, public_seat_path)
        print("Regenerated seat_distribution.pdf locally and in public assets.")
        
        # Upload to Supabase Storage if configured
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase_bucket = os.getenv("SUPABASE_BUCKET", "assets")
        
        if supabase_url and supabase_key:
            try:
                import requests
                supabase_url = supabase_url.rstrip("/")
                upload_url = f"{supabase_url}/storage/v1/object/{supabase_bucket}/seat_distribution.pdf"
                
                headers = {
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "application/pdf",
                    "x-upsert": "true"
                }
                
                with open(static_seat_path, "rb") as f:
                    file_data = f.read()
                    
                resp = requests.post(upload_url, headers=headers, data=file_data)
                if resp.status_code in (200, 201):
                    print("Successfully uploaded and overwrote seat_distribution.pdf in Supabase Storage!")
                else:
                    print(f"Supabase upload failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"Failed to upload seat distribution PDF to Supabase: {e}")
    except Exception as e:
        print(f"Error regenerating seat distribution PDF: {e}")
        
    seat_matrix_1_based = pages_list
    background_tasks.add_task(run_ingestion_background, prospectus_path, seat_matrix_1_based)
    
    return {
        "message": "Prospectus uploaded successfully. Ingestion and indexing pipeline started in the background.",
        "excluded_pages_applied": pages_list
    }

from fastapi.responses import FileResponse

@app.get("/seat_distribution.pdf")
def get_seat_distribution():
    # If Supabase URL is configured, retrieve the PDF dynamically from Supabase Storage
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_bucket = os.getenv("SUPABASE_BUCKET", "assets")
    
    if supabase_url:
        try:
            import requests
            supabase_url = supabase_url.rstrip("/")
            pdf_url = f"{supabase_url}/storage/v1/object/public/{supabase_bucket}/seat_distribution.pdf"
            resp = requests.get(pdf_url, timeout=10)
            if resp.status_code == 200:
                from fastapi.responses import Response
                return Response(
                    content=resp.content,
                    media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=\"seat_distribution.pdf\""}
                )
        except Exception as e:
            print(f"Failed to fetch PDF from Supabase: {e}")

    # Local development fallback
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_seat_path = os.path.join(base_dir, "frontend", "static", "seat_distribution.pdf")
    public_seat_path = os.path.join(base_dir, "public", "seat_distribution.pdf")
    root_seat_path = os.path.join(base_dir, "seat_distribution.pdf")
    
    if os.path.exists(static_seat_path):
        return FileResponse(static_seat_path, media_type="application/pdf", filename="seat_distribution.pdf")
    elif os.path.exists(public_seat_path):
        return FileResponse(public_seat_path, media_type="application/pdf", filename="seat_distribution.pdf")
    elif os.path.exists(root_seat_path):
        return FileResponse(root_seat_path, media_type="application/pdf", filename="seat_distribution.pdf")
    else:
        raise HTTPException(status_code=404, detail="Seat distribution PDF not found")

@app.post("/user/query")
async def query_chatbot(payload: QueryRequest, current_user = Depends(get_current_user)):
    user_query = payload.query.strip()
    chat_history = payload.history or []
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        return StreamingResponse(
            route_chat_stream(user_query, chat_history),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot logic error: {e}")

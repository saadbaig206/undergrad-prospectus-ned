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
    await asyncio.sleep(3)
    try:
        from core.retrieval.pinecone_retriever import get_pinecone_index
        index = get_pinecone_index()
        dim = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        while True:
            try:
                await index.query(
                    vector=[0.0] * 384,
                    top_k=1
                )
            except Exception:
                pass
            await asyncio.sleep(10)  # Keepalive query every 10 seconds
    except Exception:
        pass

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
    except Exception as exc:
        print(f"Database initialization skipped or failed: {exc}")
        
    # Skip pre-warming on Vercel to prevent startup timeouts (under 10s limits)
    if os.environ.get("VERCEL"):
        print("Running on Vercel: skipping startup pre-warming to avoid timeouts.")
        return
        
    # Pre-warm AI and database clients to cut down first-query latency from 3s to under 1s
    try:
        from core.chatbot import get_llm
        from core.retrieval.pinecone_retriever import get_pinecone_index
        from core.embeddings import get_embeddings_model
        embed_model = get_embeddings_model()
        embed_model.embed_query("warmup query")
        get_llm()
        
        # Warm up the Pinecone connection pool by running a lightweight query (384 dim vector)
        try:
            from core.retrieval.bm25 import get_bm25_instance
            get_bm25_instance("undergraduate")
            get_bm25_instance("postgraduate")
            print("[BM25] Local instances loaded successfully!")
        except Exception as bm25_err:
            print(f"[BM25] Warming warning: {bm25_err}")

        try:
            index = get_pinecone_index()
            dim = int(os.getenv("EMBEDDING_DIMENSION", "384"))
            await index.query(
                vector=[0.0] * dim,
                top_k=1
            )
            print("[PINECONE] Connection warming completed successfully!")
        except Exception as pe:
            print(f"[PINECONE] Connection pool warming warning: {pe}")
        print("Pre-warming of Embeddings, Pinecone, and LLM clients completed successfully!")
    except Exception as e:
        print(f"Pre-warming warning: {e}")
        
    # Launch background keepalive loop to maintain a hot connection (only if not on Vercel)
    if not os.environ.get("VERCEL"):
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

async def run_ingestion_background(pdf_path: str, seat_matrix_pages: list, academic_level: str):
    from backend.database import update_ingestion_status
    try:
        from core.ingestion.ingestion_service import ingest_prospectus

        print(f"Background Ingestion Started for {academic_level}. Matrix pages: {seat_matrix_pages}")
        update_ingestion_status(academic_level, "processing")
        res = await ingest_prospectus(pdf_path, academic_level, 2026, "seat_distribution.pdf")
        if not res.success:
            raise Exception(res.message)
        print(f"Background Ingestion Completed Successfully for {academic_level}!")
        update_ingestion_status(academic_level, "completed")
    except Exception as e:
        print(f"Background Ingestion Failed for {academic_level}: {e}")
        update_ingestion_status(academic_level, "failed", str(e))

@app.post("/admin/upload-prospectus")
def upload_prospectus(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    excluded_pages: str = Form(""),
    academic_level: str = Form("undergraduate"),
    admin_user = Depends(get_current_admin)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    academic_level = academic_level.strip().lower()
    if academic_level not in ("undergraduate", "postgraduate"):
        raise HTTPException(status_code=400, detail="academic_level must be either 'undergraduate' or 'postgraduate'")

    prospectus_path = "UGProspectus.pdf" if academic_level == "undergraduate" else "PGProspectus.pdf"
    try:
        with open(prospectus_path, "wb") as buffer:
            buffer.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded PDF: {e}")
        
    pages_list = []
    if academic_level == "undergraduate":
        try:
            pages_list = [int(p.strip()) for p in excluded_pages.split(",") if p.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="excluded_pages must be a comma-separated list of integers")
            
    # Save the configuration to the PostgreSQL database instead of writing code to disk
    from backend.database import save_prospectus_metadata, update_ingestion_status
    try:
        save_prospectus_metadata(academic_level, pages_list)
        update_ingestion_status(academic_level, "processing")
        print(f"Saved {academic_level} excluded_pages: {pages_list} and set status to processing in PostgreSQL.")
    except Exception as e:
        print(f"Could not save prospectus metadata to database: {e}")

    # Generate Seat Distribution PDF only for undergraduate level
    if academic_level == "undergraduate" and pages_list:
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

    background_tasks.add_task(run_ingestion_background, prospectus_path, pages_list, academic_level)
    
    return {
        "message": f"{academic_level.capitalize()} prospectus uploaded successfully. Ingestion and indexing pipeline started in the background.",
        "excluded_pages_applied": pages_list
    }


@app.get("/admin/ingestion-status")
def get_admin_ingestion_status(academic_level: str = "undergraduate", admin_user = Depends(get_current_admin)):
    from backend.database import get_ingestion_status
    academic_level = academic_level.strip().lower()
    if academic_level not in ("undergraduate", "postgraduate"):
        raise HTTPException(status_code=400, detail="academic_level must be either 'undergraduate' or 'postgraduate'")
    
    return get_ingestion_status(academic_level)

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

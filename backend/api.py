import os
import secrets
import hashlib
import datetime
import asyncio
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.database import get_db_connection, init_db
from backend.schemas import AuthRequest, QueryRequest, CreateAdminRequest

from core.chatbot import is_seat_distribution_query, get_rag_response, SEAT_DIST_FILE_LINK
from core.main import run_parallel_pipeline
from core.admin_process import EXCLUDED_PAGES

app = FastAPI(title="UG Prospectus Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

# Password utility functions
def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()

# Authentication dependency
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
    
    cur.execute("SELECT id FROM users WHERE username = %s;", (payload.username.strip(),))
    if cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
        
    salt = secrets.token_hex(16)
    p_hash = hash_password(payload.password, salt)
    
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (%s, %s, %s, 'USER');",
            (payload.username.strip(), p_hash, salt)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Signup database error: {e}")
        
    cur.close()
    conn.close()
    return {"message": "User registered successfully"}

@app.post("/auth/login")
def login(payload: AuthRequest):
    from psycopg2.extras import RealDictCursor
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM users WHERE username = %s;", (payload.username.strip(),))
    user = cur.fetchone()
    
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
        conn.commit()
    except Exception as e:
        conn.rollback()
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
    
    cur.execute("SELECT id FROM users WHERE username = %s;", (payload.username.strip(),))
    if cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
        
    salt = secrets.token_hex(16)
    p_hash = hash_password(payload.password, salt)
    
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (%s, %s, %s, 'ADMIN');",
            (payload.username.strip(), p_hash, salt)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Admin creation database error: {e}")
        
    cur.close()
    conn.close()
    return {"message": f"Admin user '{payload.username}' created successfully"}

def run_ingestion_background(pdf_path: str, seat_matrix_pages: list):
    try:
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
        
    prospectus_path = "UGProspectus2025.pdf"
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
            if 0 <= p < doc.page_count:
                seat_doc.insert_pdf(doc, from_page=p, to_page=p)
                
        os.makedirs("frontend/static", exist_ok=True)
        static_seat_path = "frontend/static/seat_distribution.pdf"
        root_seat_path = "seat_distribution.pdf"
        
        seat_doc.save(static_seat_path)
        seat_doc.close()
        doc.close()
        
        import shutil
        shutil.copy(static_seat_path, root_seat_path)
        print("Regenerated seat_distribution.pdf with new page exclusions.")
    except Exception as e:
        print(f"Error regenerating seat distribution PDF: {e}")
        
    seat_matrix_1_based = [p + 1 for p in pages_list]
    background_tasks.add_task(run_ingestion_background, prospectus_path, seat_matrix_1_based)
    
    return {
        "message": "Prospectus uploaded successfully. Ingestion and indexing pipeline started in the background.",
        "excluded_pages_applied": pages_list
    }

from fastapi.responses import FileResponse

@app.get("/seat_distribution.pdf")
def get_seat_distribution():
    static_seat_path = "frontend/static/seat_distribution.pdf"
    root_seat_path = "seat_distribution.pdf"
    
    if os.path.exists(static_seat_path):
        return FileResponse(static_seat_path, media_type="application/pdf", filename="seat_distribution.pdf")
    elif os.path.exists(root_seat_path):
        return FileResponse(root_seat_path, media_type="application/pdf", filename="seat_distribution.pdf")
    else:
        raise HTTPException(status_code=404, detail="Seat distribution PDF not found")

@app.post("/user/query")
def query_chatbot(payload: QueryRequest, current_user = Depends(get_current_user)):
    user_query = payload.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        if is_seat_distribution_query(user_query):
            answer = f"For the complete and accurate Seat Distribution Matrix, please refer to the official document: [Seat Distribution PDF]({SEAT_DIST_FILE_LINK})."
            is_seat_query = True
        else:
            answer = get_rag_response(user_query)
            is_seat_query = False
            
        return {
            "answer": answer,
            "is_seat_query": is_seat_query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot logic error: {e}")

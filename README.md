# 🎓 University UG Prospectus RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) assistant for querying University Undergraduate Prospectuses. Built with a secure **FastAPI backend** (using **Neon serverless Postgres** for role-based access control), a premium **Streamlit glassmorphism frontend**, and an asynchronous parallel ingestion pipeline leveraging **LlamaParse**, **Pinecone**, **Model2Vec (minishlab/potion-base-8M)** local embeddings, and **Groq LLMs**.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Client [Streamlit Frontend]
        UI[Streamlit App]
    end

    subgraph Server [FastAPI Backend API]
        API[FastAPI App]
        Auth[Auth Middleware]
        Ingest[Background Ingestion Task]
    end

    subgraph Data [Data & AI Services]
        Neon[(Neon Postgres DB)]
        Pinecone[(Pinecone Vector DB)]
        Groq[Groq Llama-3.1 8B / 70B]
        Model2Vec[Model2Vec Potion-8M]
        LlamaParse[LlamaParse Cloud API]
    end

    %% Client-Server Requests
    UI -->|1. Signin / Query Request| API
    API --> Auth
    
    %% Auth check
    Auth -->|2. Verify Credentials & Session| Neon
    
    %% Query Flow
    API -->|3. Get Potion-8M Embeddings| Model2Vec
    API -->|4. Query Vector Embeddings| Pinecone
    API -->|5. Context-enrich Prompt| Groq
    Groq -->|6. Structured Answer| API
    API -->|7. JSON Response| UI

    %% Admin Ingestion Flow
    UI -->|Admin Upload PDF & Pages| API
    API -->|Trigger Ingestion| Ingest
    Ingest -->|Slice Excluded Pages| LlamaParse
    LlamaParse -->|Parsed Text Chunks| Ingest
    Ingest -->|Split into Overlapping Chunks| Ingest
    Ingest -->|Batch Embed via Model2Vec| Model2Vec
    Ingest -->|Upsert Chunks| Pinecone
```

---

## 📁 Directory Structure

```
workspace/
├── backend/
│   ├── __init__.py
│   ├── api.py           # FastAPI routes, auth middleware, and background workers
│   ├── database.py      # Neon Postgres connection, schema init, & connection validator
│   └── schemas.py       # Pydantic validation request/response objects
├── frontend/
│   ├── __init__.py
│   └── app.py           # Streamlit UI client with glassmorphism chat & Admin panels
├── core/
│   ├── __init__.py
│   ├── chatbot.py       # Core RAG querying engine (Groq model + Pinecone lookup)
│   ├── main.py          # Parallel page-by-page PDF extraction, text splitting, & vector indexing
│   └── admin_process.py # PDF processing operations, layout splits & OCR fallbacks
├── public/
│   └── seat_distribution.pdf  # Extracted seat matrix pages (served via Vercel Edge CDN)
├── api/
│   └── index.py         # Vercel Serverless Function entrypoint (with root path resolver)
├── vercel.json          # Vercel deployment routing & rewrite configurations
├── .env                 # API Keys and database configuration
├── requirements.txt     # Virtual environment dependencies
└── UGProspectus2025.pdf # Main prospectus source document
```

---

## 🛡️ User Types & Roles

The system supports two access levels configured dynamically:

| Role | Permissions | Available UI Panels |
| :--- | :--- | :--- |
| **USER** | Sign up, Log in, query RAG chatbot, download seat distribution PDF. | `💬 Chatbot Interface` |
| **ADMIN** | Sign up, Log in, query RAG chatbot, **create new Admin profiles**, **upload new prospectus PDF**, specify **excluded seat matrix page numbers** for ingestion. | `💬 Chatbot`, `📤 Ingest New Prospectus`, `🔑 Create New Admin` |

---

## ⚙️ Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration (Neon PostgreSQL)
DATABASE_URL="postgresql://<user>:<password>@<neon_host>/<dbname>?sslmode=require"

# Vector Database (Pinecone)
PINECONE_API_KEY="your-pinecone-api-key"

# Ingestion Processing (Llama Cloud)
LLAMA_CLOUD_API_KEY="your-llama-cloud-api-key"

# LLM Inference API (Groq)
GROQ_API_KEY="your-groq-api-key"

# External Cloud PDF Storage (Optional: Supabase)
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_KEY="your-supabase-service-role-key"
SUPABASE_BUCKET="assets"
SEAT_DIST_FILE_LINK="https://your-project-id.supabase.co/storage/v1/object/public/assets/seat_distribution.pdf"
```

---

## 🚀 Running the Application

Ensure you have activated your virtual environment before running the commands:

### 1. Start the FastAPI Backend
Start the backend server on port 8000. It will automatically connect to Neon Postgres, create the tables, and seed the default admin account:
```powershell
.\venv\Scripts\python.exe -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Streamlit Frontend Client
Start the Streamlit interface on port 8501:
```powershell
.\venv\Scripts\streamlit.exe run frontend/app.py --server.port 8501
```

### 3. Log In (Default Admin Credentials)
Access the UI in your browser at **[http://localhost:8501](http://localhost:8501)** and authenticate with:
*   **Username**: `admin`
*   **Password**: `admin123`

---

## 🔌 API Documentation

| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| **POST** | `/auth/signup` | Public | Register a new normal user account (`USER` role). |
| **POST** | `/auth/login` | Public | Validate credentials, create session, return token and role. |
| **POST** | `/admin/create-admin` | `ADMIN` Only | Create another user with `ADMIN` privileges. |
| **POST** | `/admin/upload-prospectus` | `ADMIN` Only | Upload a new prospectus, save locally, update page splits, and run background re-indexing. |
| **POST** | `/user/query` | Authenticated | Query RAG engine and retrieve structured answers from Pinecone index. |
| **GET** | `/seat_distribution.pdf` | Public | Download the compiled seat distribution PDF (served from Vercel Edge CDN in production). |

---

## 🌐 Vercel Backend Deployment

To deploy the FastAPI backend on Vercel:

### 1. Configure Vercel Project
Create a project on Vercel and link it to your GitHub repository.

### 2. Configure Environment Variables
In your Vercel project dashboard, navigate to **Settings > Environment Variables** and add the following keys:
- `DATABASE_URL`: Your Neon PostgreSQL connection string.
- `PINECONE_API_KEY`: Your Pinecone credentials.
- `GROQ_API_KEY`: Your Groq API key.
- `SUPABASE_URL`: (Optional) Your Supabase project URL (e.g. `https://peijeipftutzujpmzttd.supabase.co`).
- `SUPABASE_KEY`: (Optional) Your Supabase service role secret key.
- `SUPABASE_BUCKET`: (Optional) Your public storage bucket name (e.g. `assets`).
- `SEAT_DIST_FILE_LINK`: (Optional) The public link to serve the PDF to users (e.g. `https://peijeipftutzujpmzttd.supabase.co/storage/v1/object/public/assets/seat_distribution.pdf?download=`).

### 3. Deploy
Push your commits to your `main` branch on GitHub to trigger automatic Vercel builds:
```bash
git add .
git commit -m "Deploy to Vercel"
git push origin main
```
Once built, the API and PDF download link will be served globally at your Vercel deployment domain.

---

## ⚡ Key Features & Custom Routing

### 1. Robust PDF Delivery Pipeline (Supabase Storage)
Serving large binary files (like PDFs) from serverless containers on Vercel is highly limited due to read-only filesystems and ephemeral container lifetimes. 
*   **Programmatic Uploads**: The FastAPI backend includes a programmatic integration that uploads and overwrites `seat_distribution.pdf` in a **Supabase Storage** bucket with `x-upsert: true` whenever the admin updates the prospectus.
*   **Direct Cloud Downloads**: Adding `?download=` to the Supabase link forces the browser to download the file directly instead of opening a preview tab.
*   **Streaming Fallback**: The `/seat_distribution.pdf` backend endpoint dynamically streams the file from Supabase in production and falls back to local disk in development.

### 2. Intelligent Administrative Query Routing
University deans oversee whole **Faculties** rather than individual **Departments**. The RAG system handles this using a custom context routing layout:
*   **Hierarchy Mapping**: When queries about department-level deans (e.g., "Who is the dean of Software Engineering?") are received, the LLM maps the department to its parent Faculty (e.g., ECE) and correctly identifies the respective Dean (e.g., **Prof. Dr. Saad Ahmed Qazi**).
*   **Direct Answer Context**: The system instructions guide the LLM to provide direct, detailed answers based on organizational hierarchy instead of returning general refusal fallbacks.

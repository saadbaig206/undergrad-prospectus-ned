import os
import secrets
import hashlib
import psycopg2

from psycopg2.pool import ThreadedConnectionPool

DATABASE_URL = os.environ.get("DATABASE_URL")
_pool = None

def init_pool():
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL is not set in the environment variables.")
        # Min 2, Max 15 pooled connections
        _pool = ThreadedConnectionPool(2, 15, DATABASE_URL)

class PooledConnectionWrapper:
    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool
        self._pooled_returned = False

    def close(self):
        if not self._pooled_returned:
            try:
                self._pool.putconn(self._conn)
                self._pooled_returned = True
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass

    def __getattr__(self, name):
        return getattr(self._conn, name)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def get_db_connection():
    init_pool()
    max_retries = 5
    for _ in range(max_retries):
        conn = _pool.getconn()
        
        is_alive = False
        try:
            conn.rollback()
            conn.autocommit = True
            # Run a quick check query to verify the connection is healthy
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            is_alive = True
        except Exception:
            pass
            
        if is_alive and conn.closed == 0:
            return PooledConnectionWrapper(conn, _pool)
        else:
            try:
                _pool.putconn(conn, close=True)
            except Exception:
                pass
                
    raise RuntimeError("Failed to obtain a healthy database connection from the pool.")

def init_db():
    # Initialize the database tables
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role VARCHAR(20) NOT NULL CHECK (role IN ('ADMIN', 'USER'))
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        expires_at TIMESTAMP NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prospectus_metadata (
        academic_level VARCHAR(50) PRIMARY KEY,
        excluded_pages INT[] NOT NULL DEFAULT '{}'::INT[],
        ingestion_status VARCHAR(50) DEFAULT 'idle',
        ingestion_error TEXT
    );
    """)
    try:
        cur.execute("ALTER TABLE prospectus_metadata ADD COLUMN IF NOT EXISTS ingestion_status VARCHAR(50) DEFAULT 'idle';")
        cur.execute("ALTER TABLE prospectus_metadata ADD COLUMN IF NOT EXISTS ingestion_error TEXT;")
    except Exception as e:
        print(f"Failed to alter prospectus_metadata table: {e}")
    
    # Check if there is an admin seeded, if not seed a default
    cur.execute("SELECT COUNT(*) FROM users WHERE role = 'ADMIN';")
    admin_count = cur.fetchone()[0]
    if admin_count == 0:
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            "admin123".encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        cur.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES ('admin', %s, %s, 'ADMIN');",
            (password_hash, salt)
        )
        print("Default admin seeded successfully (admin/admin123)")
    conn.commit()
    cur.close()
    conn.close()

def save_prospectus_metadata(level: str, excluded_pages: list[int]):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO prospectus_metadata (academic_level, excluded_pages)
            VALUES (%s, %s)
            ON CONFLICT (academic_level) DO UPDATE SET excluded_pages = EXCLUDED.excluded_pages;
            """,
            (level, excluded_pages)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def get_prospectus_metadata(level: str) -> list[int]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT excluded_pages FROM prospectus_metadata WHERE academic_level = %s;", (level,))
        row = cur.fetchone()
        if row:
            return list(row[0])
        return []
    finally:
        cur.close()
        conn.close()

def update_ingestion_status(level: str, status: str, error_msg: str = None):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO prospectus_metadata (academic_level, ingestion_status, ingestion_error)
            VALUES (%s, %s, %s)
            ON CONFLICT (academic_level) DO UPDATE SET 
                ingestion_status = EXCLUDED.ingestion_status,
                ingestion_error = EXCLUDED.ingestion_error;
            """,
            (level, status, error_msg)
        )
        conn.commit()
    except Exception as e:
        print(f"Error updating ingestion status for {level} to {status}: {e}")
    finally:
        cur.close()
        conn.close()

def get_ingestion_status(level: str) -> dict:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT ingestion_status, ingestion_error FROM prospectus_metadata WHERE academic_level = %s;",
            (level,)
        )
        row = cur.fetchone()
        if row:
            return {"status": row[0] or "idle", "error": row[1]}
        return {"status": "idle", "error": None}
    except Exception as e:
        print(f"Error getting ingestion status for {level}: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        cur.close()
        conn.close()

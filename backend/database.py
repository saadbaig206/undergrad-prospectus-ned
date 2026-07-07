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

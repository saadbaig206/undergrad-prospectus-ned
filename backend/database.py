import os
import secrets
import hashlib
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in the environment variables.")
    return psycopg2.connect(DATABASE_URL)

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

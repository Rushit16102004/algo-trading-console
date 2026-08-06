# PASSKEY: rushit2712
import os
import sqlite3
import hashlib
from cryptography.fernet import Fernet

DB_PATH = "users.db"

# Static Fernet key for symmetric encryption of API keys and credentials
# This allows background threads/workers to decrypt credentials during auto-restarts.
SECRET_KEY = b'NGI13jlaczp6MbF9XE9eNzx0rBCvP06yxMrnk5lWUac='
fernet = Fernet(SECRET_KEY)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            pin_hash TEXT NOT NULL,
            enc_api_key TEXT,
            enc_client_id TEXT,
            enc_password TEXT,
            enc_totp_secret TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def encrypt_value(value: str) -> str:
    if not value:
        return ""
    return fernet.encrypt(value.encode('utf-8')).decode('utf-8')

def decrypt_value(encrypted_value: str) -> str:
    if not encrypted_value:
        return ""
    try:
        return fernet.decrypt(encrypted_value.encode('utf-8')).decode('utf-8')
    except Exception:
        return ""

def hash_pin(pin: str, salt: str) -> str:
    # Use SHA-256 with the user's email as salt
    salted = pin + salt
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()

def register_user(email: str, pin: str, api_key: str, client_id: str, password: str, totp_secret: str) -> bool:
    init_db()
    email_clean = email.strip().lower()
    hashed = hash_pin(pin, email_clean)
    
    enc_api = encrypt_value(api_key.strip())
    enc_client = encrypt_value(client_id.strip())
    enc_pass = encrypt_value(password.strip())
    enc_totp = encrypt_value(totp_secret.strip())
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (email, pin_hash, enc_api_key, enc_client_id, enc_password, enc_totp_secret)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email_clean, hashed, enc_api, enc_client, enc_pass, enc_totp))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def verify_user(email: str, pin: str) -> dict:
    init_db()
    email_clean = email.strip().lower()
    hashed = hash_pin(pin, email_clean)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, email, enc_api_key, enc_client_id, enc_password, enc_totp_secret
        FROM users
        WHERE email = ? AND pin_hash = ?
    """, (email_clean, hashed))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "email": row[1],
            "api_key": decrypt_value(row[2]),
            "client_id": decrypt_value(row[3]),
            "password": decrypt_value(row[4]),
            "totp_secret": decrypt_value(row[5])
        }
    return None

def get_user_by_email(email: str) -> dict:
    init_db()
    email_clean = email.strip().lower()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, email, enc_api_key, enc_client_id, enc_password, enc_totp_secret
        FROM users
        WHERE email = ?
    """, (email_clean,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "email": row[1],
            "api_key": decrypt_value(row[2]),
            "client_id": decrypt_value(row[3]),
            "password": decrypt_value(row[4]),
            "totp_secret": decrypt_value(row[5])
        }
    return None

def get_all_users() -> list:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, enc_api_key, enc_client_id, enc_password, enc_totp_secret FROM users")
    rows = cursor.fetchall()
    conn.close()
    
    users = []
    for r in rows:
        users.append({
            "id": r[0],
            "email": r[1],
            "api_key": decrypt_value(r[2]),
            "client_id": decrypt_value(r[3]),
            "password": decrypt_value(r[4]),
            "totp_secret": decrypt_value(r[5])
        })
    return users

# Ensure database is initialized upon import
init_db()

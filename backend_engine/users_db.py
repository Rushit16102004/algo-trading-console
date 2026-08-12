# PASSKEY: rushit2712
import os
from sqlalchemy.orm import Session
from backend_engine.database import SessionLocal, engine, Base
from backend_engine.models import User, BrokerAccount
from backend_engine.auth import hash_password, verify_password
from backend_engine.credentials_encryptor import encryptor

def init_db():
    # If using local SQLite users.db, check if it's the old schema
    from backend_engine.database import db_url
    if db_url.startswith("sqlite"):
        # Resolve path
        db_path = db_url.replace("sqlite:///", "")
        if os.path.exists(db_path):
            import sqlite3
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(users)")
                cols = [c[1] for c in cursor.fetchall()]
                conn.close()
                if cols and "pin_hash" in cols and "password_hash" not in cols:
                    print("[Migration] Detected legacy SQLite database. Resetting schema for secure password hashing...")
                    # Dispose engine connections to unlock the file
                    engine.dispose()
                    # Remove database files
                    for suffix in ["", "-wal", "-shm"]:
                        p = db_path + suffix
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception as ex:
                                print(f"[Migration] Could not remove {p}: {ex}")
            except Exception as e:
                print(f"[Migration] Error checking legacy SQLite: {e}")
                
    # Create all schemas
    Base.metadata.create_all(bind=engine)

def register_user(email: str, pin: str, api_key: str, client_id: str, password: str, totp_secret: str) -> bool:
    init_db()
    email_clean = email.strip().lower()
    
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email_clean).first()
        if existing:
            return False
            
        hashed_pin = hash_password(pin)
        user = User(
            email=email_clean,
            password_hash=hashed_pin,
            role="ADMIN" if email_clean == "developer@gmail.com" else "USER"
        )
        db.add(user)
        db.flush()
        
        enc_api = encryptor.encrypt(api_key.strip())
        enc_pass = encryptor.encrypt(password.strip())
        enc_totp = encryptor.encrypt(totp_secret.strip())
        
        account = BrokerAccount(
            user_id=user.id,
            client_id=client_id.strip(),
            encrypted_api_key=enc_api,
            encrypted_password=enc_pass,
            encrypted_totp_secret=enc_totp
        )
        db.add(account)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error registering user: {e}")
        return False
    finally:
        db.close()

def verify_user(email: str, pin: str) -> dict:
    init_db()
    email_clean = email.strip().lower()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email_clean).first()
        if not user:
            return None
            
        if not verify_password(pin, user.password_hash):
            return None
            
        account = db.query(BrokerAccount).filter(
            BrokerAccount.user_id == user.id,
            BrokerAccount.is_active == True
        ).first()
        
        api_key = ""
        client_id = ""
        password = ""
        totp_secret = ""
        if account:
            api_key = encryptor.decrypt(account.encrypted_api_key)
            client_id = account.client_id
            password = encryptor.decrypt(account.encrypted_password)
            totp_secret = encryptor.decrypt(account.encrypted_totp_secret)
            
        return {
            "id": user.id,
            "email": user.email,
            "api_key": api_key,
            "client_id": client_id,
            "password": password,
            "totp_secret": totp_secret
        }
    finally:
        db.close()

def get_user_by_email(email: str) -> dict:
    init_db()
    email_clean = email.strip().lower()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email_clean).first()
        if not user:
            return None
            
        account = db.query(BrokerAccount).filter(
            BrokerAccount.user_id == user.id,
            BrokerAccount.is_active == True
        ).first()
        
        api_key = ""
        client_id = ""
        password = ""
        totp_secret = ""
        if account:
            api_key = encryptor.decrypt(account.encrypted_api_key)
            client_id = account.client_id
            password = encryptor.decrypt(account.encrypted_password)
            totp_secret = encryptor.decrypt(account.encrypted_totp_secret)
            
        return {
            "id": user.id,
            "email": user.email,
            "api_key": api_key,
            "client_id": client_id,
            "password": password,
            "totp_secret": totp_secret
        }
    finally:
        db.close()

def get_all_users() -> list:
    init_db()
    
    db = SessionLocal()
    try:
        users = db.query(User).all()
        result = []
        for user in users:
            account = db.query(BrokerAccount).filter(
                BrokerAccount.user_id == user.id,
                BrokerAccount.is_active == True
            ).first()
            
            api_key = ""
            client_id = ""
            password = ""
            totp_secret = ""
            if account:
                api_key = encryptor.decrypt(account.encrypted_api_key)
                client_id = account.client_id
                password = encryptor.decrypt(account.encrypted_password)
                totp_secret = encryptor.decrypt(account.encrypted_totp_secret)
                
            result.append({
                "id": user.id,
                "email": user.email,
                "api_key": api_key,
                "client_id": client_id,
                "password": password,
                "totp_secret": totp_secret
            })
        return result
    finally:
        db.close()

def check_any_custom_user_exists() -> bool:
    init_db()
    db = SessionLocal()
    try:
        count = db.query(User).filter(
            User.email != "developer@gmail.com",
            User.email != "demo@gmail.com"
        ).count()
        return count > 0
    finally:
        db.close()

# Ensure tables exist
init_db()

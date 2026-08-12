import datetime
from sqlalchemy.orm import Session
from backend_engine.database import SessionLocal
from backend_engine.models import AuditLog

def log_audit_event(user_id: int, event: str, details: str = None, request_id: str = None):
    """
    Creates a structured audit log entry in the database.
    Prevents printing or logging sensitive security parameters.
    """
    # Filter out sensitive fields
    if details:
        for secret_word in ("password", "pin", "totp", "secret", "api_key", "jwt", "token"):
            if secret_word in details.lower():
                details = "[REDACTED SENSITIVE DATA]"
                break
                
    db = SessionLocal()
    try:
        log = AuditLog(
            user_id=user_id,
            event=event,
            details=details,
            request_id=request_id,
            timestamp=datetime.datetime.utcnow()
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"[AuditLogger] Error writing audit log: {e}")
    finally:
        db.close()

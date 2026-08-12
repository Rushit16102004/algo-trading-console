import datetime
import pytz
from backend_engine.config import FORCE_EXIT_HOUR, FORCE_EXIT_MINUTE

def get_ist_time() -> datetime.datetime:
    """Returns the current date and time in Asia/Kolkata timezone."""
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.datetime.now(ist)

def is_market_open() -> bool:
    """Returns True if the Nifty market is currently open for trading (excluding weekends)."""
    now = get_ist_time()
    
    # 0 = Monday, 5 = Saturday, 6 = Sunday
    if now.weekday() >= 5:
        return False
        
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return start_time <= now <= end_time

def is_exit_only_zone() -> bool:
    """Returns True if we are past the cutoff time (typically 15:10 IST) where only exits are allowed."""
    now = get_ist_time()
    if now.weekday() >= 5:
        return True
        
    cutoff_time = now.replace(hour=FORCE_EXIT_HOUR, minute=FORCE_EXIT_MINUTE, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return cutoff_time <= now <= end_time

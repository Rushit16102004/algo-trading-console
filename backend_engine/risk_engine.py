import datetime
from sqlalchemy.orm import Session
from backend_engine.database import SessionLocal
from backend_engine.models import PaperTrade, AuditLog
from backend_engine.kill_switch import get_kill_switch_state
from backend_engine.market_session import is_market_open, is_exit_only_zone
from backend_engine.health_monitor import monitor as health_monitor
from backend_engine.position_manager import position_manager
from backend_engine.config import (
    DEMO_MODE, TRADING_MODE, MAX_DAILY_LOSS, 
    MAX_TRADES_PER_DAY, MAX_CONCURRENT_POSITIONS
)

# Unique signal duplicate memory cache
processed_signal_hashes = set()

class RiskEngine:
    def __init__(self):
        pass
        
    def can_execute_trade(self, user_id: int, strategy: str, symbol: str, candle_timestamp: str, signal_type: str) -> dict:
        """
        Runs centralized security and risk validations on every signal before routing it to order execution.
        Returns a dict e.g. {"allowed": True} or {"allowed": False, "reason": "KILL_SWITCH_ACTIVE"}
        """
        # 1. Kill Switch
        if get_kill_switch_state():
            return {"allowed": False, "reason": "KILL_SWITCH_ACTIVE"}
            
        # 2. Market Hours Safety Check
        if not is_market_open():
            return {"allowed": False, "reason": "MARKET_CLOSED"}
            
        if is_exit_only_zone() and signal_type in ("BUY", "LONG", "SELL", "SHORT"):
            if signal_type not in ("EXIT", "CLOSE"):
                return {"allowed": False, "reason": "EXIT_ONLY_ZONE_ACTIVE"}
                
        # 3. Market Data Health Check
        if health_monitor.is_feed_stale():
            return {"allowed": False, "reason": "MARKET_DATA_STALE"}
            
        # 4. Position Manager Reconciliation block
        if position_manager.mismatch_active:
            return {"allowed": False, "reason": "POSITION_MISMATCH_ACTIVE"}
            
        # 5. Duplicate Signal Protection
        signal_hash = hash(f"{strategy}:{symbol}:{candle_timestamp}:{signal_type}")
        if signal_hash in processed_signal_hashes:
            return {"allowed": False, "reason": "DUPLICATE_SIGNAL"}
            
        # Initialize Database connection for transactional limits
        db = SessionLocal()
        try:
            # 8. Maximum Concurrent Positions Check
            open_positions = db.query(PaperTrade).filter(
                PaperTrade.user_id == user_id,
                PaperTrade.status == "OPEN"
            ).count()
            
            if open_positions >= MAX_CONCURRENT_POSITIONS and signal_type not in ("EXIT", "CLOSE"):
                log = AuditLog(
                    user_id=user_id,
                    event="RISK_LIMIT_BREACH",
                    details=f"Trade blocked due to Maximum Concurrent Positions. Active: {open_positions}"
                )
                db.add(log)
                db.commit()
                return {"allowed": False, "reason": "MAX_CONCURRENT_POSITIONS_EXCEEDED"}
                
            # Cache the hash since it is a valid unique signal passing all checks
            processed_signal_hashes.add(signal_hash)
            
            log = AuditLog(
                user_id=user_id,
                event="SIGNAL_RISK_APPROVED",
                details=f"Strategy: {strategy} | Symbol: {symbol} | Signal: {signal_type}"
            )
            db.add(log)
            db.commit()
            
            return {"allowed": True}
            
        except Exception as e:
            print(f"[RiskEngine] Error verifying trade logic: {e}")
            return {"allowed": False, "reason": "SYSTEM_EXCEPTION"}
        finally:
            db.close()

risk_engine = RiskEngine()

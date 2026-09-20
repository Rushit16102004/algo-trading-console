import time
from sqlalchemy.orm import Session
from backend_engine.database import SessionLocal
from backend_engine.models import PaperTrade, AuditLog

class PositionManager:
    def __init__(self):
        self.mismatch_active = False
        self.last_check_time = 0.0
        
    def reconcile_positions(self, user_id: int, broker_positions: list):
        """
        Compares local database position state with actual broker/simulator state.
        If a mismatch is found, it raises an alert and blocks trading.
        """
        db = SessionLocal()
        try:
            # Query open positions in local DB
            local_open_trades = db.query(PaperTrade).filter(
                PaperTrade.user_id == user_id,
                PaperTrade.status == "OPEN"
            ).all()
            
            # Map BUY as +1, SELL as -1
            local_qty = sum(t.quantity for t in local_open_trades if t.side in ("BUY", "LONG")) - \
                        sum(t.quantity for t in local_open_trades if t.side in ("SELL", "SHORT"))
                        
            # Query broker position for NIFTY
            broker_qty = 0
            for pos in broker_positions:
                if "NIFTY" in pos.get("symbol", "").upper():
                    net_qty = int(pos.get("net_qty", 0))
                    broker_qty += net_qty
                    
            if local_qty != broker_qty:
                self.mismatch_active = True
                log = AuditLog(
                    user_id=user_id,
                    event="POSITION_MISMATCH",
                    details=f"Position Mismatch! Local Net Qty: {local_qty} | Broker Net Qty: {broker_qty}"
                )
                db.add(log)
                db.commit()
                print(f"[PositionManager] POSITION MISMATCH DETECTED: Local {local_qty} vs Broker {broker_qty}")
            else:
                if self.mismatch_active:
                    print("[PositionManager] Position reconciliation successful. Mismatch resolved.")
                    log = AuditLog(
                        user_id=user_id,
                        event="POSITION_RECONCILED",
                        details="Position mismatch resolved cleanly."
                    )
                    db.add(log)
                    db.commit()
                self.mismatch_active = False
                
        except Exception as e:
            print(f"[PositionManager] Error during reconciliation: {e}")
        finally:
            db.close()

position_manager = PositionManager()

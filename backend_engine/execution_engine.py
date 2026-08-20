import datetime
from sqlalchemy.orm import Session
from backend_engine.database import SessionLocal
from backend_engine.models import PaperTrade, AuditLog
from backend_engine.risk_engine import risk_engine
from backend_engine.config import TRADING_MODE

class ExecutionEngine:
    def __init__(self):
        pass
        
    def execute_order(self, user_session, strategy: str, symbol: str, side: str, quantity: int, price: float, current_time, entry_reason: str, hmm_regime=None):
        """
        Receives order request, validates it through the Central Risk Engine, 
        and routes it to the appropriate simulator/broker based on active mode.
        """
        user_id = user_session.user_id
        
        # 1. Validate through Risk Engine
        candle_ts = str(current_time)
        risk_res = risk_engine.can_execute_trade(
            user_id=user_id,
            strategy=strategy,
            symbol=symbol,
            candle_timestamp=candle_ts,
            signal_type=side
        )
        
        if not risk_res.get("allowed", False):
            reason = risk_res.get("reason", "RISK_REJECTED")
            print(f"[ExecutionEngine] Trade BLOCKED by Risk Engine: {reason}")
            db = SessionLocal()
            try:
                log = AuditLog(
                    user_id=user_id,
                    event="ORDER_REJECTED",
                    details=f"Strategy: {strategy} | Symbol: {symbol} | Side: {side} | Reason: {reason}"
                )
                db.add(log)
                db.commit()
            except Exception as e:
                print(f"Error logging rejected trade: {e}")
            finally:
                db.close()
            return False, reason
            
        # 2. Log accepted order event
        db = SessionLocal()
        try:
            log = AuditLog(
                user_id=user_id,
                event="ORDER_ACCEPTED",
                details=f"Strategy: {strategy} | Symbol: {symbol} | Side: {side} | Qty: {quantity} | Price: {price}"
            )
            db.add(log)
            db.commit()
        except Exception:
            pass
        finally:
            db.close()
            
        if TRADING_MODE == "PAPER":
            # Route to Paper Simulator
            if side in ("BUY", "LONG", "SELL", "SHORT"):
                # Determine if we exit matching
                exit_pos_list = [p for p in user_session.paper_trade_engine.active_positions if p["position_type"] != side]
                if exit_pos_list and side in ("EXIT", "CLOSE"):
                    for pos in list(user_session.paper_trade_engine.active_positions):
                        user_session.paper_trade_engine.exit_position(
                            pos=pos,
                            exit_reason=entry_reason,
                            nifty_exit_price=price,
                            current_time=current_time
                        )
                else:
                    user_session.paper_trade_engine.enter_position(
                        pos_type=side,
                        nifty_close=price,
                        current_time=current_time,
                        entry_reason=entry_reason,
                        hmm_regime=hmm_regime
                    )
            return True, "EXECUTED_SIMULATOR"
        elif TRADING_MODE == "LIVE":
            if not user_session.smart_connect:
                return False, "BROKER_NOT_CONNECTED"
                
            try:
                tx_type = "BUY" if side in ("BUY", "LONG") else "SELL"
                token = user_session.future_token or "58072"
                
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": "NIFTY",
                    "symboltoken": token,
                    "transactiontype": tx_type,
                    "exchange": "NFO",
                    "ordertype": "MARKET",
                    "producttype": "CARRYOVER",
                    "duration": "DAY",
                    "qty": str(quantity)
                }
                
                response = user_session.smart_connect.placeOrder(order_params)
                print(f"[ExecutionEngine] LIVE ORDER PLACED: {response}")
                
                db = SessionLocal()
                try:
                    log = AuditLog(
                        user_id=user_id,
                        event="LIVE_ORDER_PLACED",
                        details=f"Token: {token} | Side: {tx_type} | Qty: {quantity} | Response: {response}"
                    )
                    db.add(log)
                    db.commit()
                except Exception:
                    pass
                finally:
                    db.close()
                    
                return True, "EXECUTED_LIVE"
            except Exception as e:
                print(f"[ExecutionEngine] Live order execution exception: {e}")
                return False, f"BROKER_EXECUTION_ERROR: {e}"
        else:
            return False, "UNKNOWN_TRADING_MODE"

execution_engine = ExecutionEngine()

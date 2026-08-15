# PASSKEY: rushit2712
import json
import os
import pandas as pd
from datetime import datetime
from backend_engine.config import ACTIVE_POSITIONS_PATH, INITIAL_CAPITAL, FIXED_SL
from src.risk_manager import ThermalDissipationSizer

class PaperTradeEngine:
    def __init__(self, ws_handler=None, option_ltp_cache=None, trade_logger=None, state_path=None, user_id=None):
        self.ws_handler = ws_handler
        self.option_ltp_cache = option_ltp_cache if option_ltp_cache is not None else {}
        self.capital = INITIAL_CAPITAL
        self.active_positions = []
        self.trades = []
        self.trade_logger = trade_logger
        self.state_path = state_path if state_path is not None else ACTIVE_POSITIONS_PATH
        self.sizer = ThermalDissipationSizer(base_qty=65) # NIFTY standard lot size is 65
        self.user_id = user_id
        self.load_state()
    def log(self, message):
        """Helper to write to trade logger if available, otherwise print."""
        if self.trade_logger:
            self.trade_logger.log_activity(message)
        else:
            print(message)

    def load_state(self):
        """Loads capital and active positions from user's state JSON if it exists."""
        self.last_signal = "None"
        self.last_was_sl = False
        self.active_positions = []
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
                    if state:
                        self.capital = state.get("capital", INITIAL_CAPITAL)
                        self.active_positions = state.get("active_positions", [])
                        self.trades = state.get("trades", [])
                        self.last_signal = state.get("last_signal", "None")
                        self.last_was_sl = state.get("last_was_sl", False)
                        self.sizer.T = state.get("sizer_T", 1.0)
                        self.log(f"[PaperTradeEngine] State loaded from {self.state_path}. Positions Count: {len(self.active_positions)} | Capital: {self.capital:.2f}")
            except Exception as e:
                self.log(f"[PaperTradeEngine] Error loading state from {self.state_path}: {e}. Starting fresh.")

    def save_state(self):
        """Saves current state to user's state JSON."""
        state = {
            "capital": self.capital,
            "active_positions": self.active_positions,
            "trades": self.trades,
            "last_signal": self.last_signal,
            "last_was_sl": self.last_was_sl,
            "sizer_T": self.sizer.T
        }
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            self.log(f"[PaperTradeEngine] Error saving state: {e}")

    def enter_position(self, pos_type, nifty_close, current_time, entry_reason, hmm_regime=None, override_option_price=None):
        """
        Enters a new direct Nifty Spot position.
        pos_type: 'LONG' or 'SHORT'
        """
        # Calculate SL in Nifty spot points
        sl_nifty = nifty_close - FIXED_SL if pos_type == 'LONG' else nifty_close + FIXED_SL
        
        # Determine TP points based on regime or default to 150 points
        if hmm_regime in ['markup', 'markdown', 'expansionup', 'expansiondown']:
            tp_points = 200.0
        else:
            tp_points = 150.0
        
        tp_nifty = nifty_close + tp_points if pos_type == 'LONG' else nifty_close - tp_points
        
        # Determine current pyramid count in the same direction
        same_dir_count = sum(1 for p in self.active_positions if p["position_type"] == pos_type)
        
        # Check lot size
        dt_obj = pd.to_datetime(current_time, format='mixed')
        lots = self.sizer.get_multiplier(dt_obj, same_dir_count)
        
        new_pos = {
            "position_type": pos_type,
            "entry_nifty_price": float(nifty_close),
            "sl_nifty_price": float(sl_nifty),
            "tp_nifty_price": float(tp_nifty),
            "entry_time": str(current_time),
            "option_token": 0,
            "option_tsym": "NIFTY50",
            "entry_option_price": 0.0,
            "option_lots": int(lots),
            "lot_size": 65,
            "entry_reason": entry_reason,
            "high_since_entry": float(nifty_close),
            "low_since_entry": float(nifty_close)
        }
        
        self.active_positions.append(new_pos)
        self.last_signal = "Long" if pos_type == "LONG" else "Short"
        self.log(f"[PaperTradeEngine] NIFTY ENTRY {pos_type} | Nifty: {nifty_close:.2f} | SL: {sl_nifty:.2f} | TP: {tp_nifty:.2f} | Lots: {lots}")
        self.save_state()

        if self.user_id:
            try:
                from backend_engine.database import SessionLocal
                from backend_engine.models import PaperTrade
                db = SessionLocal()
                db_pos = PaperTrade(
                    user_id=self.user_id,
                    strategy="243A" if "consensus" in entry_reason.lower() else "LONGPING",
                    symbol="NIFTY",
                    side=pos_type,
                    quantity=int(lots),
                    entry_price=float(nifty_close),
                    entry_time=datetime.now(),
                    status="OPEN"
                )
                db.add(db_pos)
                db.commit()
                db.close()
            except Exception as e:
                self.log(f"[PaperTradeEngine] Database insertion error in enter_position: {e}")

        return True

    def exit_position(self, pos, exit_reason, nifty_exit_price, current_time, signal=0):
        """
        Exits a specific active direct Nifty Spot position.
        """
        if pos not in self.active_positions:
            return None
            
        pos_type = pos["position_type"]
        entry_nifty = pos["entry_nifty_price"]
        sl_nifty = pos["sl_nifty_price"]
        lot_size = pos.get("lot_size", 65)
        lots = pos.get("option_lots", 1)
        qty = lot_size * lots
        
        # Nifty PnL = (exit - entry) if LONG else (entry - exit)
        nifty_pnl_points = (nifty_exit_price - entry_nifty) if pos_type == "LONG" else (entry_nifty - nifty_exit_price)
        pnl = nifty_pnl_points * qty
        
        self.capital += pnl
        self.last_was_sl = (exit_reason in ("SL", "FIXED_SL"))
        
        trade_record = {
            "strategy": getattr(self, "strategy_name", "243A"),
            "signal": "BUY" if pos_type == "LONG" else "SELL",
            "entry_time": pos["entry_time"],
            "exit_time": str(current_time),
            "entry_nifty": round(entry_nifty, 2),
            "exit_nifty": round(nifty_exit_price, 2),
            "entry_option": 0.0,
            "exit_option": 0.0,
            "entry_reason": pos.get("entry_reason", "Consensus signal"),
            "sl_price": round(sl_nifty, 2),
            "exit_reason": exit_reason,
            "option_selected": "NIFTY50",
            "option_lots": int(lots),
            "pnl": round(pnl, 2),
            "net pnl": round(pnl - 40.0, 2)  # Reduced flat fee since no options
        }
        
        self.trades.append(trade_record)
        self.log(f"[PaperTradeEngine] NIFTY EXIT {pos_type} ({exit_reason}) | Entry: {entry_nifty:.2f} | Exit: {nifty_exit_price:.2f} | PnL: Rs. {pnl:.2f} | Capital: Rs. {self.capital:.2f}")
        
        self.active_positions.remove(pos)
        self.sizer.record_outcome(nifty_pnl_points)
        self.save_state()

        if self.user_id:
            try:
                from backend_engine.database import SessionLocal
                from backend_engine.models import PaperTrade
                db = SessionLocal()
                db_pos = db.query(PaperTrade).filter(
                    PaperTrade.user_id == self.user_id,
                    PaperTrade.symbol == "NIFTY",
                    PaperTrade.side == pos_type,
                    PaperTrade.status == "OPEN"
                ).order_by(PaperTrade.id.asc()).first()
                if db_pos:
                    db_pos.exit_price = float(nifty_exit_price)
                    db_pos.exit_time = datetime.now()
                    db_pos.pnl = float(pnl)
                    db_pos.status = "CLOSED"
                    db.commit()
                db.close()
            except Exception as e:
                self.log(f"[PaperTradeEngine] Database update error in exit_position: {e}")
        
        # Log via TradeLogger
        if self.trade_logger:
            self.trade_logger.log_completed_trade(trade_record)
            self.trade_logger.update_daily_pnl(str(current_time), self.trades)
            
        # Reversing Logic (only before 3:10 PM)
        timestamp = pd.to_datetime(current_time, format='mixed')
        current_hour = timestamp.hour
        current_minute = timestamp.minute
        
        should_reverse = False
        if pos_type == "LONG" and signal == -1:
            should_reverse = True
        elif pos_type == "SHORT" and signal == 1:
            should_reverse = True
            
        if should_reverse and not (current_hour == 15 and current_minute >= 10):
            reverse_pos_type = "SHORT" if pos_type == "LONG" else "LONG"
            self.log(f"[PaperTradeEngine] 🔄 REVERSE SIGNAL triggered. Reopening in {reverse_pos_type} position...")
            return reverse_pos_type
            
        return None

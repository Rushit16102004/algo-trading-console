# PASSKEY: rushit2712
import os
import csv
import datetime
import pandas as pd
from backend_engine.config import (
    SYSTEM_LOG_PATH,
    SIGNAL_LOG_PATH,
    TRADE_LOG_PATH,
    POSITION_LOG_PATH,
    DAILY_PNL_PATH
)

class TradeLogger:
    def __init__(self, user_dir=None):
        if user_dir:
            self.system_log_path = os.path.join(user_dir, "logs_dryrun", "system.log")
            self.signal_log_path = os.path.join(user_dir, "logs_dryrun", "signal_log.csv")
            self.trade_log_path = os.path.join(user_dir, "logs_dryrun", "trade_log.csv")
            self.position_log_path = os.path.join(user_dir, "logs_dryrun", "position_log.csv")
            self.daily_pnl_path = os.path.join(user_dir, "logs_dryrun", "daily_pnl.csv")
        else:
            self.system_log_path = SYSTEM_LOG_PATH
            self.signal_log_path = SIGNAL_LOG_PATH
            self.trade_log_path = TRADE_LOG_PATH
            self.position_log_path = POSITION_LOG_PATH
            self.daily_pnl_path = DAILY_PNL_PATH
            
        os.makedirs(os.path.dirname(self.system_log_path), exist_ok=True)
        self.setup_csv_files()

    def setup_csv_files(self):
        """Initializes all CSV log files with the user's requested headers if they don't exist."""
        # 1. Signal Log CSV Setup (every candle signal)
        if not os.path.exists(self.signal_log_path):
            headers = ["datetime", "open", "close", "gbm_signal", "tcn_signal", "hmm_regime", "final_entry_signal"]
            with open(self.signal_log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)

        # 2. Trade Log CSV Setup
        if not os.path.exists(self.trade_log_path):
            headers = [
                "signal", "entry_time", "exit_time", "entry_nifty", "exit_nifty", 
                "lot_size", "nifty_pnl_points", "nifty_pnl_inr", "exit_reason"
            ]
            with open(self.trade_log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)

        # 3. Position Log CSV Setup
        if not os.path.exists(self.position_log_path):
            headers = [
                "entry_time", "exit_time", "entry_nifty", "exit_nifty", 
                "signal", "lot_size", "nifty_pnl_points", "nifty_pnl_inr", "exit_reason"
            ]
            with open(self.position_log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)

        # 4. Daily PnL CSV Setup
        if not os.path.exists(self.daily_pnl_path):
            headers = ["datetime", "total_trade", "wintrade", "1 day pnl", "total pnl", "net pnl"]
            with open(self.daily_pnl_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)

    def log_activity(self, message):
        """Logs a general system message to system.log."""
        now = datetime.datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        msecs = int(now.microsecond / 1000)
        formatted_msg = f"{ts},{msecs:03d} [INFO] {message}\n"
        print(formatted_msg.strip())
        with open(self.system_log_path, "a", encoding="utf-8") as f:
            f.write(formatted_msg)

    def log_candle_metrics(self, timestamp, ohlcv, predictions, signal, final_entry_signal):
        """
        Logs every candle's metrics to signal_log.csv.
        """
        gbm_buy = predictions.get('gbm_prob_buy', 0.0)
        gbm_sell = predictions.get('gbm_prob_sell', 0.0)
        if gbm_buy >= 0.5:
            gbm_sig_str = f"BUY({gbm_buy:.4f})"
        else:
            gbm_sig_str = f"SELL({gbm_sell:.4f})"

        tcn_pred = predictions.get('tcn_predicted', 'HOLD')
        tcn_buy = predictions.get('tcn_prob_buy', 0.0)
        tcn_sell = predictions.get('tcn_prob_sell', 0.0)
        tcn_hold = 1.0 - tcn_buy - tcn_sell
        if tcn_pred == 'BUY':
            tcn_sig_str = f"BUY({tcn_buy:.4f})"
        elif tcn_pred == 'SELL':
            tcn_sig_str = f"SELL({tcn_sell:.4f})"
        else:
            tcn_sig_str = f"HOLD({tcn_hold:.4f})"

        hmm_regime = predictions.get('hmm_regime_name', 'compression')
        
        try:
            dt = pd.to_datetime(timestamp, format='mixed')
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            dt_str = timestamp

        row = [
            dt_str,
            ohlcv['open'],
            ohlcv['close'],
            gbm_sig_str,
            tcn_sig_str,
            hmm_regime,
            final_entry_signal
        ]
        
        with open(self.signal_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def log_completed_trade(self, trade):
        """
        Logs a completed trade to both trade_log.csv and position_log.csv.
        """
        try:
            entry_dt = pd.to_datetime(trade.get("entry_time"), format='mixed')
            entry_dt_str = entry_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            entry_dt_str = trade.get("entry_time")

        try:
            exit_dt = pd.to_datetime(trade.get("exit_time"), format='mixed')
            exit_dt_str = exit_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            exit_dt_str = trade.get("exit_time")

        entry_nifty = float(trade.get("entry_nifty", 0))
        exit_nifty = float(trade.get("exit_nifty", 0))
        signal = trade.get("signal")
        lots = int(trade.get("option_lots", 1))
        lot_size = 75 * lots # Nifty Direct lot size
        
        nifty_pnl_points = (exit_nifty - entry_nifty) if signal == "BUY" else (entry_nifty - exit_nifty)
        nifty_pnl_inr = nifty_pnl_points * lot_size

        trade_row = [
            signal,
            entry_dt_str,
            exit_dt_str,
            entry_nifty,
            exit_nifty,
            lot_size,
            round(nifty_pnl_points, 2),
            round(nifty_pnl_inr, 2),
            trade.get("exit_reason")
        ]
        with open(self.trade_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(trade_row)

        pos_row = [
            entry_dt_str,
            exit_dt_str,
            entry_nifty,
            exit_nifty,
            signal,
            lot_size,
            round(nifty_pnl_points, 2),
            round(nifty_pnl_inr, 2),
            trade.get("exit_reason")
        ]
        with open(self.position_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(pos_row)

        # Also append to the global strategy trade tracking CSV files
        strategy = trade.get("strategy", "243A")
        if strategy == "243A":
            try:
                model_signal_path = "backend_engine/model signal.csv"
                if os.path.exists(model_signal_path):
                    df_sig = pd.read_csv(model_signal_path)
                    new_df = pd.DataFrame([{
                        'Entry Time': entry_dt_str,
                        'Exit Time': exit_dt_str,
                        'Direction': signal,
                        'Nifty Enter Price': entry_nifty,
                        'Nifty Exit Price': exit_nifty,
                        '1 QTY PnL': round(nifty_pnl_points, 2),
                        '65 QTY PnL': round(nifty_pnl_points * 65.0, 2),
                        'Exit Reason': trade.get("exit_reason", "Consensus signal"),
                        'Lot Size': float(lots)
                    }])
                    df_sig = pd.concat([df_sig, new_df], ignore_index=True)
                    df_sig.to_csv(model_signal_path, index=False)
                    # Sync with model_2024_25
                    dest_nifty = "model_2024_25/merged_01-01-2024_to_06-30-2026_nifty.csv"
                    df_sig.to_csv(dest_nifty, index=False)
            except Exception as e:
                print(f"[TradeLogger] Error saving to model signal.csv: {e}")
        elif strategy == "LONGPING":
            try:
                lp_path = "model_2024_25/backtest_results_longping.csv"
                if os.path.exists(lp_path):
                    df_lp = pd.read_csv(lp_path)
                    new_df = pd.DataFrame([{
                        'strategy': 'LONGPING',
                        'signal': 'BUY',
                        'entry_time': entry_dt_str,
                        'exit_time': exit_dt_str,
                        'entry_nifty': entry_nifty,
                        'exit_nifty': exit_nifty,
                        'qty': lots * 65,
                        'pnl_points': round(nifty_pnl_points, 2),
                        'pnl_inr': round(nifty_pnl_points * lots * 65, 2),
                        'exit_reason': trade.get("exit_reason")
                    }])
                    df_lp = pd.concat([df_lp, new_df], ignore_index=True)
                    df_lp.to_csv(lp_path, index=False)
            except Exception as e:
                print(f"[TradeLogger] Error saving to backtest_results_longping.csv: {e}")

    def update_daily_pnl(self, exit_time, all_trades):
        """
        Calculates and updates daily_pnl.csv.
        """
        if not exit_time:
            return
        
        try:
            exit_dt = pd.to_datetime(exit_time, format='mixed')
            target_date = exit_dt.date()
        except Exception:
            return
        
        daily_trades = []
        for t in all_trades:
            t_exit_time = t.get("exit_time")
            if t_exit_time:
                try:
                    t_exit_dt = pd.to_datetime(t_exit_time, format='mixed')
                    if t_exit_dt.date() == target_date:
                        daily_trades.append(t)
                except Exception:
                    pass
                    
        if not daily_trades:
            return
            
        realized_pnl = sum(t.get("pnl", 0.0) for t in daily_trades)
        total_trade = len(daily_trades)
        wintrade = sum(1 for t in daily_trades if t.get("pnl", 0.0) > 0.0)
        
        total_pnl = sum(t.get("pnl", 0.0) for t in all_trades)
        total_net_pnl = sum(t.get("pnl", 0.0) - 40.0 for t in all_trades)
        
        records = {}
        if os.path.exists(self.daily_pnl_path):
            try:
                df = pd.read_csv(self.daily_pnl_path, encoding="utf-8")
                for _, row_data in df.iterrows():
                    records[str(row_data["datetime"])] = {
                        "total_trade": int(row_data["total_trade"]),
                        "wintrade": int(row_data["wintrade"]),
                        "1 day pnl": float(row_data["1 day pnl"]),
                        "total pnl": float(row_data["total pnl"]),
                        "net pnl": float(row_data["net pnl"]) if "net pnl" in row_data else float(row_data["total pnl"])
                    }
            except Exception as e:
                self.log_activity(f"Error reading daily_pnl.csv: {e}")
                
        dt_key = f"{target_date} 15:30:00"
        records[dt_key] = {
            "total_trade": total_trade,
            "wintrade": wintrade,
            "1 day pnl": round(realized_pnl, 2),
            "total pnl": round(total_pnl, 2),
            "net pnl": round(total_net_pnl, 2)
        }
        
        try:
            with open(self.daily_pnl_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["datetime", "total_trade", "wintrade", "1 day pnl", "total pnl", "net pnl"])
                for d, data in sorted(records.items()):
                    writer.writerow([d, data["total_trade"], data["wintrade"], data["1 day pnl"], data["total pnl"], data["net pnl"]])
        except Exception as e:
            self.log_activity(f"Error writing daily_pnl.csv: {e}")

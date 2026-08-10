# PASSKEY: rushit2712
import os
import pandas as pd
import numpy as np
from backend_engine.strategies import get_strategy

def get_strategy_signals_for_chart(df: pd.DataFrame, strategy_name: str) -> list:
    """
    Computes signals across the entire combined dataset.
    Returns Lightweight Charts markers: [{time, position, color, shape, text}]
    Uses vector acceleration for ZFTF to process 100k+ rows instantly.
    For 243A, it only evaluates ML models on candles where volume > 0 and only for the last 30 days.
    """
    markers = []
    if len(df) < 150:
        return markers
        
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    import pytz
    ist_tz = pytz.timezone("Asia/Kolkata")
    
    def to_epoch(dt):
        if dt.tzinfo is None:
            return int(ist_tz.localize(dt).timestamp())
        return int(dt.astimezone(ist_tz).timestamp())

    # Try to load precalculated backtest signals from CSV files (instant!)
    if strategy_name == "LONGPINE_ZFTF":
        csv_sig_path = "longpine/backtest_signals.csv"
        if os.path.exists(csv_sig_path):
            try:
                df_sig = pd.read_csv(csv_sig_path)
                df_sig['timestamp'] = pd.to_datetime(df_sig['timestamp'])
                for _, row in df_sig.iterrows():
                    t = to_epoch(row['timestamp'])
                    sig_type = row['type']
                    sig_text = row['signal'] # BUY or SELL
                    
                    if sig_type == "ENTRY":
                        markers.append({"time": t, "position": "belowBar", "color": "#10b981", "shape": "arrowUp", "text": "BUY"})
                    else:
                        markers.append({"time": t, "position": "aboveBar", "color": "#ef4444", "shape": "arrowDown", "text": "SELL"})
                return markers
            except Exception as e:
                print(f"Error loading ZFTF backtest signals: {e}")
    elif strategy_name == "243A":
        csv_sig_path = "243A/backtest_signals.csv"
        csv_results_path = "243A/backtest_results.csv"
        last_backtest_time = None
        if os.path.exists(csv_sig_path) and os.path.exists(csv_results_path):
            try:
                df_results = pd.read_csv(csv_results_path)
                for idx, row in df_results.iterrows():
                    num = idx + 1
                    direction = row.get("Direction", "BUY")
                    entry_time = pd.to_datetime(row["Entry Time"])
                    exit_time = pd.to_datetime(row["Exit Time"])
                    exit_reason = row.get("Exit Reason", "EOD")
                    
                    if last_backtest_time is None or exit_time > last_backtest_time:
                        last_backtest_time = exit_time
                        
                    t_entry = to_epoch(entry_time)
                    t_exit = to_epoch(exit_time)
                    
                    entry_label = f"BUY {num}" if direction == "BUY" else f"SELL {num}"
                    exit_label = f"EXIT {num} ({exit_reason})"
                    
                    markers.append({
                        "time": t_entry,
                        "position": "belowBar" if direction == "BUY" else "aboveBar",
                        "color": "#10b981" if direction == "BUY" else "#ef4444",
                        "shape": "arrowUp" if direction == "BUY" else "arrowDown",
                        "text": entry_label
                    })
                    markers.append({
                        "time": t_exit,
                        "position": "aboveBar",
                        "color": "#3b82f6",
                        "shape": "circle",
                        "text": exit_label
                    })
            except Exception as e:
                print(f"Error loading 243A backtest signals: {e}")

    df['time_epoch'] = df['timestamp'].apply(lambda x: to_epoch(x))
    
    # 1. Fallback for Longpine ZFTF Strategy Markers
    if strategy_name == "LONGPINE_ZFTF":
        df['mean_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['zscore'] = (df['close'] - df['mean_20']) / df['std_20']
        
        # Calculate linear regression slope
        x = np.arange(20)
        x_mean = x.mean()
        x_dev = x - x_mean
        x_var = (x_dev**2).sum()
        def calc_slope(y):
            return np.dot(x_dev, y) / x_var
        df['slope'] = df['close'].rolling(20).apply(calc_slope, raw=True)
        
        close_prices = df['close'].values
        zscores = df['zscore'].values
        slopes = df['slope'].values
        epochs = df['time_epoch'].values
        
        in_position = False
        entry_price = 0.0
        
        for idx in range(20, len(df)):
            close = float(close_prices[idx])
            z = float(zscores[idx]) if not np.isnan(zscores[idx]) else 0.0
            slope = float(slopes[idx]) if not np.isnan(slopes[idx]) else 0.0
            t = int(epochs[idx])
            
            if not in_position:
                if z > 2.0 and slope > 0.001:
                    markers.append({"time": t, "position": "belowBar", "color": "#10b981", "shape": "arrowUp", "text": "BUY"})
                    entry_price = close
                    in_position = True
            else:
                target_hit = close >= entry_price * 1.06
                stop_hit = close <= entry_price * 0.97
                trend_exit = slope < -0.001
                
                if target_hit or stop_hit or trend_exit:
                    markers.append({"time": t, "position": "aboveBar", "color": "#ef4444", "shape": "arrowDown", "text": "SELL"})
                    in_position = False
                    
    # 2. Cache-enabled 243A Consensus Strategy Markers simulation
    elif strategy_name == "243A":
        strategy = get_strategy("243A")
        
        # Determine simulation start period
        if 'last_backtest_time' in locals() and last_backtest_time is not None:
            # Start slightly before the last backtest trade to capture any live continuation
            cutoff_date = last_backtest_time - pd.Timedelta(hours=24)
        else:
            last_date = df['timestamp'].max()
            cutoff_date = last_date - pd.Timedelta(days=30)
            
        matching_indices = df[df['timestamp'] >= cutoff_date].index
        start_idx = max(150, matching_indices[0]) if len(matching_indices) > 0 else 150
        
        # Load persistent signals cache
        from backend_engine.signal_cacher import load_cached_predictions, get_cached_prediction, save_predictions_batch
        cache_df = load_cached_predictions()
        
        existing_marker_times = set(m["time"] for m in markers)
        active_positions = []
        trade_counter = len(df_results) if 'df_results' in locals() else 0
        
        new_predictions = []
        
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            volume = float(row.get('volume', 0))
            if volume <= 0:
                continue
                
            t = int(row['time_epoch'])
            close = float(row['close'])
            timestamp = row['timestamp']
            
            # Read from prediction cache or evaluate ML model
            pred = get_cached_prediction(cache_df, timestamp, "243A")
            if pred is None:
                lookback = df.iloc[max(0, idx - 149) : idx + 1].reset_index(drop=True)
                try:
                    pred = strategy.predict(lookback)
                    new_predictions.append({
                        "timestamp": timestamp,
                        "strategy": "243A",
                        "signal": pred.get("signal", 0),
                        "metrics": pred.get("metrics", {})
                    })
                except Exception:
                    pred = {"signal": 0, "metrics": {}}
                    
            signal = pred.get('signal', 0)
            
            # Position management tracking
            exited_positions = []
            for pos in list(active_positions):
                pos_type = pos["type"]
                entry_price = pos["entry_price"]
                sl_hit = False
                tp_hit = False
                trend_exit = False
                
                if pos_type == 'LONG':
                    if close <= entry_price - 60.0:
                        sl_hit = True
                    elif close >= entry_price + 150.0:
                        tp_hit = True
                    elif signal == -1:
                        trend_exit = True
                else:
                    if close >= entry_price + 60.0:
                        sl_hit = True
                    elif close <= entry_price - 150.0:
                        tp_hit = True
                    elif signal == 1:
                        trend_exit = True
                        
                if sl_hit or tp_hit or trend_exit:
                    reason = "SL" if sl_hit else ("TP" if tp_hit else "REV")
                    if t not in existing_marker_times:
                        markers.append({
                            "time": t,
                            "position": "aboveBar" if pos_type == "LONG" else "belowBar",
                            "color": "#3b82f6",
                            "shape": "circle",
                            "text": f"EXIT {pos['num']} ({reason})"
                        })
                    exited_positions.append(pos)
                    
            for pos in exited_positions:
                active_positions.remove(pos)
                
            if len(active_positions) < 3 and signal != 0:
                pos_type = "LONG" if signal == 1 else "SHORT"
                trade_counter += 1
                active_positions.append({
                    "type": pos_type,
                    "entry_price": close,
                    "num": trade_counter
                })
                if t not in existing_marker_times:
                    markers.append({
                        "time": t,
                        "position": "belowBar" if pos_type == "LONG" else "aboveBar",
                        "color": "#10b981" if pos_type == "LONG" else "#ef4444",
                        "shape": "arrowUp" if pos_type == "LONG" else "arrowDown",
                        "text": f"BUY {trade_counter}" if pos_type == "LONG" else f"SELL {trade_counter}"
                    })
                    
        if new_predictions:
            save_predictions_batch(new_predictions)
                
    return markers

def run_strategy_backtest(data_df: pd.DataFrame, strategy_name: str, out_csv_path: str) -> dict:
    """
    Simulates a strategy over a historical Nifty dataset.
    Optimized via vectorized pandas execution for ZFTF strategy.
    Only simulates the last 3 months of the dataset to make it extremely fast,
    using preceding rows as a natural technical indicator warmup buffer.
    """
    df = data_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate 6-month cutoff date (180 days)
    last_date = df['timestamp'].max()
    cutoff_date = last_date - pd.Timedelta(days=180)
    matching_indices = df[df['timestamp'] >= cutoff_date].index
    
    capital = 100000.0
    starting_capital = capital
    trades_log = []
    
    # ZFTF Backtest (with 20-period window, rolling mean/std/linreg, LONG entry/exit only)
    if strategy_name == "LONGPINE_ZFTF":
        start_idx = max(20, matching_indices[0]) if len(matching_indices) > 0 else 20
        
        df['mean_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['zscore'] = (df['close'] - df['mean_20']) / df['std_20']
        
        x = np.arange(20)
        x_mean = x.mean()
        x_dev = x - x_mean
        x_var = (x_dev**2).sum()
        
        def calc_slope(y):
            return np.dot(x_dev, y) / x_var
            
        df['slope'] = df['close'].rolling(20).apply(calc_slope, raw=True)
        
        close_prices = df['close'].values
        zscores = df['zscore'].values
        slopes = df['slope'].values
        timestamps = df['timestamp'].values
        
        active_pos = None
        
        for idx in range(start_idx, len(df)):
            close = float(close_prices[idx])
            z = float(zscores[idx]) if not np.isnan(zscores[idx]) else 0.0
            slope = float(slopes[idx]) if not np.isnan(slopes[idx]) else 0.0
            curr_time = pd.Timestamp(timestamps[idx])
            
            # ZFTF EOD exit at 15:00
            is_eod = (curr_time.hour == 15 and curr_time.minute >= 0) or curr_time.hour > 15
            
            exited_this_bar = False
            if active_pos:
                pos_type = active_pos['position_type']
                entry_price = active_pos['entry_price']
                
                exit_reason = None
                stop_level = entry_price * 0.97
                target_level = entry_price * 1.06
                
                if is_eod:
                    exit_reason = "EOD"
                elif close <= stop_level:
                    exit_reason = "SL"
                elif close >= target_level:
                    exit_reason = "TP"
                elif slope < -0.001:
                    exit_reason = "Trend Exit"
                    
                if exit_reason:
                    qty = 65
                    pts_pnl = close - entry_price
                    pnl = pts_pnl * qty
                    capital += pnl
                    
                    trades_log.append({
                        "strategy": strategy_name,
                        "signal": "BUY",
                        "entry_time": active_pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        "exit_time": curr_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "entry_nifty": entry_price,
                        "exit_nifty": close,
                        "qty": qty,
                        "pnl_points": round(pts_pnl, 2),
                        "pnl_inr": round(pnl, 2),
                        "exit_reason": exit_reason
                    })
                    active_pos = None
                    exited_this_bar = True
                    
            if active_pos is None and not exited_this_bar and not is_eod:
                if z > 2.0 and slope > 0.001:
                    active_pos = {"position_type": "LONG", "entry_price": close, "entry_time": curr_time}
                    
    # Standard path for 243A Consensus Strategy (requires model lookbacks)
    else:
        start_idx = max(150, matching_indices[0]) if len(matching_indices) > 0 else 150
        strategy = get_strategy(strategy_name)
        active_pos = None
        
        for idx in range(start_idx, len(df)):
            current_bar = df.iloc[idx]
            curr_time = current_bar['timestamp']
            close = float(current_bar['close'])
            
            lookback = df.iloc[max(0, idx - 499) : idx + 1].reset_index(drop=True)
            
            try:
                pred = strategy.predict(lookback)
                signal = pred.get('signal', 0)
            except Exception:
                signal = 0
                
            exited_this_bar = False
            if active_pos:
                pos_type = active_pos['position_type']
                entry_price = active_pos['entry_price']
                
                sl_hit = False
                tp_hit = False
                trend_exit = False
                is_eod = (curr_time.hour == 15 and curr_time.minute >= 10) or curr_time.hour > 15
                
                if pos_type == 'LONG':
                    if close <= entry_price - 60.0:
                        sl_hit = True
                    elif close >= entry_price + 150.0:
                        tp_hit = True
                    elif signal == -1:
                        trend_exit = True
                else:
                    if close >= entry_price + 60.0:
                        sl_hit = True
                    elif close <= entry_price - 150.0:
                        tp_hit = True
                    elif signal == 1:
                        trend_exit = True
                        
                exit_reason = None
                if is_eod:
                    exit_reason = "EOD"
                elif sl_hit:
                    exit_reason = "SL"
                elif tp_hit:
                    exit_reason = "TP"
                elif trend_exit:
                    exit_reason = "Trend Exit"
                    
                if exit_reason:
                    qty = 65
                    pts_pnl = (close - entry_price) if pos_type == 'LONG' else (entry_price - close)
                    pnl = pts_pnl * qty
                    capital += pnl
                    
                    trades_log.append({
                        "strategy": strategy_name,
                        "signal": "BUY" if pos_type == "LONG" else "SELL",
                        "entry_time": active_pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        "exit_time": curr_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "entry_nifty": entry_price,
                        "exit_nifty": close,
                        "qty": qty,
                        "pnl_points": round(pts_pnl, 2),
                        "pnl_inr": round(pnl, 2),
                        "exit_reason": exit_reason
                    })
                    active_pos = None
                    exited_this_bar = True
                    
            if active_pos is None and not exited_this_bar:
                allow_entry = not ((curr_time.hour == 15 and curr_time.minute >= 10) or curr_time.hour > 15)
                if allow_entry and signal != 0:
                    pos_type = None
                    if signal == 1:
                        pos_type = "LONG"
                    elif signal == -1:
                        pos_type = "SHORT"
                        
                    if pos_type:
                        active_pos = {
                            "position_type": pos_type,
                            "entry_price": close,
                            "entry_time": curr_time
                        }
                        
    trades_df = pd.DataFrame(trades_log)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    trades_df.to_csv(out_csv_path, index=False)
    
    total_trades = len(trades_log)
    wins = sum(1 for t in trades_log if t['pnl_inr'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t['pnl_inr'] for t in trades_log)
    total_pts = sum(t['pnl_points'] for t in trades_log)
    
    return {
        "strategy": strategy_name,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "total_pnl_points": round(total_pts, 2),
        "total_pnl_inr": round(total_pnl, 2),
        "starting_capital": starting_capital,
        "ending_capital": round(capital, 2)
    }

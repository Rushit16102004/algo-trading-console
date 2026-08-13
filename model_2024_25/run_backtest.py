# PASSKEY: rushit2712
import os
import sys
import pandas as pd
import numpy as np
import time

# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_engine.strategies import get_strategy

def load_mixed_csv(path):
    """Parses mixed-column CSV files line-by-line to prevent tokenization errors."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    records = []
    headers = ['timestamp', 'open', 'high', 'low', 'close']
    for line in lines:
        parts = line.strip().split(',')
        if not parts or parts[0] == 'timestamp':
            if parts and parts[0] == 'timestamp':
                headers = parts
            continue
        rec = {headers[i]: parts[i] for i in range(min(len(headers), len(parts)))}
        records.append(rec)
        
    df = pd.DataFrame(records)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
    else:
        df['volume'] = 0.0
    return df

def run_longping_backtest(df):
    print("\n[LONGPING] Starting backtest (2024-01-01 to Present)...")
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 20-period technical indicator setup
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
    
    cutoff_date = pd.Timestamp('2024-01-01')
    matching_indices = df[df['timestamp'] >= cutoff_date].index
    start_idx = max(20, matching_indices[0]) if len(matching_indices) > 0 else 20
    
    capital = 100000.0
    starting_capital = capital
    trades_log = []
    active_pos = None
    
    for idx in range(start_idx, len(df)):
        close = float(close_prices[idx])
        z = float(zscores[idx]) if not np.isnan(zscores[idx]) else 0.0
        slope = float(slopes[idx]) if not np.isnan(slopes[idx]) else 0.0
        curr_time = pd.Timestamp(timestamps[idx])
        
        # EOD exit at 15:00
        is_eod = (curr_time.hour == 15 and curr_time.minute >= 0) or curr_time.hour > 15
        
        exited_this_bar = False
        if active_pos:
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
                    "strategy": "LONGPING",
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
                
    trades_df = pd.DataFrame(trades_log)
    
    total_trades = len(trades_log)
    wins = sum(1 for t in trades_log if t['pnl_inr'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t['pnl_inr'] for t in trades_log)
    total_pts = sum(t['pnl_points'] for t in trades_log)
    
    # Calculate Max Drawdown
    equity = np.cumsum([t['pnl_inr'] for t in trades_log]) if total_trades > 0 else []
    max_dd = 0.0
    if len(equity) > 0:
        equity_with_start = 100000.0 + equity
        peaks = np.maximum.accumulate(equity_with_start)
        drawdowns = peaks - equity_with_start
        max_dd = drawdowns.max()
        
    return trades_df, {
        "strategy": "LONGPING",
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "total_pnl_points": round(total_pts, 2),
        "total_pnl_inr": round(total_pnl, 2),
        "max_drawdown_inr": round(max_dd, 2),
        "starting_capital": starting_capital,
        "ending_capital": round(capital, 2)
    }

def run_243a_backtest(df):
    print("\n[243A] Starting backtest (2024-01-01 to Present)...")
    print("[243A] NOTE: This runs deep learning TCN, LightGBM, and HMM consensus logic. This will take a few minutes.")
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    cutoff_date = pd.Timestamp('2024-01-01')
    matching_indices = df[df['timestamp'] >= cutoff_date].index
    start_idx = max(500, matching_indices[0]) if len(matching_indices) > 0 else 500
    
    strategy = get_strategy("243A")
    capital = 100000.0
    starting_capital = capital
    trades_log = []
    active_pos = None
    
    total_steps = len(df) - start_idx
    print(f"[243A] Total candles to simulate: {total_steps}")
    
    for current_idx, idx in enumerate(range(start_idx, len(df))):
        current_bar = df.iloc[idx]
        curr_time = current_bar['timestamp']
        close = float(current_bar['close'])
        
        # Periodic progress log to keep terminal informed
        if idx % 1000 == 0 or current_idx == total_steps - 1:
            progress_pct = (current_idx / total_steps) * 100
            print(f"[243A] Progress: {progress_pct:.1f}% ({current_idx}/{total_steps} candles simulated)...")
            
        lookback = df.iloc[max(0, idx - 499) : idx + 1].reset_index(drop=True)
        
        try:
            pred = strategy.predict(lookback)
            signal = pred.get('signal', 0)
        except Exception as e:
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
                    "strategy": "243A",
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
                pos_type = "LONG" if signal == 1 else "SHORT"
                active_pos = {
                    "position_type": pos_type,
                    "entry_price": close,
                    "entry_time": curr_time
                }
                
    trades_df = pd.DataFrame(trades_log)
    
    total_trades = len(trades_log)
    wins = sum(1 for t in trades_log if t['pnl_inr'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t['pnl_inr'] for t in trades_log)
    total_pts = sum(t['pnl_points'] for t in trades_log)
    
    # Calculate Max Drawdown
    equity = np.cumsum([t['pnl_inr'] for t in trades_log]) if total_trades > 0 else []
    max_dd = 0.0
    if len(equity) > 0:
        equity_with_start = 100000.0 + equity
        peaks = np.maximum.accumulate(equity_with_start)
        drawdowns = peaks - equity_with_start
        max_dd = drawdowns.max()
        
    return trades_df, {
        "strategy": "243A",
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "total_pnl_points": round(total_pts, 2),
        "total_pnl_inr": round(total_pnl, 2),
        "max_drawdown_inr": round(max_dd, 2),
        "starting_capital": starting_capital,
        "ending_capital": round(capital, 2)
    }

def main():
    start_time = time.time()
    out_dir = "model_2024_25"
    os.makedirs(out_dir, exist_ok=True)
    
    data_path = "backend_engine/old data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Database file '{data_path}' not found!")
        return
        
    print(f"Loading candles from {data_path}...")
    df = load_mixed_csv(data_path)
    print(f"Loaded {len(df)} candles.")
    
    # 1. Run LONGPING Backtest
    lp_df, lp_stats = run_longping_backtest(df)
    lp_df.to_csv(os.path.join(out_dir, "backtest_results_longping.csv"), index=False)
    
    # 2. Run 243A Backtest
    a243_df, a243_stats = run_243a_backtest(df)
    a243_df.to_csv(os.path.join(out_dir, "backtest_results_243a.csv"), index=False)
    
    # 3. Save Summary Report
    report_path = os.path.join(out_dir, "backtest_summary.txt")
    summary = f"""==================================================
        HISTORICAL BACKTEST RESULTS (2024 - 2026)
==================================================
Date Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Date Range: 2024-01-01 to 2026-08-13 (Nifty Spot)

--------------------------------------------------
1. STRATEGY: LONGPING (Trend-Following)
--------------------------------------------------
Total Trades Taken:  {lp_stats['total_trades']}
Wins / Losses:        {lp_stats['wins']} W / {lp_stats['losses']} L
Win Rate (WR):        {lp_stats['win_rate']}%
Total PnL (Points):  {lp_stats['total_pnl_points']} pts
Total PnL (INR):     {lp_stats['total_pnl_inr']:.2f} INR
Max Drawdown (DD):   {lp_stats['max_drawdown_inr']:.2f} INR
Starting Capital:    {lp_stats['starting_capital']:.2f} INR
Ending Capital:      {lp_stats['ending_capital']:.2f} INR

--------------------------------------------------
2. STRATEGY: 243A (TCN + LGBM + HMM Consensus)
--------------------------------------------------
Total Trades Taken:  {a243_stats['total_trades']}
Wins / Losses:        {a243_stats['wins']} W / {a243_stats['losses']} L
Win Rate (WR):        {a243_stats['win_rate']}%
Total PnL (Points):  {a243_stats['total_pnl_points']} pts
Total PnL (INR):     {a243_stats['total_pnl_inr']:.2f} INR
Max Drawdown (DD):   {a243_stats['max_drawdown_inr']:.2f} INR
Starting Capital:    {a243_stats['starting_capital']:.2f} INR
Ending Capital:      {a243_stats['ending_capital']:.2f} INR

==================================================
Total Backtest Elapsed Time: {time.time() - start_time:.2f} seconds
==================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
        
    print(summary)
    print(f"\nAll backtests complete! Files saved in: {out_dir}")

if __name__ == "__main__":
    main()

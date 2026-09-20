# PASSKEY: rushit2712
import os
import sys
import pandas as pd
import numpy as np
import subprocess

def run_longping_backtest(df):
    print("\n[LONGPING] Starting backtest (2024-01-01 to Present, EOD exits disabled)...")
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
        
        # EOD exit disabled (no force exits for LONGPING!)
        is_eod = False
        
        exited_this_bar = False
        if active_pos:
            entry_price = active_pos['entry_price']
            exit_reason = None
            stop_level = entry_price * 0.97
            target_level = entry_price * 1.06
            
            if close <= stop_level:
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
                
        if active_pos is None and not exited_this_bar:
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

def main():
    out_dir = "model_2024_25"
    os.makedirs(out_dir, exist_ok=True)
    
    data_path = "backend_engine/old data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Database file '{data_path}' not found!")
        return
        
    print(f"Loading candles from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp_dt').reset_index(drop=True)
    
    # Get index of first candle in 2024
    matching_indices = df[df['timestamp_dt'] >= '2024-01-01'].index
    if len(matching_indices) == 0:
        print("Error: No data found on or after 2024-01-01!")
        return
        
    start_idx = matching_indices[0]
    # Slice including 500 candles of warmup buffer before 2024-01-01
    warmup_start_idx = max(0, start_idx - 500)
    df_sliced = df.iloc[warmup_start_idx:].copy()
    
    sliced_csv_path = os.path.join(out_dir, "nifty_2024_26_warmup.csv")
    df_sliced.to_csv(sliced_csv_path, index=False)
    print(f"Saved sliced data with warmup to {sliced_csv_path} ({len(df_sliced)} rows).")
    
    # 1. Run LONGPING Backtest
    lp_df, lp_stats = run_longping_backtest(df)
    lp_df.to_csv(os.path.join(out_dir, "backtest_results_longping.csv"), index=False)
    
    # 2. Run the exact 243A/AAAback.py backtest as a subprocess
    print("\n--- Starting Model Backtest (AAAback.py) ---")
    backtest_cmd = [
        sys.executable,
        "243A/AAAback.py",
        "--data_path", sliced_csv_path,
        "--tick_dir", "nonexistent_directory", # force candle-based fallback
        "--start_date", "2024-01-01"
    ]
    
    # Run and capture output with PYTHONPATH set to project root
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")
    result = subprocess.run(backtest_cmd, capture_output=True, text=True, env=env)
    
    # Print output to console
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:")
        print(result.stderr)
        
    # Move the generated results from the 243A folder to the model_2024_25 folder
    source_results = os.path.join("243A", "backtest_results.csv")
    source_signals = os.path.join("243A", "backtest_signals.csv")
    
    dest_results = os.path.join(out_dir, "backtest_results.csv")
    dest_signals = os.path.join(out_dir, "backtest_signals.csv")
    
    if os.path.exists(source_results):
        os.replace(source_results, dest_results)
        print(f"Moved backtest results to: {dest_results}")
    if os.path.exists(source_signals):
        os.replace(source_signals, dest_signals)
        print(f"Moved backtest signals to: {dest_signals}")
        
    # Append the LONGPING results summary into the summary file along with 243A
    summary_path = os.path.join(out_dir, "backtest_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("        HISTORICAL BACKTEST RESULTS (2024 - 2026)\n")
        f.write("==================================================\n")
        f.write(f"Date Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Dataset Date Range: 2024-01-01 to Present (Nifty Spot)\n\n")
        
        f.write("--------------------------------------------------\n")
        f.write("1. STRATEGY: LONGPING (EOD Exits Disabled)\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Total Trades Taken:  {lp_stats['total_trades']}\n")
        f.write(f"Wins / Losses:        {lp_stats['wins']} W / {lp_stats['losses']} L\n")
        f.write(f"Win Rate (WR):        {lp_stats['win_rate']}%\n")
        f.write(f"Total PnL (Points):  {lp_stats['total_pnl_points']} pts\n")
        f.write(f"Total PnL (INR):     {lp_stats['total_pnl_inr']:.2f} INR\n")
        f.write(f"Max Drawdown (DD):   {lp_stats['max_drawdown_inr']:.2f} INR\n")
        f.write(f"Starting Capital:    {lp_stats['starting_capital']:.2f} INR\n")
        f.write(f"Ending Capital:      {lp_stats['ending_capital']:.2f} INR\n\n")
        
        f.write("--------------------------------------------------\n")
        f.write("2. STRATEGY: 243A Consensus Strategy (Risk Sizer Enabled)\n")
        f.write("--------------------------------------------------\n")
        f.write(result.stdout)
        
        if result.stderr:
            f.write("\n\n=== Errors/Warnings ===\n")
            f.write(result.stderr)
            
    print(f"\nAll backtests complete! Files saved in: {out_dir}")
    
    # 4. Automatically push results to GitHub
    print("\n[GITHUB] Uploading results to repository...")
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update backtest results (LONGPING without EOD, 243A with sizer)"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("[GITHUB] Push completed successfully!")
    except Exception as git_err:
        print(f"[GITHUB] Error pushing to GitHub: {git_err}")

if __name__ == "__main__":
    main()

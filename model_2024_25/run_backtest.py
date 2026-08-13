# PASSKEY: rushit2712
import os
import sys
import pandas as pd
import subprocess

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
    
    # Run the exact 243A/AAAback.py backtest as a subprocess
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
        
    # Save the output console print as the summary report
    summary_path = os.path.join(out_dir, "backtest_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n=== Errors/Warnings ===\n")
            f.write(result.stderr)
            
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
        
    # Remove old longping backtest files if they exist to keep the folder clean
    old_lp_results = os.path.join(out_dir, "backtest_results_longping.csv")
    old_lp_243a_results = os.path.join(out_dir, "backtest_results_243a.csv")
    if os.path.exists(old_lp_results):
        os.remove(old_lp_results)
    if os.path.exists(old_lp_243a_results):
        os.remove(old_lp_243a_results)
        
    print(f"\nAll backtests complete! Files saved in: {out_dir}")

if __name__ == "__main__":
    main()

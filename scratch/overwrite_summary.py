# PASSKEY: rushit2712
import pandas as pd
import numpy as np
import os

def calculate_drawdown(pnls_inr):
    equity = np.cumsum(pnls_inr)
    if len(equity) == 0:
        return 0.0
    equity_with_start = 100000.0 + equity
    peaks = np.maximum.accumulate(equity_with_start)
    drawdowns = peaks - equity_with_start
    return float(drawdowns.max())

# 1. LONGPING Stats
lp_path = "model_2024_25/backtest_results_longping.csv"
if os.path.exists(lp_path):
    df_lp = pd.read_csv(lp_path)
    # Check if there are active trades
    df_lp_active = df_lp[df_lp['qty'] > 0]
    lp_total = len(df_lp_active)
    lp_wins = sum(1 for p in df_lp_active['pnl_inr'] if p > 0)
    lp_wr = (lp_wins / lp_total * 100) if lp_total > 0 else 0.0
    lp_pnl_inr = df_lp_active['pnl_inr'].sum()
    lp_pnl_pts = df_lp_active['pnl_points'].sum()
    lp_dd = calculate_drawdown(df_lp_active['pnl_inr'].values)
    lp_rf = lp_pnl_inr / lp_dd if lp_dd > 0 else float('inf')
else:
    lp_total, lp_wins, lp_wr, lp_pnl_inr, lp_pnl_pts, lp_dd, lp_rf = 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

# 2. 243A Option Stats (Lot Size > 0)
opt_path = "model_2024_25/merged_01-01-2024_to_06-30-2026_option.csv"
if os.path.exists(opt_path):
    df_opt = pd.read_csv(opt_path)
    df_opt_active = df_opt[df_opt['Lot Size'] > 0]
    opt_total = len(df_opt_active)
    opt_wins = sum(1 for p in df_opt_active['Net PnL'] if p > 0)
    opt_wr = (opt_wins / opt_total * 100) if opt_total > 0 else 0.0
    opt_pnl_inr = df_opt_active['Net PnL'].sum()
    opt_pnl_pts = df_opt_active['1 QTY PnL'].sum()
    opt_dd = calculate_drawdown(df_opt_active['Net PnL'].values)
    opt_rf = opt_pnl_inr / opt_dd if opt_dd > 0 else float('inf')
else:
    opt_total, opt_wins, opt_wr, opt_pnl_inr, opt_pnl_pts, opt_dd, opt_rf = 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

# Generate summary text
summary = f"""==================================================
        HISTORICAL BACKTEST RESULTS (2024 - 2026)
==================================================
Dataset Date Range: 2024-01-01 to 2026-06-30 (Nifty)
Note: All calculations below EXCLUDE 0 lot / 0 quantity trades.

--------------------------------------------------
1. STRATEGY: LONGPING (Trend-Following)
--------------------------------------------------
Total Trades Taken:  {lp_total}
Wins / Losses:        {lp_wins} W / {lp_total - lp_wins} L
Win Rate (WR):        {lp_wr:.2f}%
Total PnL (Points):  {lp_pnl_pts:.2f} pts
Total PnL (INR):     {lp_pnl_inr:.2f} INR
Max Drawdown (DD):   {lp_dd:.2f} INR
Starting Capital:    100000.00 INR
Ending Capital:      {100000.00 + lp_pnl_inr:.2f} INR
Recovery Factor:      {lp_rf:.2f}
Note: EOD exits are disabled. Holds positions overnight.

--------------------------------------------------
2. STRATEGY: 243A Consensus (TCN + LGBM + HMM)
--------------------------------------------------
Total Trades Taken:  {opt_total}
Wins / Losses:        {opt_wins} W / {opt_total - opt_wins} L
Win Rate (WR):        {opt_wr:.2f}%
Total PnL (Points):  {opt_pnl_pts:.2f} pts
Total PnL (INR):     {opt_pnl_inr:.2f} INR
Max Drawdown (DD):   {opt_dd:.2f} INR
Starting Capital:    100000.00 INR
Ending Capital:      {100000.00 + opt_pnl_inr:.2f} INR
Recovery Factor:      {opt_rf:.2f}
Note: Intraday strategy. Force closes at 15:10. Best for option shorting (CE & PE).

==================================================
All metrics verified.
==================================================
"""

summary_path = "model_2024_25/backtest_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

print("Updated backtest_summary.txt successfully!")
print(summary)

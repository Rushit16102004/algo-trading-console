# PASSKEY: rushit2712
import pandas as pd
import numpy as np

path = "model_2024_25/merged_01-01-2024_to_06-30-2026_option.csv"
df = pd.read_csv(path)

# Filter out 0 lot trades
df_active = df[df['Lot Size'] > 0].copy()

print("Original rows:", len(df))
print("Active rows (Lot Size > 0):", len(df_active))

# Calculate Win Rate based on Option PnL
# Let's check both Nifty 1 Qty PnL, Option 1 Lot PnL and Net PnL to see which represents the outcome
pnl_col = 'Net PnL' if 'Net PnL' in df_active.columns else 'Option 1 Lot PnL'
print("Using PnL column:", pnl_col)

pnls = df_active[pnl_col].values
wins = sum(1 for p in pnls if p > 0)
total = len(pnls)
wr = (wins / total * 100) if total > 0 else 0.0

total_pnl = pnls.sum()

# Max Drawdown
equity = np.cumsum(pnls)
if len(equity) > 0:
    # Assuming standard capital base of 100,000 INR
    equity_curve = 100000.0 + equity
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = peaks - equity_curve
    max_dd = drawdowns.max()
else:
    max_dd = 0.0

rec_factor = total_pnl / max_dd if max_dd > 0 else float('inf')

print(f"\n--- OPTION BACKTEST STATISTICS (Active Trades Only) ---")
print(f"Total Active Trades: {total}")
print(f"Wins / Losses:       {wins} W / {total - wins} L")
print(f"Win Rate (WR):       {wr:.2f}%")
print(f"Total PnL:           {total_pnl:.2f} INR")
print(f"Max Drawdown (DD):   {max_dd:.2f} INR")
print(f"Recovery Factor:     {rec_factor:.2f}")

# Let's also check nifty points PnL for active trades
nifty_pts = df_active['1 QTY PnL'].values
nifty_total_pts = nifty_pts.sum()
nifty_wins = sum(1 for p in nifty_pts if p > 0)
nifty_wr = (nifty_wins / total * 100) if total > 0 else 0.0
print(f"\n--- NIFTY POINTS STATISTICS (Active Trades Only) ---")
print(f"Total Nifty Points PnL: {nifty_total_pts:.2f} pts")
print(f"Nifty Win Rate:         {nifty_wr:.2f}%")

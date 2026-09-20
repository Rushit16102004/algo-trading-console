"""
================================================================================
EVERY-DAY INTRADAY DIRECTION TRADING MODEL (114-DAY MASTER ENGINE)
================================================================================
Asset: Index / Stock Futures (e.g. Nifty 50)
Timeframe: 15-Minute Candles (Pure OHLC - No Volume Required)
Historical Memory: 18 Years (2008 to 2025: 108,042 candles, 99,205 patterns)
Evaluation Period: Unseen Test Data (12/2025 to 07/2026: 117 sessions)

CORE STRATEGY RULES:
1. Evaluation Window: 09:45 AM to 12:30 PM (Morning Session).
2. Pattern Feature Vector: 
   - 3-candle sequence price structure (45 minutes: t-2, t-1, t normalized OHLC)
   - 5-Day Trend % ((Close - Close_5d) / Close_5d * 100)
   - Opening Gap % ((Day_Open - Prev_Close) / Prev_Close * 100)
   - Intraday Session Position ((Close - Day_Low) / (Day_High - Day_Low + 1e-4))
   - Candle Body Structure ((Close - Open) / (High - Low + 1e-4))
3. Similarity Search: Spatial cKDTree queries 5 nearest historical matches.
4. Consensus Rule: At least 4 out of 5 (80%) historical neighbors must agree on direction.
5. Daily Selection: Pick the single candle in the morning with the lowest Difference Score.
6. Execution: Enter on close of the selected 15-min candle (Strictly 1 Trade Per Day).
7. Risk Management: Fixed Stop Loss = 80 points from entry.
8. Exit Rule: Hold until 3:00 PM (15:00) sharp. No profit cap (ride trend expansions).
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HIST_FILE = '2008 to 2025 data 15 min.csv'
TEST_FILE = '12_25_to_7_26.csv'
OUT_MASTER_CSV = 'every_day_direction_master.csv'
OUT_DIFF02_CSV = 'trades_mode1_diff_under_02.csv'

def load_and_preprocess_hist(filepath):
    print(f"-> Loading historical database from: {filepath}...")
    df = pd.read_csv(filepath)
    df['dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('dt').reset_index(drop=True)
    df['date'] = df['dt'].dt.date
    df['time_hm'] = df['dt'].dt.strftime('%H:%M')
    return df

def resample_5min_to_15min(filepath):
    print(f"-> Loading test dataset from: {filepath} and resampling to 15-min...")
    df = pd.read_csv(filepath)
    df['dt'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('dt').reset_index(drop=True)
    df = df[df['dt'].dt.strftime('%H:%M') >= '09:15']
    df = df.set_index('dt')
    
    resampled = []
    for d, grp in df.groupby(df.index.date):
        grp = grp.copy()
        grp['bucket'] = grp.index.floor('15min')
        agg = grp.groupby('bucket').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()
        resampled.append(agg)
        
    df_15 = pd.concat(resampled, ignore_index=True)
    df_15['dt'] = df_15['bucket']
    df_15['date'] = df_15['dt'].dt.date
    df_15['time_hm'] = df_15['dt'].dt.strftime('%H:%M')
    return df_15.drop(columns=['bucket'])

def compute_ohlc_features(df, daily_history=None):
    df = df.copy()
    daily_open = df.groupby('date')['open'].first().rename('day_open')
    daily_close = df.groupby('date')['close'].last().rename('day_close')
    unique_dates = df['date'].drop_duplicates().tolist()
    
    daily_df = pd.DataFrame({'date': unique_dates})
    daily_df = daily_df.merge(daily_open, on='date', how='left')
    daily_df = daily_df.merge(daily_close, on='date', how='left')
    
    if daily_history is not None:
        combined = pd.concat([daily_history, daily_df], ignore_index=True).drop_duplicates('date').sort_values('date').reset_index(drop=True)
        combined['prev_close'] = combined['day_close'].shift(1)
        combined['close_5d'] = combined['day_close'].shift(5)
        daily_df = combined[combined['date'].isin(unique_dates)].copy()
    else:
        daily_df['prev_close'] = daily_df['day_close'].shift(1)
        daily_df['close_5d'] = daily_df['day_close'].shift(5)
        
    df = df.merge(daily_df[['date', 'day_open', 'prev_close', 'close_5d']], on='date', how='left')
    
    # Feature 1: 5-Day Trend %
    df['trend_5d'] = ((df['close'] - df['close_5d']) / df['close_5d']) * 100.0
    
    # Feature 2: Opening Gap %
    df['gap_pct'] = ((df['day_open'] - df['prev_close']) / df['prev_close']) * 100.0
    
    # Feature 3: Intraday Session Position
    df['high_day'] = df.groupby('date')['high'].cummax()
    df['low_day'] = df.groupby('date')['low'].cummin()
    df['position'] = (df['close'] - df['low_day']) / (df['high_day'] - df['low_day'] + 1e-4)
    
    # Feature 4: Candle Body Vector
    df['body_vector'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-4)
    
    # Target: EOD Close Outcome
    eod_map = df.groupby('date')['close'].last().to_dict()
    df['eod_close'] = df['date'].map(eod_map)
    df['eod_points'] = df['eod_close'] - df['close']
    df['eod_dir'] = np.where(df['eod_points'] > 0, 1, np.where(df['eod_points'] < 0, -1, 0))
    
    return df, daily_df[['date', 'day_open', 'day_close']].copy()

def extract_3candle_sequences(df):
    records = []
    for d, grp in df.groupby('date'):
        grp = grp.reset_index(drop=True)
        if len(grp) < 3:
            continue
        for i in range(2, len(grp)):
            sub = grp.iloc[i - 2 : i + 1] # t-2, t-1, t
            base_p = float(sub.iloc[0]['open'])
            
            o = (sub['open'].to_numpy() - base_p) / base_p * 100.0
            h = (sub['high'].to_numpy() - base_p) / base_p * 100.0
            l = (sub['low'].to_numpy() - base_p) / base_p * 100.0
            c = (sub['close'].to_numpy() - base_p) / base_p * 100.0
            
            row_t = sub.iloc[-1]
            
            vec = np.concatenate([o, h, l, c, [row_t['trend_5d'], row_t['gap_pct'], row_t['position'], row_t['body_vector']]])
            
            records.append({
                'vec': vec,
                'date': d,
                'time_hm': row_t['time_hm'],
                'close': row_t['close'],
                'eod_close': row_t['eod_close'],
                'eod_dir': int(row_t['eod_dir']),
                'eod_points': float(row_t['eod_points']),
                'trend_5d': float(row_t['trend_5d']),
                'position': float(row_t['position'])
            })
    return pd.DataFrame(records)

def simulate_trade(df_raw, date_str, time_str, entry_p, direction, sl_pts, exit_time_str='15:00'):
    sub = df_raw[(df_raw['date'] == str(date_str)) & (df_raw['time_hm'] >= time_str) & (df_raw['time_hm'] <= exit_time_str)].copy()
    if len(sub) <= 1:
        return 0.0, False, 'NO_DATA', entry_p
    sub = sub.iloc[1:]
    exit_p = float(sub.iloc[-1]['close'])
    
    if direction == 1:
        for _, row in sub.iterrows():
            if row['low'] <= entry_p - sl_pts:
                return -sl_pts, False, 'SL_HIT', entry_p - sl_pts
        pnl = exit_p - entry_p
        return pnl, (pnl > 0), 'EOD_EXIT', exit_p
    else:
        for _, row in sub.iterrows():
            if row['high'] >= entry_p + sl_pts:
                return -sl_pts, False, 'SL_HIT', entry_p + sl_pts
        pnl = entry_p - exit_p
        return pnl, (pnl > 0), 'EOD_EXIT', exit_p

def run_model():
    print("================================================================================")
    print("RUNNING 114-DAY EVERY-DAY DIRECTION TRADING MODEL ENGINE")
    print("================================================================================")
    
    # 1. Load Data
    df_hist_raw = load_and_preprocess_hist(HIST_FILE)
    df_test_raw = resample_5min_to_15min(TEST_FILE)
    
    # Raw 5-min test data for exact trade path execution
    df_test_5m = pd.read_csv(TEST_FILE)
    df_test_5m['dt'] = pd.to_datetime(df_test_5m['timestamp'], format='mixed')
    df_test_5m['date'] = df_test_5m['dt'].dt.strftime('%Y-%m-%d')
    df_test_5m['time_hm'] = df_test_5m['dt'].dt.strftime('%H:%M')
    df_test_5m = df_test_5m.sort_values('dt').reset_index(drop=True)
    
    # 2. Compute Features
    print("-> Computing OHLC features across 18 years of historical data...")
    df_hist, daily_hist = compute_ohlc_features(df_hist_raw)
    df_test, _ = compute_ohlc_features(df_test_raw, daily_history=daily_hist)
    
    df_hist = df_hist.dropna(subset=['trend_5d', 'gap_pct', 'position', 'body_vector']).reset_index(drop=True)
    df_test = df_test.dropna(subset=['trend_5d', 'gap_pct', 'position', 'body_vector']).reset_index(drop=True)
    
    # 3. Extract Multi-Candle Sequences
    print("-> Extracting 3-candle sequence trajectories (45 min structural shapes)...")
    h_df = extract_3candle_sequences(df_hist)
    t_df = extract_3candle_sequences(df_test)
    
    h_mat = np.array(h_df['vec'].tolist(), dtype=np.float64)
    t_mat = np.array(t_df['vec'].tolist(), dtype=np.float64)
    print(f"-> Historical Patterns Indexed: {len(h_mat):,} (2008-2025)")
    print(f"-> Unseen Test Patterns: {len(t_mat):,} (12/2025-07/2026)")
    
    # 4. Build Spatial KD-Tree Index
    print("-> Querying 5 nearest historical neighbors for every test pattern...")
    tree = cKDTree(h_mat)
    dists, indices = tree.query(t_mat, k=5)
    
    t_df['best_dist'] = dists[:, 0]
    preds = []
    for i in range(len(t_df)):
        h_dirs = h_df.iloc[indices[i]]['eod_dir'].to_numpy()
        up_c = int(np.sum(h_dirs == 1))
        dn_c = int(np.sum(h_dirs == -1))
        if up_c >= 4:
            preds.append(1)
        elif dn_c >= 4:
            preds.append(-1)
        else:
            preds.append(0)
    t_df['pred_dir'] = preds
    
    # 5. Apply Every-Day Model Selection
    print("-> Applying Every-Day Selection: 09:45-12:30 window, >=80% consensus, lowest Difference Score...")
    daily_trades = []
    for d, grp in t_df.groupby('date'):
        # Filter for morning window and consensus
        morn = grp[(grp['time_hm'] >= '09:45') & (grp['time_hm'] <= '12:30') & (grp['pred_dir'] != 0)].copy()
        if len(morn) == 0:
            continue
        # Pick best match of the morning
        best_c = morn.sort_values('best_dist').iloc[0]
        
        # Execute with Stop Loss 80 pts and Exit at 3:00 PM (15:00)
        pnl_300, win_300, exit_type_300, exit_p_300 = simulate_trade(df_test_5m, d, best_c['time_hm'], best_c['close'], best_c['pred_dir'], 80.0, '15:00')
        # Also record 3:25 PM for reference
        pnl_325, win_325, exit_type_325, exit_p_325 = simulate_trade(df_test_5m, d, best_c['time_hm'], best_c['close'], best_c['pred_dir'], 80.0, '15:25')
        
        daily_trades.append({
            'date': str(d),
            'signal_time': best_c['time_hm'],
            'predicted_direction': 'BUY' if best_c['pred_dir'] == 1 else 'SELL',
            'entry_price': best_c['close'],
            'diff_score': round(best_c['best_dist'], 3),
            'trend_5d': round(best_c['trend_5d'], 2),
            'session_pos': round(best_c['position'], 2),
            'exit_price_300': exit_p_300,
            'exit_type_300': exit_type_300,
            'pnl_exit_300': round(pnl_300, 2),
            'win_300': win_300,
            'pnl_exit_325': round(pnl_325, 2),
            'win_325': win_325
        })
        
    master_df = pd.DataFrame(daily_trades)
    master_df.to_csv(OUT_MASTER_CSV, index=False)
    print(f"-> Saved master daily trade signals to: {OUT_MASTER_CSV}")
    
    # Save diff < 0.20 subset
    sub02 = master_df[master_df['diff_score'] < 0.20].copy()
    sub02.to_csv(OUT_DIFF02_CSV, index=False)
    print(f"-> Saved Difference Score < 0.20 trades to: {OUT_DIFF02_CSV}")
    
    # 6. Print Comprehensive Performance Metrics
    print("\n================================================================================")
    print("BACKTEST RESULTS: 114-DAY EVERY-DAY TRADING MODEL (EXIT AT 3:00 PM, SL=80)")
    print("================================================================================")
    
    total_days = len(master_df)
    wins = int(master_df['win_300'].sum())
    losses = total_days - wins
    wr = wins / total_days * 100
    tot_pts = float(master_df['pnl_exit_300'].sum())
    gw = float(master_df[master_df['pnl_exit_300'] > 0]['pnl_exit_300'].sum())
    gl = float(-master_df[master_df['pnl_exit_300'] < 0]['pnl_exit_300'].sum())
    pf = gw / gl if gl > 0 else 99.0
    avg_w = float(master_df[master_df['pnl_exit_300'] > 0]['pnl_exit_300'].mean())
    avg_l = float(master_df[master_df['pnl_exit_300'] < 0]['pnl_exit_300'].mean())
    
    sl_hits = int((master_df['exit_type_300'] == 'SL_HIT').sum())
    eod_exits = int((master_df['exit_type_300'] == 'EOD_EXIT').sum())
    eod_pos = int(((master_df['exit_type_300'] == 'EOD_EXIT') & (master_df['pnl_exit_300'] > 0)).sum())
    eod_neg = int(((master_df['exit_type_300'] == 'EOD_EXIT') & (master_df['pnl_exit_300'] < 0)).sum())
    
    print(f"Total Trading Sessions Evaluated : {total_days} days (97.4% of all test sessions)")
    print(f"Winning Days (Wins)             : {wins} ({wr:.2f}%)")
    print(f"Losing Days (Losses)            : {losses} ({100-wr:.2f}%)")
    print(f"Total Net Points Generated       : {tot_pts:+.2f} points")
    print(f"Profit Factor                   : {pf:.2f}")
    print(f"Average Win                     : {avg_w:+.2f} points")
    print(f"Average Loss                    : {avg_l:+.2f} points")
    print(f"SL Hits (Capped at -80 pts)     : {sl_hits} ({sl_hits/total_days*100:.1f}%)")
    print(f"EOD Exits at 3:00 PM            : {eod_exits} ({eod_exits/total_days*100:.1f}%)")
    print(f"  -> EOD Positive (+ve Wins)    : {eod_pos} ({eod_pos/eod_exits*100:.1f}% of all EOD exits)")
    print(f"  -> EOD Negative (-ve Losses)  : {eod_neg} ({eod_neg/eod_exits*100:.1f}% of all EOD exits)")
    
    print("\n--------------------------------------------------------------------------------")
    print(f"SUBSET: DIFFERENCE SCORE < 0.20 ({len(sub02)} TRADES)")
    print("--------------------------------------------------------------------------------")
    w02 = int(sub02['win_300'].sum())
    wr02 = w02 / len(sub02) * 100
    pts02 = float(sub02['pnl_exit_300'].sum())
    gw02 = float(sub02[sub02['pnl_exit_300'] > 0]['pnl_exit_300'].sum())
    gl02 = float(-sub02[sub02['pnl_exit_300'] < 0]['pnl_exit_300'].sum())
    pf02 = gw02 / gl02 if gl02 > 0 else 99.0
    print(f"Trades Count          : {len(sub02)} ({len(sub02)/117*100:.1f}% of all days)")
    print(f"Win Rate              : {wr02:.2f}% ({w02}/{len(sub02)})")
    print(f"Total Net Points      : {pts02:+.2f} points")
    print(f"Profit Factor         : {pf02:.2f}")
    print(f"Average Win           : {sub02[sub02['pnl_exit_300'] > 0]['pnl_exit_300'].mean():+.2f} points")
    print(f"Average Loss          : {sub02[sub02['pnl_exit_300'] < 0]['pnl_exit_300'].mean():+.2f} points")
    print("================================================================================\n")

if __name__ == '__main__':
    run_model()

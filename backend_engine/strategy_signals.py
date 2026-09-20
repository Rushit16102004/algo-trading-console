# PASSKEY: rushit2712
import os
import pandas as pd
import numpy as np
from backend_engine.strategies import get_strategy

def get_strategy_signals_for_chart(df: pd.DataFrame, strategy_name: str = "243A") -> list:
    """
    Computes signals across historical candles for the 243A ML consensus strategy.
    Returns Lightweight Charts markers: [{time, position, color, shape, text}].
    """
    markers = []
    if len(df) < 150:
        return markers

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

    import pytz
    ist_tz = pytz.timezone("Asia/Kolkata")

    # Load precalculated signals from model signal.csv if available
    csv_results_path = "backend_engine/model signal.csv"
    last_backtest_time = None
    if os.path.exists(csv_results_path):
        try:
            df_results = pd.read_csv(csv_results_path)
            df_results['Entry Time'] = pd.to_datetime(df_results['Entry Time'])
            df_results['Exit Time'] = pd.to_datetime(df_results['Exit Time'])
            try:
                df_results['t_entry'] = df_results['Entry Time'].dt.tz_convert('Asia/Kolkata').astype('int64') // 10**9
                df_results['t_exit'] = df_results['Exit Time'].dt.tz_convert('Asia/Kolkata').astype('int64') // 10**9
            except TypeError:
                df_results['t_entry'] = df_results['Entry Time'].dt.tz_localize('Asia/Kolkata').astype('int64') // 10**9
                df_results['t_exit'] = df_results['Exit Time'].dt.tz_localize('Asia/Kolkata').astype('int64') // 10**9

            for idx, row in df_results.iterrows():
                num = idx + 1
                direction = row.get("Direction", "BUY")
                exit_time = row["Exit Time"]
                exit_reason = row.get("Exit Reason", "EOD")

                if last_backtest_time is None or exit_time > last_backtest_time:
                    last_backtest_time = exit_time

                t_entry = int(row['t_entry'])
                t_exit = int(row['t_exit'])

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
            print(f"[StrategySignals] Error loading 243A backtest signals: {e}")

    try:
        df['time_epoch'] = df['timestamp'].dt.tz_convert('Asia/Kolkata').astype('int64') // 10**9
    except TypeError:
        df['time_epoch'] = df['timestamp'].dt.tz_localize('Asia/Kolkata').astype('int64') // 10**9

    strategy = get_strategy("243A")
    if not strategy:
        return markers

    # Determine simulation start period
    if last_backtest_time is not None:
        cutoff_date = last_backtest_time - pd.Timedelta(hours=24)
    else:
        last_date = df['timestamp'].max()
        cutoff_date = last_date - pd.Timedelta(days=30)

    matching_indices = df[df['timestamp'] >= cutoff_date].index
    start_idx = max(150, matching_indices[0]) if len(matching_indices) > 0 else 150

    from backend_engine.signal_cacher import load_cached_predictions, get_cached_prediction, save_predictions_batch
    cache_df = load_cached_predictions()

    existing_marker_times = set(m["time"] for m in markers)
    active_positions = []
    trade_counter = len(df_results) if 'df_results' in locals() else 0
    new_predictions = []

    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        timestamp = row['timestamp']
        t = int(row['time_epoch'])
        close = float(row['close'])

        pred = get_cached_prediction(cache_df, timestamp, "243A")
        if pred is None:
            volume = float(row.get('volume', 0))
            if volume <= 0:
                continue
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

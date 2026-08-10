# PASSKEY: rushit2712
import os
import pandas as pd

CACHE_FILE = "backend_engine/signals_cache.csv"

def load_cached_predictions():
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame(columns=[
            "timestamp", "strategy", "signal", "hmm_regime",
            "gbm_prob_buy", "gbm_prob_sell", "tcn_predicted",
            "tcn_prob_buy", "tcn_prob_sell"
        ])
    try:
        df = pd.read_csv(CACHE_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "timestamp", "strategy", "signal", "hmm_regime",
            "gbm_prob_buy", "gbm_prob_sell", "tcn_predicted",
            "tcn_prob_buy", "tcn_prob_sell"
        ])

def get_cached_prediction(cache_df, timestamp, strategy):
    ts_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    match = cache_df[(cache_df['timestamp'] == ts_str) & (cache_df['strategy'] == strategy)]
    if not match.empty:
        row = match.iloc[0]
        return {
            "signal": int(row["signal"]),
            "metrics": {
                "hmm_regime": str(row["hmm_regime"]),
                "gbm_prob_buy": float(row["gbm_prob_buy"]),
                "gbm_prob_sell": float(row["gbm_prob_sell"]),
                "tcn_predicted": str(row["tcn_predicted"]),
                "tcn_prob_buy": float(row["tcn_prob_buy"]),
                "tcn_prob_sell": float(row["tcn_prob_sell"])
            }
        }
    return None

def save_predictions_batch(predictions_list):
    if not predictions_list:
        return
    df = load_cached_predictions()
    new_rows = []
    for item in predictions_list:
        ts_str = pd.to_datetime(item["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        strategy = item["strategy"]
        
        # Remove existing if any
        df = df[~((df['timestamp'] == ts_str) & (df['strategy'] == strategy))]
        
        new_rows.append({
            "timestamp": ts_str,
            "strategy": strategy,
            "signal": int(item["signal"]),
            "hmm_regime": item["metrics"].get("hmm_regime", "Unknown"),
            "gbm_prob_buy": float(item["metrics"].get("gbm_prob_buy", 0.5)),
            "gbm_prob_sell": float(item["metrics"].get("gbm_prob_sell", 0.5)),
            "tcn_predicted": item["metrics"].get("tcn_predicted", "HOLD"),
            "tcn_prob_buy": float(item["metrics"].get("tcn_prob_buy", 0.5)),
            "tcn_prob_sell": float(item["metrics"].get("tcn_prob_sell", 0.5))
        })
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(CACHE_FILE, index=False)

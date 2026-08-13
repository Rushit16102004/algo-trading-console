# PASSKEY: rushit2712
import pandas as pd
df = pd.read_csv("backend_engine/old data.csv")
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='mixed')
df_filtered = df[df['timestamp_dt'] >= '2024-01-01']
print("Number of candles since 2024-01-01:", len(df_filtered))

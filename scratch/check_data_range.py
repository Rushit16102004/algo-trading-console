# PASSKEY: rushit2712
import pandas as pd
df = pd.read_csv("backend_engine/old data.csv")
print("Date Range:")
print("First timestamp:", df['timestamp'].iloc[0])
print("Last timestamp:", df['timestamp'].iloc[-1])
print("Total rows:", len(df))

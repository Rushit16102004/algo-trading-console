# PASSKEY: rushit2712
import pandas as pd
import numpy as np

path = "model_2024_25/merged_01-01-2024_to_06-30-2026_option.csv"
df = pd.read_csv(path)
print("Columns:")
print(df.columns.tolist())
print("Total rows:", len(df))

# Check for lot size columns
lot_cols = [c for c in df.columns if 'lot' in c.lower() or 'qty' in c.lower() or 'size' in c.lower()]
print("Possible lot columns:", lot_cols)

"""
Download last 2 days OHLCV data from Angel One (SmartAPI).

Requirements:
    pip install smartapi-python pyotp pandas --break-system-packages

You need the following from your Angel One account:
    1. API Key        -> generated on https://smartapi.angelbroking.com (create an app)
    2. Client ID       -> your Angel One login / trading ID
    3. Password / PIN  -> your Angel One login password (MPIN)
    4. TOTP Secret     -> the secret key shown when you enable TOTP-based 2FA
                          (NOT the 6-digit code itself — the base32 secret used to generate it)

Fill in the CONFIG section below, or set them as environment variables instead
(recommended, so you don't hardcode credentials in the file).
"""

import os
import time
import datetime as dt
import pandas as pd
import pyotp
from SmartApi import SmartConnect   # pip package name: smartapi-python
from SmartApi.smartExceptions import DataException


# ---------------------- CONFIG ----------------------
API_KEY      = os.getenv("ANGEL_API_KEY",   "YOUR_API_KEY")
CLIENT_ID    = os.getenv("ANGEL_CLIENT_ID", "YOUR_CLIENT_ID")
PASSWORD     = os.getenv("ANGEL_PASSWORD",  "YOUR_MPIN")
TOTP_SECRET  = os.getenv("ANGEL_TOTP_SECRET", "YOUR_TOTP_SECRET")


# Instruments to fetch. You need the exchange symbol token (not the trading symbol).
# Find tokens from the master contract file:
# https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json
#
# NIFTY 50 index token is well-known and included below.
# Note: the index itself always has volume = 0 (nothing trades at the index
# value). Set INCLUDE_NIFTY_FUTURE = True below to also fetch the current-month
# NIFTY futures contract, which DOES have real traded volume.
SYMBOLS = [
    {"exchange": "NSE", "symboltoken": "99926000", "tradingsymbol": "NIFTY50"},
    # {"exchange": "NSE", "symboltoken": "3045", "tradingsymbol": "SBIN-EQ"},
]

INCLUDE_NIFTY_FUTURE = True   # adds the current-month NIFTY future (real volume) to SYMBOLS
INCLUDE_NIFTY50_VOLUME_SUM = True  # sums volume across all 50 constituent stocks as "Nifty volume"

# Candle interval. Options include:
# ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE, TEN_MINUTE, FIFTEEN_MINUTE,
# THIRTY_MINUTE, ONE_HOUR, ONE_DAY
INTERVAL = "FIVE_MINUTE"
# ------------------------------------------------------


def format_timestamp(ts):
    """Format like '2026-06-18 13:45:00'."""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def login():
    smart_api = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    session = smart_api.generateSession(CLIENT_ID, PASSWORD, totp)

    if not session or not session.get("status"):
        raise RuntimeError(f"Login failed: {session}")

    print("Login successful for:", CLIENT_ID)
    return smart_api


def get_current_nifty_future_token():
    """
    The NIFTY 50 index itself never has volume (nothing trades at the index
    value directly). To get real traded volume, use the current-month NIFTY
    futures contract instead. Futures tokens change every expiry, so this
    looks up the current one from Angel One's instrument master file.
    """
    import requests

    url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    instruments = requests.get(url, timeout=30).json()

    nifty_futs = [
        i for i in instruments
        if i.get("name") == "NIFTY"
        and i.get("instrumenttype") == "FUTIDX"
        and i.get("exch_seg") == "NFO"
    ]

    if not nifty_futs:
        raise RuntimeError("Could not find any NIFTY futures contracts in the instrument master.")

    # Pick the nearest expiry (front-month contract)
    nifty_futs.sort(key=lambda i: dt.datetime.strptime(i["expiry"], "%d%b%Y"))
    nearest = nifty_futs[0]

    print(f"Using NIFTY future: {nearest['symbol']} (token {nearest['token']}, expiry {nearest['expiry']})")
    return {"exchange": "NFO", "symboltoken": nearest["token"], "tradingsymbol": nearest["symbol"]}



def get_nifty50_constituent_tokens():
    """
    Looks up NSE tokens for the CURRENT Nifty 50 constituent stocks.
    Pulls the live official list from NSE (rebalanced semi-annually, Jan 31 /
    Jul 31 cutoffs) instead of a hardcoded list, so this stays accurate and
    always returns exactly 50 symbols even after index reshuffles.
    """
    import requests

    # Official NSE list of current Nifty 50 constituents
    nse_url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}  # NSE blocks requests with no user-agent
    nse_resp = requests.get(nse_url, headers=headers, timeout=30)
    nse_resp.raise_for_status()

    from io import StringIO
    nifty_list_df = pd.read_csv(StringIO(nse_resp.text))
    NIFTY_50_SYMBOLS = nifty_list_df["Symbol"].tolist()
    print(f"Fetched {len(NIFTY_50_SYMBOLS)} current Nifty 50 constituents from NSE.")

    url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    instruments = requests.get(url, timeout=30).json()

    lookup = {
        i["symbol"]: i["token"]
        for i in instruments
        if i.get("exch_seg") == "NSE" and i.get("symbol", "").endswith("-EQ")
    }

    result = []
    missing = []
    for sym in NIFTY_50_SYMBOLS:
        key = f"{sym}-EQ"
        if key in lookup:
            result.append({"exchange": "NSE", "symboltoken": lookup[key], "tradingsymbol": key})
        else:
            missing.append(sym)

    if missing:
        print(f"Warning: could not find tokens for: {missing}")

    print(f"Resolved {len(result)}/{len(NIFTY_50_SYMBOLS)} Nifty 50 constituent tokens.")
    return result


def build_nifty50_volume_sum(smart_api):
    """
    Fetches OHLCV for all Nifty 50 constituent stocks and sums their volume
    per timestamp, as a proxy for "Nifty 50 volume" (the index itself has none).
    """
    constituents = get_nifty50_constituent_tokens()
    combined = None

    for i, sym in enumerate(constituents):
        if i > 0:
            time.sleep(1)  # stay under Angel One's rate limit across 50 requests
        print(f"[{i+1}/{len(constituents)}] Fetching volume for {sym['tradingsymbol']}...")
        try:
            df = fetch_ohlcv(smart_api, sym["exchange"], sym["symboltoken"])
        except Exception as e:
            print(f"  Skipping {sym['tradingsymbol']}: {e}")
            continue

        vol = df[["timestamp", "volume"]].rename(columns={"volume": sym["tradingsymbol"]})
        combined = vol if combined is None else combined.merge(vol, on="timestamp", how="outer")

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    stock_cols = [c for c in combined.columns if c != "timestamp"]
    combined["nifty50_volume_sum"] = combined[stock_cols].fillna(0).sum(axis=1)
    return combined



def get_last_2_days_range():
    """
    Angel One expects dates in 'YYYY-MM-DD HH:MM' format.
    Covers the last 2 calendar days plus today (i.e. from 2 days ago at
    market open, through right now), so today's candles are always included.
    """
    to_date = dt.datetime.now()
    from_date = (to_date - dt.timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)
    return (
        from_date.strftime("%Y-%m-%d %H:%M"),
        to_date.strftime("%Y-%m-%d %H:%M"),
    )


def fetch_ohlcv(smart_api, exchange, symboltoken, max_retries=5):
    from_date, to_date = get_last_2_days_range()

    params = {
        "exchange": exchange,
        "symboltoken": symboltoken,
        "interval": INTERVAL,
        "fromdate": from_date,
        "todate": to_date,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = smart_api.getCandleData(params)
        except DataException as e:
            # Angel One returns this when you hit their rate limit (too many
            # requests/second). Back off and retry instead of crashing.
            last_error = e
            wait = attempt * 2  # 2s, 4s, 6s, 8s, 10s
            print(f"  Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
            continue

        if not response or not response.get("status"):
            last_error = RuntimeError(f"Failed to fetch candle data: {response}")
            wait = attempt * 2
            print(f"  Request failed (attempt {attempt}/{max_retries}): {response}. Waiting {wait}s...")
            time.sleep(wait)
            continue

        candles = response["data"]  # list of [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    raise RuntimeError(f"Giving up after {max_retries} attempts. Last error: {last_error}")


def main():
    smart_api = login()
    try:
        symbols = list(SYMBOLS)
        if INCLUDE_NIFTY_FUTURE:
            symbols.append(get_current_nifty_future_token())

        for i, sym in enumerate(symbols):
            if i > 0:
                time.sleep(1)  # space out requests to stay under Angel One's rate limit

            print(f"\nFetching {sym['tradingsymbol']} ({sym['exchange']} / {sym['symboltoken']})...")
            df = fetch_ohlcv(smart_api, sym["exchange"], sym["symboltoken"])
            print(df)

            out_file = f"{sym['tradingsymbol']}_last2days_{INTERVAL}.csv"
            df.to_csv(out_file, index=False)
            print(f"Saved {len(df)} rows to {out_file}")

        # Fetch NIFTY50 index price separately for the final combined output
        index_df = fetch_ohlcv(smart_api, "NSE", "99926000")

        if INCLUDE_NIFTY50_VOLUME_SUM:
            print("\nBuilding Nifty 50 constituent volume sum (this fetches 50 stocks, takes ~1-2 min)...")
            vol_df = build_nifty50_volume_sum(smart_api)
            out_file = f"NIFTY50_volume_sum_{INTERVAL}.csv"
            vol_df.to_csv(out_file, index=False)
            print(f"Saved raw per-stock volume breakdown to {out_file}")

            # Build the final combined file: NIFTY50 OHLC + summed constituent volume
            final = index_df[["timestamp", "open", "high", "low", "close"]].merge(
                vol_df[["timestamp", "nifty50_volume_sum"]], on="timestamp", how="left"
            )
            final = final.rename(columns={"nifty50_volume_sum": "volume"})
            final["volume"] = final["volume"].fillna(0).astype(int)
            final["timestamp"] = final["timestamp"].apply(format_timestamp)

            final_file = f"NIFTY50_final_OHLCV_{INTERVAL}.csv"
            final.to_csv(final_file, index=False)
            print(f"\nFinal combined file saved to {final_file}:")
            print(final.head())
    finally:
        smart_api.terminateSession(CLIENT_ID)


if __name__ == "__main__":
    main()
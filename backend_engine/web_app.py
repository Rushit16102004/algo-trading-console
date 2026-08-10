# PASSKEY: rushit2712
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import os
import pandas as pd
import json
import shutil
import datetime

# Import modular live dryrun manager
import backend_engine.live_dryrun as live_dryrun
from longpine.backtest_runner import run_strategy_backtest, get_strategy_signals_for_chart
from backend_engine.users_db import register_user, verify_user, get_user_by_email

app = FastAPI(title="243A Multi-User Strategy Console")

# Enable CORS and GZIP compression for fast JSON transfers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global variables for caching 108k historical candles and precalculated markers in RAM
CANDLE_DATA_PATH = "backend_engine/old data.csv"
HISTORICAL_CANDLES = []
HISTORICAL_MARKERS = {
    "243A": [],
    "LONGPINE_ZFTF": []
}

def download_local_assets():
    """Downloads the standalone lightweight-charts library to local storage to prevent CDN errors."""
    path = "ui_ux/static/lightweight-charts.js"
    if not os.path.exists(path):
        print("[ASSETS] Downloading lightweight-charts.js locally...")
        try:
            import requests
            os.makedirs("ui_ux/static", exist_ok=True)
            r = requests.get("https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js", timeout=15)
            if r.status_code == 200:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                print("[ASSETS] lightweight-charts.js downloaded successfully.")
            else:
                print(f"[ASSETS] Failed to download CDN asset: HTTP {r.status_code}")
        except Exception as e:
            print(f"[ASSETS] Error downloading CDN asset: {e}")

def load_mixed_csv(path):
    """Parses mixed-column CSV files line-by-line to prevent tokenization errors."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    records = []
    headers = ['timestamp', 'open', 'high', 'low', 'close']
    for line in lines:
        parts = line.strip().split(',')
        if not parts or parts[0] == 'timestamp':
            if parts and parts[0] == 'timestamp':
                headers = parts
            continue
        rec = {headers[i]: parts[i] for i in range(min(len(headers), len(parts)))}
        records.append(rec)
        
    df = pd.DataFrame(records)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
    else:
        df['volume'] = 0.0
    return df

def init_historical_caches():
    """Reads 108,000 candles and calculates their signals once on boot to serve them from RAM instantly."""
    global HISTORICAL_CANDLES
    old_data_path = "backend_engine/old data.csv"
    if not os.path.exists(old_data_path):
        print("[CACHE WARNING] backend_engine/old data.csv not found. RAM caching bypassed.")
        return
        
    print("[CACHE INITIALIZER] Parsing and caching 108,000 candles from backend_engine/old data.csv...")
    df_old = load_mixed_csv(old_data_path)
    df_old['timestamp'] = pd.to_datetime(df_old['timestamp'], format='mixed')
    import pytz
    ist_tz = pytz.timezone("Asia/Kolkata")
    df_old['time_epoch'] = df_old['timestamp'].apply(lambda x: int(ist_tz.localize(x).timestamp()) if x.tzinfo is None else int(x.astimezone(ist_tz).timestamp()))
    
    candles_temp = []
    seen_epochs = set()
    for _, row in df_old.iterrows():
        t = int(row['time_epoch'])
        if t in seen_epochs:
            continue
        seen_epochs.add(t)
        candles_temp.append({
            "time": t,
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row['volume'])
        })
    HISTORICAL_CANDLES = candles_temp
    
    print("[CACHE INITIALIZER] Pre-calculating historical strategy signal markers...")
    HISTORICAL_MARKERS["243A"] = get_strategy_signals_for_chart(df_old, "243A")
    HISTORICAL_MARKERS["LONGPINE_ZFTF"] = get_strategy_signals_for_chart(df_old, "LONGPINE_ZFTF")
    print(f"[CACHE INITIALIZER] Pre-calculation complete! (243A Markers: {len(HISTORICAL_MARKERS['243A'])}, ZFTF Markers: {len(HISTORICAL_MARKERS['LONGPINE_ZFTF'])})")

starting_sessions = set()

def get_user_session(email: str, strategy_name: str = "243A"):
    if not email:
        return None
    user = get_user_by_email(email)
    if not user:
        return None
        
    user_id = user["id"]
    if user_id in starting_sessions:
        # Wait a brief moment if another request is currently initializing this session
        import time
        for _ in range(30):
            if user_id in live_dryrun.active_sessions:
                break
            time.sleep(0.1)
            
    if user_id not in live_dryrun.active_sessions:
        starting_sessions.add(user_id)
        try:
            # Stop developer session if a custom user logs in to release resources
            dev_user = get_user_by_email("developer@gmail.com")
            if dev_user and dev_user["id"] != user_id:
                try:
                    live_dryrun.stop_user_system(dev_user["id"])
                    print(f"[Session Manager] Stopped developer default session for custom user {email}")
                except Exception as e:
                    print(f"[Session Manager] Error stopping dev session: {e}")

            # Auto-start dryrun worker tasks on demand for this user
            credentials = {
                "api_key": user["api_key"],
                "client_id": user["client_id"],
                "password": user["password"],
                "totp_secret": user["totp_secret"]
            }
            live_dryrun.start_user_system(user_id, credentials, strategy_name=strategy_name)
        finally:
            starting_sessions.discard(user_id)
            
    return live_dryrun.active_sessions.get(user_id)

def get_recent_logs(system_log_path: str, num_lines=150):
    if not os.path.exists(system_log_path):
        return ["Session logs initializing... Please wait."]
    try:
        with open(system_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-num_lines:]]
    except Exception as e:
        return [f"Error reading logs: {e}"]

@app.on_event("startup")
async def startup_event():
    # Download required static assets locally
    download_local_assets()
    # Cache historical datasets in-memory
    init_historical_caches()
    
    # Auto-register default developer user if missing
    try:
        from backend_engine.users_db import get_user_by_email, register_user
        if not get_user_by_email("developer@gmail.com"):
            register_user(
                email="developer@gmail.com",
                pin="YOUR_PIN_6DIGIT",
                api_key="YOUR_API_KEY",
                client_id="YOUR_CLIENT_ID",
                password="YOUR_MPIN",
                totp_secret="YOUR_TOTP_SECRET"
            )
            print("[Startup] Auto-registered default developer user.")
    except Exception as e:
        print(f"[Startup] Error auto-registering default developer user: {e}")
        
    # Start the background checker to manage daily resets
    live_dryrun.start_background_system()

@app.on_event("shutdown")
async def shutdown_event():
    live_dryrun.stop_background_system()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    template_path = os.path.join("ui_ux", "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard UI templates/index.html not found.")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/static/js/lightweight-charts.js")
async def get_js():
    path = "ui_ux/static/lightweight-charts.js"
    if not os.path.exists(path):
        download_local_assets()
    if os.path.exists(path):
        return FileResponse(path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Static Lightweight Charts script not available.")

@app.post("/api/auth/register")
async def register(
    email: str = Form(...),
    pin: str = Form(...),
    api_key: str = Form(...),
    client_id: str = Form(...),
    password: str = Form(...),
    totp_secret: str = Form(...)
):
    """Registers a new user and validates their credentials on Angel One first."""
    import pyotp
    from SmartApi import SmartConnect
    
    # 1. Validate credentials with Angel One OpenAPI
    try:
        smart_connect = SmartConnect(api_key=api_key.strip())
        totp = pyotp.TOTP(totp_secret.strip()).now()
        data = smart_connect.generateSession(client_id.strip(), password.strip(), totp)
        if data.get('status') != True:
            err_msg = data.get('message', 'Angel One validation failed.')
            return {"status": "error", "message": f"Credential check failed: {err_msg}"}
    except Exception as e:
        return {"status": "error", "message": f"Angel One validation error: {str(e)}"}
        
    # 2. Register inside database
    success = register_user(
        email=email.strip().lower(),
        pin=pin.strip(),
        api_key=api_key.strip(),
        client_id=client_id.strip(),
        password=password.strip(),
        totp_secret=totp_secret.strip()
    )
    if success:
        return {"status": "success", "message": "Account created successfully! You can now log in using your PIN."}
    return {"status": "error", "message": "An account with this email already exists."}

is_syncing_in_progress = False

async def run_auto_sync_in_background(sc, email, session):
    global is_syncing_in_progress
    if is_syncing_in_progress:
        return
    is_syncing_in_progress = True
    try:
        print("[Background Auto-Sync] Syncing last 72 candles...")
        from backend_engine.live_dryrun import sync_last_72_candles
        success = await asyncio.to_thread(sync_last_72_candles, sc, CANDLE_DATA_PATH, email)
        if success:
            print("[Background Auto-Sync] Sync complete! Re-initializing RAM cache...")
            init_historical_caches()
            if session:
                session.candles_df = pd.read_csv(CANDLE_DATA_PATH)
    except Exception as e:
        print(f"[Background Auto-Sync] Error: {e}")
    finally:
        is_syncing_in_progress = False

@app.get("/api/status")
async def get_status(email: str = Query(None), strategy: str = Query("243A")):
    session = get_user_session(email, strategy_name=strategy)
    if not session:
        return {
            "index_ltp": 0.0,
            "connection_status": "offline",
            "mode": "none",
            "today_realized_pnl": 0.0,
            "today_unrealized_pnl": 0.0,
            "today_total_pnl": 0.0,
            "active_positions": [],
            "current_candle": None,
            "logs": ["Please log in to view status."],
            "sync_in_progress": is_syncing_in_progress
        }
        
    # Update active strategy name dynamically
    if strategy in ["243A", "LONGPINE_ZFTF"]:
        session.strategy_name = strategy

    ltp = session.index_ltp
    conn_status = "offline"
    if session.ws_handler:
        conn_status = session.ws_handler.conn_status.get("status", "offline")
        
    active_positions = []
    realized_pnl_inr = 0.0
    unrealized_pnl_inr = 0.0
    
    # Calculate realized P&L
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    for t in session.paper_trade_engine.trades:
        exit_time_str = t.get("exit_time", "")
        if exit_time_str.startswith(today_str):
            realized_pnl_inr += float(t.get("pnl", 0.0))
            
    # Calculate unrealized positions
    for pos in session.paper_trade_engine.active_positions:
        pos_type = pos.get("position_type")
        entry_nifty = float(pos.get("entry_nifty_price"))
        lots = int(pos.get("option_lots", 1))
        qty = pos.get("lot_size", 65) * lots
        
        pnl_pts = (ltp - entry_nifty) if pos_type == "LONG" else (entry_nifty - ltp)
        pos_pnl_inr = pnl_pts * qty
        unrealized_pnl_inr += pos_pnl_inr
        
        active_positions.append({
            "position_type": pos_type,
            "entry_nifty": entry_nifty,
            "sl_nifty": pos.get("sl_nifty_price"),
            "tp_nifty": pos.get("tp_nifty_price"),
            "lots": lots,
            "pnl_points": round(pnl_pts, 2),
            "pnl": round(pos_pnl_inr, 2),
            "entry_time": pos.get("entry_time"),
            "entry_reason": pos.get("entry_reason"),
        })
        
    recent_logs = get_recent_logs(session.trade_logger.system_log_path)
    
    # Expose running incomplete candle
    current_candle = None
    c_builder = session.candle_builder
    if c_builder and ltp > 0:
        bucket = c_builder.index_bucket
        if bucket and bucket in c_builder.index_ohlc:
            ohlc = c_builder.index_ohlc[bucket]
            import pytz
            ist_tz = pytz.timezone("Asia/Kolkata")
            localized_bucket = ist_tz.localize(bucket) if bucket.tzinfo is None else bucket.astimezone(ist_tz)
            current_candle = {
                "time": int(localized_bucket.timestamp()),
                "open": float(ohlc["open"]),
                "high": float(ohlc["high"]),
                "low": float(ohlc["low"]),
                "close": float(ltp)
            }
            
    return {
        "index_ltp": ltp,
        "connection_status": conn_status,
        "mode": "angel" if session.credentials.get("api_key") else "custom",
        "today_realized_pnl": round(realized_pnl_inr, 2),
        "today_unrealized_pnl": round(unrealized_pnl_inr, 2),
        "today_total_pnl": round(realized_pnl_inr + unrealized_pnl_inr, 2),
        "active_positions": active_positions,
        "current_candle": current_candle,
        "logs": recent_logs,
        "sync_in_progress": is_syncing_in_progress
    }

@app.get("/api/signals")
async def get_signals(email: str = Query(None)):
    session = get_user_session(email)
    if not session:
        return {"current_regime": "Unknown", "signal_history": []}
        
    regime = "Unknown"
    signal_log_path = session.trade_logger.signal_log_path
    if os.path.exists(signal_log_path):
        try:
            df = pd.read_csv(signal_log_path)
            if not df.empty:
                regime = df.iloc[-1].get("hmm_regime", "Unknown")
        except Exception:
            pass
            
    trade_history = []
    trade_log_path = session.trade_logger.trade_log_path
    if os.path.exists(trade_log_path):
        try:
            df_trades = pd.read_csv(trade_log_path)
            if not df_trades.empty:
                for _, row in df_trades.tail(100).iterrows():
                    trade_history.append({
                        "signal_type": row.get("signal"),
                        "entry_time": row.get("entry_time"),
                        "exit_time": row.get("exit_time") if not pd.isna(row.get("exit_time")) else None,
                        "entry_nifty": float(row.get("entry_nifty", 0)),
                        "exit_nifty": float(row.get("exit_nifty", 0)) if not pd.isna(row.get("exit_nifty")) else None,
                        "lot_size": int(row.get("lot_size", 65)) // 65,
                        "pnl_points": float(row.get("nifty_pnl_points", 0)) if not pd.isna(row.get("nifty_pnl_points")) else 0,
                        "pnl": float(row.get("nifty_pnl_inr", 0)) if not pd.isna(row.get("nifty_pnl_inr")) else 0,
                        "exit_reason": row.get("exit_reason") if not pd.isna(row.get("exit_reason")) else ""
                    })
        except Exception:
            pass
            
    trade_history.reverse()
    return {
        "current_regime": regime,
        "signal_history": trade_history
    }

@app.get("/api/sync_72")
async def api_sync_72(email: str = Query(None)):
    global is_syncing_in_progress
    if is_syncing_in_progress:
        return {"status": "error", "message": "A synchronization is already running. Please wait."}
        
    session = get_user_session(email)
    sc = session.smart_connect if session else None
    
    is_syncing_in_progress = True
    try:
        from backend_engine.live_dryrun import sync_last_72_candles
        success = await asyncio.to_thread(sync_last_72_candles, sc, CANDLE_DATA_PATH, email)
        if success:
            init_historical_caches()
            if session:
                session.candles_df = pd.read_csv(CANDLE_DATA_PATH)
            return {"status": "success", "message": "Successfully synchronized last 72 candles and updated signals."}
    except Exception as e:
        print(f"Manual sync error: {e}")
    finally:
        is_syncing_in_progress = False
        
    return {"status": "error", "message": "Failed to sync candles. Please check credentials or try again later."}

@app.get("/api/candles")
async def get_candles(email: str = Query(None), strategy: str = Query("243A")):
    """
    Returns a unified, continuous dataset of Nifty candles.
    Uses RAM-cached historical candles (2008-2025) to achieve sub-5ms response times.
    """
    try:
        live_candles = []
        live_markers = []
        
        session = get_user_session(email)
        
        # Check and sync missing candles automatically in the background to avoid page blocking!
        from backend_engine.live_dryrun import check_and_sync_missing
        sc = session.smart_connect if session else None
        should_sync = await asyncio.to_thread(check_and_sync_missing, CANDLE_DATA_PATH)
        if should_sync and not is_syncing_in_progress:
            asyncio.create_task(run_auto_sync_in_background(sc, email, session))
        
        if session and session.candles_df is not None and not session.candles_df.empty:
            df_live = session.candles_df.copy()
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], format='mixed')
            import pytz
            ist_tz = pytz.timezone("Asia/Kolkata")
            df_live['time_epoch'] = df_live['timestamp'].apply(lambda x: int(ist_tz.localize(x).timestamp()) if x.tzinfo is None else int(x.astimezone(ist_tz).timestamp()))
            
            seen_epochs = set()
            for _, row in df_live.iterrows():
                t = int(row['time_epoch'])
                if t in seen_epochs:
                    continue
                seen_epochs.add(t)
                live_candles.append({
                    "time": t,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']) if not pd.isna(row['volume']) else 0.0
                })
                
            # Precompute strategy signals ONLY on the live portion of the dataset (fast!)
            live_markers = get_strategy_signals_for_chart(df_live, strategy)
            
        # Combine RAM-cached history with live portion and deduplicate/sort strictly
        combined_candles_map = {}
        for c in HISTORICAL_CANDLES:
            combined_candles_map[c["time"]] = c
        for c in live_candles:
            combined_candles_map[c["time"]] = c
            
        candles = [combined_candles_map[t] for t in sorted(combined_candles_map.keys())]
        
        combined_markers_map = {}
        for m in HISTORICAL_MARKERS.get(strategy, []):
            combined_markers_map[m["time"]] = m
        for m in live_markers:
            combined_markers_map[m["time"]] = m
            
        markers = [combined_markers_map[t] for t in sorted(combined_markers_map.keys())]
        
        return {
            "candles": candles,
            "markers": markers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error merging unified dataset: {e}")

@app.post("/api/backtest")
async def post_backtest(
    strategy: str = Form("243A"),
    file: UploadFile = File(...)
):
    """Runs a backtest simulation on the uploaded file for the selected strategy."""
    temp_input = "backend_engine/logs_dryrun/temp_backtest_input.csv"
    temp_output = "backend_engine/logs_dryrun/backtest_results_web.csv"
    
    os.makedirs("backend_engine/logs_dryrun", exist_ok=True)
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        metrics = run_strategy_backtest(pd.read_csv(temp_input), strategy, temp_output)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        return {"status": "error", "message": f"Backtest execution error: {str(e)}"}

@app.get("/api/download/candles")
async def download_candles(email: str = Query(None)):
    session = get_user_session(email)
    if session and os.path.exists(session.candle_builder.candle_data_path):
        return FileResponse(session.candle_builder.candle_data_path, filename="live_nifty50_candles.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Candle data file not found.")

@app.get("/api/download/signals")
async def download_signals(email: str = Query(None)):
    session = get_user_session(email)
    if session and os.path.exists(session.trade_logger.signal_log_path):
        return FileResponse(session.trade_logger.signal_log_path, filename="live_signals_log.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Signal log file not found.")

@app.get("/api/download/trades")
async def download_trades(email: str = Query(None)):
    session = get_user_session(email)
    if session and os.path.exists(session.trade_logger.trade_log_path):
        return FileResponse(session.trade_logger.trade_log_path, filename="live_trades_ledger_nifty.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Trade log file not found.")

@app.get("/api/download/backtest")
async def download_backtest():
    path = "backend_engine/logs_dryrun/backtest_results_web.csv"
    if os.path.exists(path):
        return FileResponse(path, filename="backtest_results.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Backtest results file not found.")

@app.get("/api/download/old_data")
async def download_old_data(year: str = Query("all")):
    path = "backend_engine/old data.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Historical old data file not found.")
        
    if year == "all":
        return FileResponse(path, filename="historical_old_data_2008_2026.csv", media_type="text/csv")
        
    try:
        df = load_mixed_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df_filtered = df[df['timestamp'].dt.year == int(year)]
        
        import io
        from fastapi.responses import StreamingResponse
        stream = io.StringIO()
        df_filtered.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=historical_old_data_{year}.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to filter old data by year {year}: {e}")

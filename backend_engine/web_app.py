# PASSKEY: rushit2712
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import os
import pandas as pd
import json
import shutil
import datetime
import uuid
import time
from collections import defaultdict

# Import modular live dryrun manager
import backend_engine.live_dryrun as live_dryrun
from longpine.backtest_runner import run_strategy_backtest, get_strategy_signals_for_chart
from backend_engine.users_db import register_user, verify_user, get_user_by_email
from backend_engine.config import ALLOWED_ORIGINS

app = FastAPI(title="243A Multi-User Strategy Console")

# Enable CORS and GZIP compression for fast JSON transfers
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global Security Headers & Request ID middleware
@app.middleware("http")
async def security_headers_and_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com "
            "https://fonts.googleapis.com https://fonts.gstatic.com ws: wss:; object-src 'none';"
        )
        return response
    except Exception as e:
        print(f"[ERROR-LOGGER] Request {request_id} failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"An internal server error occurred. Request ID: {request_id}"}
        )

# Simple in-memory rate-limiter
RATE_LIMIT_WINDOWS = defaultdict(list)

def check_rate_limit(client_ip: str, limit: int = 10, window: int = 60) -> bool:
    now = time.time()
    RATE_LIMIT_WINDOWS[client_ip] = [t for t in RATE_LIMIT_WINDOWS[client_ip] if now - t < window]
    if len(RATE_LIMIT_WINDOWS[client_ip]) >= limit:
        return False
    RATE_LIMIT_WINDOWS[client_ip].append(now)
    return True

# Global variables for caching 108k historical candles and precalculated markers in RAM
CANDLE_DATA_PATH = "backend_engine/old data.csv"
HISTORICAL_CANDLES = []
HISTORICAL_MARKERS = {
    "243A": [],
    "LONGPING": []
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
    
    try:
        # If timezone-aware:
        epochs = (df_old['timestamp'].dt.tz_convert('Asia/Kolkata').astype('int64') // 10**9).tolist()
    except TypeError:
        # If naive (default):
        epochs = (df_old['timestamp'].dt.tz_localize('Asia/Kolkata').astype('int64') // 10**9).tolist()
        
    df_old['time_epoch'] = epochs
    
    # Deduplicate epochs
    df_unique = df_old.drop_duplicates(subset=['time_epoch']).copy()
    df_unique = df_unique.rename(columns={'time_epoch': 'time'})
    
    # Fast convert to records dictionary list
    HISTORICAL_CANDLES = df_unique[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
    
    print("[CACHE INITIALIZER] Pre-calculating historical strategy signal markers...")
    HISTORICAL_MARKERS["243A"] = get_strategy_signals_for_chart(df_old, "243A")
    HISTORICAL_MARKERS["LONGPING"] = get_strategy_signals_for_chart(df_old, "LONGPING")
    print(f"[CACHE INITIALIZER] Pre-calculation complete! (243A Markers: {len(HISTORICAL_MARKERS['243A'])}, LONGPING Markers: {len(HISTORICAL_MARKERS['LONGPING'])})")

starting_sessions = set()

def get_user_session(email: str, strategy_name: str = "243A"):
    if not email:
        return None
        
    admin_user_id = 1
    if admin_user_id not in live_dryrun.active_sessions:
        admin_api_key = os.getenv("ANGEL_API_KEY")
        admin_client_id = os.getenv("ANGEL_CLIENT_ID")
        admin_password = os.getenv("ANGEL_PASSWORD")
        admin_totp_secret = os.getenv("ANGEL_TOTP_SECRET")
        
        credentials = {
            "email": "admin@algo-trading.console",
            "api_key": admin_api_key,
            "client_id": admin_client_id,
            "password": admin_password,
            "totp_secret": admin_totp_secret
        }
        print(f"[Session Manager] Starting central admin session for user {email}")
        live_dryrun.start_user_system(admin_user_id, credentials, strategy_name=strategy_name)
        
    return live_dryrun.active_sessions.get(admin_user_id)

def get_recent_logs(system_log_path: str, num_lines=150):
    if not os.path.exists(system_log_path):
        return ["Session logs initializing... Please wait."]
    try:
        with open(system_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-num_lines:]]
    except Exception as e:
        return [f"Error reading logs: {e}"]

# NSE HOLIDAYS 2026
HOLIDAYS = {
    "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
    "2026-12-25"
}

async def market_hours_scheduler_loop():
    """
    Background loop that:
    - Starts the central feed automatically at 9:00 AM IST.
    - Stops the central feed automatically at 4:00 PM IST (16:00).
    - Skips weekends and holidays.
    """
    import pytz
    ist_tz = pytz.timezone("Asia/Kolkata")
    admin_user_id = 1
    
    while True:
        try:
            now = datetime.datetime.now(ist_tz)
            today_str = now.strftime("%Y-%m-%d")
            
            # Monday-Friday and not an NSE holiday
            is_trading_day = now.weekday() < 5 and today_str not in HOLIDAYS
            
            # Start at 9:00 AM and stop at 4:00 PM (16:00)
            is_market_hours = is_trading_day and (datetime.time(9, 0) <= now.time() <= datetime.time(16, 0))
            
            if is_market_hours:
                if admin_user_id not in live_dryrun.active_sessions:
                    print(f"[Scheduler] Market is open ({now.strftime('%H:%M:%S')}). Auto-starting central trading feed...")
                    # Start the central session (credentials fallback to env inside start_user_system)
                    live_dryrun.start_user_system(admin_user_id, {}, strategy_name="243A")
            else:
                if admin_user_id in live_dryrun.active_sessions:
                    print(f"[Scheduler] Market is closed ({now.strftime('%H:%M:%S')}). Auto-stopping central trading feed...")
                    live_dryrun.stop_user_system(admin_user_id)
        except Exception as e:
            print(f"[Scheduler Error] {e}")
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    # Download required static assets locally
    download_local_assets()
    # Cache historical datasets in-memory
    init_historical_caches()
    
    # Auto-register default demo and developer users if missing
    try:
        from backend_engine.users_db import get_user_by_email, register_user
        if not get_user_by_email("demo@gmail.com"):
            register_user(
                email="demo@gmail.com",
                pin="111111",
                api_key="",
                client_id="",
                password="",
                totp_secret=""
            )
            print("[Startup] Auto-registered default demo user: demo@gmail.com / PIN: 111111")
        if not get_user_by_email("developer@gmail.com"):
            register_user(
                email="developer@gmail.com",
                pin="111111",
                api_key="",
                client_id="",
                password="",
                totp_secret=""
            )
            print("[Startup] Auto-registered developer user: developer@gmail.com / PIN: 111111")
    except Exception as e:
        print(f"[Startup] Error auto-registering default users: {e}")
        
    # Start the automated market hours scheduler task
    asyncio.create_task(market_hours_scheduler_loop())

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

@app.get("/api/auth/exists")
async def check_auth_exists():
    from backend_engine.users_db import check_any_custom_user_exists
    return {"exists": check_any_custom_user_exists()}

@app.post("/api/auth/register")
async def register(
    request: Request,
    email: str = Form(...),
    pin: str = Form(...),
    api_key: str = Form(""),
    client_id: str = Form(""),
    password: str = Form(""),
    totp_secret: str = Form("")
):
    """Registers a new user inside the local database sandbox."""
    client_ip = request.client.host
    if not check_rate_limit(client_ip, limit=3, window=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
        
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

@app.post("/api/auth/login")
async def login(
    request: Request,
    email: str = Form(...),
    pin: str = Form(...)
):
    """Logs in an existing user and returns an access token."""
    client_ip = request.client.host
    if not check_rate_limit(client_ip, limit=5, window=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
        
    user_data = verify_user(email.strip().lower(), pin.strip())
    if not user_data:
        return {"status": "error", "message": "Invalid email or PIN."}
        
    from backend_engine.auth import create_access_token
    token = create_access_token(data={"sub": email.strip().lower()})
    
    from backend_engine.audit_logger import log_audit_event
    log_audit_event(user_data["id"], "LOGIN", f"User logged in from IP: {client_ip}")
    
    return {"status": "success", "token": token, "message": "Login successful!"}

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
    if strategy in ["243A", "LONGPING"]:
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
            
    from backend_engine.kill_switch import get_kill_switch_state
    from backend_engine.config import DEMO_MODE, TRADING_MODE
    
    return {
        "index_ltp": ltp,
        "connection_status": conn_status,
        "mode": "DEMO" if DEMO_MODE else TRADING_MODE,
        "kill_switch_active": get_kill_switch_state(),
        "trades_count": len([t for t in session.paper_trade_engine.trades if t.get("exit_time", "").startswith(today_str)]),
        "today_realized_pnl": round(realized_pnl_inr, 2),
        "today_unrealized_pnl": round(unrealized_pnl_inr, 2),
        "today_total_pnl": round(realized_pnl_inr + unrealized_pnl_inr, 2),
        "active_positions": active_positions,
        "current_candle": current_candle,
        "logs": recent_logs,
        "sync_in_progress": is_syncing_in_progress or getattr(session, 'sync_in_progress', False) or getattr(session, 'warmup_in_progress', False),
        "warmup_in_progress": getattr(session, 'warmup_in_progress', False),
        "warmup_progress": getattr(session, 'warmup_progress', "0/0"),
        "warmup_time_remaining": getattr(session, 'warmup_time_remaining', 0)
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
        
        # Limit to the most recent 3,000 candles to achieve instant page loads (< 300KB payload)
        limit = 3000
        if len(candles) > limit:
            min_allowed_time = candles[-limit]["time"]
            candles = candles[-limit:]
            markers = [m for m in markers if m["time"] >= min_allowed_time]
        
        return {
            "candles": candles,
            "markers": markers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error merging unified dataset: {e}")

@app.get("/api/download/candles")
async def download_candles(email: str = Query(None)):
    session = get_user_session(email)
    path = session.candle_builder.candle_data_path if session else "backend_engine/old data.csv"
    if os.path.exists(path):
        try:
            import io
            from fastapi.responses import StreamingResponse
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            today = datetime.date.today()
            df_today = df[df['timestamp'].dt.date == today]
            
            # Fallback to the latest available day if today has no candles (e.g. weekend/holidays)
            if df_today.empty and not df.empty:
                latest_date = df['timestamp'].dt.date.max()
                df_today = df[df['timestamp'].dt.date == latest_date]
                
            stream = io.StringIO()
            df_today.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = f"attachment; filename=live_nifty50_candles_today.csv"
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error filtering candles: {e}")
    raise HTTPException(status_code=404, detail="Candle data file not found.")

@app.get("/api/download/signals")
async def download_signals(email: str = Query(None)):
    session = get_user_session(email)
    path = session.trade_logger.signal_log_path if session else "data/logs_dryrun/signal_log.csv"
    if os.path.exists(path):
        try:
            import io
            from fastapi.responses import StreamingResponse
            df_sig = pd.read_csv(path)
            
            # Rename columns to match requested format
            df_sig = df_sig.rename(columns={
                "datetime": "timestamp",
                "open": "open",
                "gbm_signal": "LIGHTGBM",
                "tcn_signal": "TCN",
                "hmm_regime": "HMM",
                "final_entry_signal": "final signal"
            })
            
            # Keep only the requested columns
            cols_to_keep = ["timestamp", "open", "LIGHTGBM", "TCN", "HMM", "final signal"]
            df_sig = df_sig[[c for c in cols_to_keep if c in df_sig.columns]]
            
            # Filter for today's signals only
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            df_filtered = df_sig[df_sig['timestamp'].astype(str).str.startswith(today_str)]
            
            # Fallback to the latest available day if today has no signals
            if df_filtered.empty and not df_sig.empty:
                try:
                    df_sig['date_only'] = pd.to_datetime(df_sig['timestamp'], format='mixed').dt.date
                    latest_date = df_sig['date_only'].max()
                    df_filtered = df_sig[df_sig['date_only'] == latest_date].copy()
                    df_filtered = df_filtered.drop(columns=['date_only'])
                except Exception:
                    df_filtered = df_sig
            
            stream = io.StringIO()
            df_filtered.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename=live_signals_log_today.csv"
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error filtering signals: {e}")
    raise HTTPException(status_code=404, detail="Signal log file not found.")

@app.get("/api/download/trades")
async def download_trades(email: str = Query(None), strategy: str = Query("243A")):
    # Select path based on strategy to guarantee we always have completed trade history
    if strategy == "243A":
        path = "backend_engine/model signal.csv"
    else:
        path = "model_2024_25/backtest_results_longping.csv"
        
    if os.path.exists(path):
        try:
            import io
            from fastapi.responses import StreamingResponse
            df = pd.read_csv(path)
            if not df.empty:
                time_col = 'Entry Time' if 'Entry Time' in df.columns else 'entry_time'
                df['entry_time_dt'] = pd.to_datetime(df[time_col], format='mixed')
                latest = df['entry_time_dt'].max()
                cutoff = latest - pd.Timedelta(days=5)
                df_filtered = df[df['entry_time_dt'] >= cutoff].copy()
                df_filtered = df_filtered.drop(columns=['entry_time_dt'])
            else:
                df_filtered = df
                
            stream = io.StringIO()
            df_filtered.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = f"attachment; filename=live_trades_last_5_days_{strategy}.csv"
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error filtering trades: {e}")
    raise HTTPException(status_code=404, detail="Trade log file not found.")

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

@app.get("/api/health")
async def health_check():
    """System diagnostic health endpoint."""
    from backend_engine.database import SessionLocal
    from backend_engine.health_monitor import monitor as health_monitor
    from backend_engine.kill_switch import get_kill_switch_state
    from backend_engine.config import DEMO_MODE, TRADING_MODE
    import sqlalchemy
    
    db_ok = False
    db = SessionLocal()
    try:
        db.execute(sqlalchemy.text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    finally:
        db.close()
        
    ws_status = health_monitor.get_status()
    ks_active = get_kill_switch_state()
    
    status = "healthy"
    if not db_ok or (not DEMO_MODE and ws_status.get("stale", True)):
        status = "unhealthy"
        
    return {
        "status": status,
        "database": "connected" if db_ok else "disconnected",
        "market_feed": "connected" if ws_status.get("connected") else "disconnected",
        "feed_stale": ws_status.get("stale"),
        "kill_switch_active": ks_active,
        "trading_mode": "DEMO" if DEMO_MODE else TRADING_MODE,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/api/killswitch/toggle")
async def toggle_killswitch(active: bool = Form(...)):
    """Triggers or resets the emergency halt switch."""
    from backend_engine.kill_switch import set_kill_switch_state
    set_kill_switch_state(active)
    return {"status": "success", "kill_switch_active": active}

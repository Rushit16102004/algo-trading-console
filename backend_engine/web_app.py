# PASSKEY: rushit2712
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request, Depends
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
from backend_engine.auth import get_current_user
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
# Cached historical dataframe for pattern matching (loaded once, reused forever)
PATTERN_DF_CACHE = None

# HMM regime cone state: tracks last regime + anchor candle per user session
HMM_CONE_STATE = {}  # key: email -> {"regime": str, "anchor_time": int, "anchor_price": float}

# Drift and volatility parameters per HMM regime (per 5-minute candle)
HMM_REGIME_PARAMS = {
    "markup":           {"drift": +0.00015, "vol": 0.00040},
    "expansionup":      {"drift": +0.00008, "vol": 0.00025},
    "distributiondown": {"drift": +0.00002, "vol": 0.00035},
    "compression":      {"drift":  0.00000, "vol": 0.00015},
    "distributionup":   {"drift": -0.00002, "vol": 0.00035},
    "expansiondown":    {"drift": -0.00008, "vol": 0.00025},
    "markdown":         {"drift": -0.00015, "vol": 0.00040},
}

# -------------------------------------------------------------------
# Pattern Library: daily LightGBM feature fingerprints (self-growing)
# -------------------------------------------------------------------
PATTERN_LIBRARY = {}   # {"YYYY-MM-DD": {"vector": np.array, "eod_change": float, "day_range": float}}
PATTERN_LIBRARY_LOCK = False   # prevent concurrent rebuild

FEATURE_COLS_PATTERN = [
    "atr_ratio", "roc_10", "displacement", "momentum_disp",
    "volume_expansion", "compression", "dist_to_swing_high", "dist_to_swing_low",
    "rsi_14", "rsi_7", "macdhist_norm", "stoch_k", "stoch_d",
    "cci_14", "willr_14", "sma_50_diff", "sma_200_diff",
    "bb_upper_diff", "bb_lower_diff", "bb_width", "realized_vol_20"
]

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
    """Reads recent historical candles and precalculates markers instantly on boot."""
    global HISTORICAL_CANDLES
    old_data_path = "backend_engine/old data.csv"
    if not os.path.exists(old_data_path):
        print("[CACHE WARNING] backend_engine/old data.csv not found. RAM caching bypassed.")
        return
        
    print("[CACHE INITIALIZER] Fast parsing and caching candles from backend_engine/old data.csv...")
    df_old = pd.read_csv(old_data_path)
    df_old['open'] = pd.to_numeric(df_old['open'], errors='coerce')
    df_old['high'] = pd.to_numeric(df_old['high'], errors='coerce')
    df_old['low'] = pd.to_numeric(df_old['low'], errors='coerce')
    df_old['close'] = pd.to_numeric(df_old['close'], errors='coerce')
    if 'volume' in df_old.columns:
        df_old['volume'] = pd.to_numeric(df_old['volume'], errors='coerce').fillna(0.0)
    else:
        df_old['volume'] = 0.0

    df_old['timestamp'] = pd.to_datetime(df_old['timestamp'], format='mixed')
    
    df_tail = df_old.tail(5000).copy()
    try:
        epochs = (df_tail['timestamp'].dt.tz_convert('Asia/Kolkata').astype('int64') // 10**9).tolist()
    except TypeError:
        epochs = (df_tail['timestamp'].dt.tz_localize('Asia/Kolkata').astype('int64') // 10**9).tolist()
        
    df_tail['time_epoch'] = epochs
    df_unique = df_tail.drop_duplicates(subset=['time_epoch']).rename(columns={'time_epoch': 'time'})
    HISTORICAL_CANDLES = df_unique[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
    
    print("[CACHE INITIALIZER] Pre-calculating historical strategy signal markers...")
    df_recent = df_old.tail(2000).copy()
    HISTORICAL_MARKERS["243A"] = get_strategy_signals_for_chart(df_recent, "243A")
    HISTORICAL_MARKERS["LONGPING"] = get_strategy_signals_for_chart(df_recent, "LONGPING")
    print(f"[CACHE INITIALIZER] Pre-calculation complete! (243A Markers: {len(HISTORICAL_MARKERS['243A'])}, LONGPING Markers: {len(HISTORICAL_MARKERS['LONGPING'])})")

starting_sessions = set()

def get_user_session(email: str, strategy_name: str = "243A"):
    if not email:
        return None
        
    admin_user_id = 1
    if admin_user_id not in live_dryrun.active_sessions:
        print(f"[Session Manager] Starting 24/7 central Angel One feed session for user {email}")
        live_dryrun.start_user_system(admin_user_id, strategy_name=strategy_name)
            
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
    Background loop that keeps the central feed always connected.
    Starts the feed on app startup and restarts it if it ever drops.
    End-of-day pattern library update still runs at 15:32 IST.
    """
    import pytz
    ist_tz = pytz.timezone("Asia/Kolkata")
    admin_user_id = 1
    
    while True:
        try:
            now = datetime.datetime.now(ist_tz)
            today_str = now.strftime("%Y-%m-%d")

            # Always keep central feed running — restart if it dropped
            if admin_user_id not in live_dryrun.active_sessions:
                print(f"[Scheduler] Central feed not running. Auto-starting 24/7 central trading feed...")
                live_dryrun.start_user_system(admin_user_id, strategy_name="243A")

            # End-of-day: add today to pattern library at 15:32 IST
            if now.weekday() < 5 and now.hour == 15 and now.minute == 32:
                import threading
                threading.Thread(target=update_pattern_library_today, daemon=True).start()

        except Exception as e:
            print(f"[Scheduler Error] {e}")
        await asyncio.sleep(30)

def precompute_pattern_library():
    """Load historical 2024-2026 data and build daily feature fingerprint library."""
    global PATTERN_LIBRARY, PATTERN_LIBRARY_LOCK, PATTERN_DF_CACHE
    if PATTERN_LIBRARY_LOCK:
        return
    PATTERN_LIBRARY_LOCK = True
    try:
        import importlib
        extract_gbm_features = importlib.import_module("243A.live_prediction_engine").extract_gbm_features
        # Load CSV only once and cache it
        if PATTERN_DF_CACHE is None:
            df_raw = pd.read_csv(CANDLE_DATA_PATH)
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], format="mixed")
            PATTERN_DF_CACHE = df_raw
        df_all = PATTERN_DF_CACHE.copy()
        # Filter to 2020-2026 (post-COVID modern market regime)
        df_all = df_all[(df_all["timestamp"].dt.year >= 2020) & (df_all["timestamp"].dt.year <= 2026)]
        df_all = df_all.set_index("timestamp").sort_index()
        dates = df_all.index.normalize().unique()
        built = 0
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            if date_str in PATTERN_LIBRARY:
                continue
            day_df = df_all[df_all.index.normalize() == date].copy()
            if len(day_df) < 30:
                continue
            try:
                day_df_reset = day_df.reset_index()
                day_df_reset = day_df_reset.rename(columns={"timestamp": "timestamp"})
                feats = extract_gbm_features(day_df_reset)
                vec_df = feats[FEATURE_COLS_PATTERN].dropna()
                if len(vec_df) < 10:
                    continue
                vec = vec_df.mean().values
                eod_change = float(day_df["close"].iloc[-1] - day_df["open"].iloc[0])
                day_range = float(day_df["high"].max() - day_df["low"].min())
                PATTERN_LIBRARY[date_str] = {
                    "vector": vec,
                    "eod_change": round(eod_change, 2),
                    "day_range": round(day_range, 2)
                }
                built += 1
            except Exception:
                continue
        print(f"[PatternLib] Built {built} new fingerprints. Total library: {len(PATTERN_LIBRARY)} days.")
    except Exception as e:
        print(f"[PatternLib] Error building library: {e}")
    finally:
        PATTERN_LIBRARY_LOCK = False


def update_pattern_library_today():
    """Called at end of trading day to add today's fingerprint to the library."""
    try:
        import datetime as _dt
        import pytz
        today_str = _dt.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d")
        if today_str in PATTERN_LIBRARY:
            return   # already added
        precompute_pattern_library()
    except Exception as e:
        print(f"[PatternLib] Error updating today: {e}")


@app.on_event("startup")
async def startup_event():
    # Download required static assets locally
    download_local_assets()
    # Cache historical datasets in-memory
    init_historical_caches()
    # Build pattern fingerprint library in background thread — delayed 60s so app starts fast
    import threading, time as _time
    def _delayed_precompute():
        _time.sleep(60)  # Wait 60 seconds after startup before heavy computation
        precompute_pattern_library()
    threading.Thread(target=_delayed_precompute, daemon=True).start()
    
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
):
    """Registers a user without accepting broker credentials."""
    client_ip = request.client.host
    if not check_rate_limit(client_ip, limit=3, window=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
        
    success = register_user(
        email=email.strip().lower(),
        pin=pin.strip(),
        api_key="",
        client_id="",
        password="",
        totp_secret=""
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
async def get_status(email: str = Query(None), strategy: str = Query("243A"), current_user=Depends(get_current_user)):
    if not current_user or current_user.email != (email or "").strip().lower():
        raise HTTPException(status_code=401, detail="Authentication required")
    # First check if the email parameter is provided
    if not email:
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
            "server_down": True,
            "sync_in_progress": is_syncing_in_progress
        }

    session = get_user_session(email, strategy_name=strategy)
    if not session:
        email_clean = email.strip().lower()
        is_valid_user = get_user_by_email(email_clean) is not None
        
        if is_valid_user:
            return {
                "index_ltp": 0.0,
                "connection_status": "offline",
                "mode": "none",
                "today_realized_pnl": 0.0,
                "today_unrealized_pnl": 0.0,
                "today_total_pnl": 0.0,
                "active_positions": [],
                "current_candle": None,
                "logs": ["Trading system is inactive. Market is closed."],
                "server_down": True,
                "sync_in_progress": is_syncing_in_progress
            }
        else:
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
                "server_down": True,
                "sync_in_progress": is_syncing_in_progress
            }
        
    # Update active strategy name dynamically
    if strategy in ["243A", "LONGPING"]:
        session.strategy_name = strategy

    ltp = session.index_ltp
    conn_status = "offline"
    if session and session.ws_handler:
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
    from backend_engine.config import TRADING_MODE
    from backend_engine.health_monitor import monitor as health_monitor
    import math
    
    # ----------------------------------------------------------------
    # HMM Volatility Cone: compute upper/lower bands anchored at the
    # candle where the regime last changed. Bands stay frozen until
    # the regime changes again.
    # ----------------------------------------------------------------
    last_prediction = getattr(session, 'last_prediction', {})
    hmm_regime_raw = last_prediction.get('hmm_regime', 'unknown') if last_prediction else 'unknown'
    hmm_regime_key = str(hmm_regime_raw).lower().replace(' ', '').replace('_', '')
    
    cone_state = HMM_CONE_STATE.get(email, {})
    prev_regime = cone_state.get('regime', None)
    
    # Detect regime change and record new anchor point
    if hmm_regime_key not in ('unknown', '') and hmm_regime_key != prev_regime and current_candle is not None:
        HMM_CONE_STATE[email] = {
            'regime': hmm_regime_key,
            'anchor_time': current_candle['time'],
            'anchor_price': current_candle['close']
        }
        cone_state = HMM_CONE_STATE[email]
    
    # Build cone data from the stored anchor (even if regime hasn't changed this tick)
    hmm_upper = []
    hmm_lower = []
    if cone_state and current_candle is not None:
        regime_params = HMM_REGIME_PARAMS.get(cone_state.get('regime', ''), {"drift": 0.0, "vol": 0.00015})
        drift = regime_params['drift']
        vol   = regime_params['vol']
        anchor_time  = cone_state['anchor_time']
        anchor_price = cone_state['anchor_price']
        for k in range(1, 11):  # project 10 candles = 50 minutes
            t_k = anchor_time + k * 300
            p_k = anchor_price * (1.0 + drift * k)
            band_k = anchor_price * vol * math.sqrt(k)
            hmm_upper.append({"time": t_k, "value": round(p_k + band_k, 2)})
            hmm_lower.append({"time": t_k, "value": round(p_k - band_k, 2)})
    
    feed_age = (time.time() - health_monitor.last_tick_time) if health_monitor.last_tick_time else float("inf")
    feed_live = conn_status == "live" and bool(session.latest_tick) and feed_age <= 30
    return {
        "index_ltp": ltp,
        "connection_status": "live" if feed_live else ("connecting" if conn_status == "connecting" else "offline"),
        "server_down": not feed_live,
        "mode": "LIVE",
        "last_tick_at": session.latest_tick.get("timestamp") if session.latest_tick else None,
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
        "warmup_time_remaining": getattr(session, 'warmup_time_remaining', 0),
        "last_prediction": last_prediction,
        "hmm_upper": hmm_upper,
        "hmm_lower": hmm_lower,
        "hmm_regime": hmm_regime_key
    }

@app.get("/api/signals")
async def get_signals(email: str = Query(None), current_user=Depends(get_current_user)):
    if not current_user or current_user.email != (email or "").strip().lower():
        raise HTTPException(status_code=401, detail="Authentication required")
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

@app.get("/api/pattern_match")
async def pattern_match_endpoint(email: str = Query(None)):
    """
    Finds the top 3 historical trading days (2024-2026) whose LightGBM
    feature fingerprint is most similar to today's live session.
    Uses cosine similarity on 21-feature mean vectors.
    """
    import numpy as np
    if not PATTERN_LIBRARY:
        return {"matches": [], "status": "building", "library_size": 0}

    # Get today's candles from the historical CSV (live fallback)
    try:
        import importlib
        extract_gbm_features = importlib.import_module("243A.live_prediction_engine").extract_gbm_features
        # Use cached dataframe — never re-read the CSV on each request
        if PATTERN_DF_CACHE is None:
            return {"matches": [], "status": "building", "library_size": len(PATTERN_LIBRARY)}
        df_all = PATTERN_DF_CACHE
        import pytz, datetime as _dt
        today = _dt.datetime.now(pytz.timezone("Asia/Kolkata")).date()
        today_df = df_all[df_all["timestamp"].dt.date == today].copy()

        # Fallback: if today has no candles yet (weekend/before open), use latest available day
        if today_df.empty:
            latest = df_all["timestamp"].dt.date.max()
            today_df = df_all[df_all["timestamp"].dt.date == latest].copy()

        if len(today_df) < 20:
            return {"matches": [], "status": "insufficient_data", "library_size": len(PATTERN_LIBRARY)}

        feats_today = extract_gbm_features(today_df.reset_index(drop=True))
        vec_today_df = feats_today[FEATURE_COLS_PATTERN].dropna()
        if len(vec_today_df) < 5:
            return {"matches": [], "status": "insufficient_features", "library_size": len(PATTERN_LIBRARY)}
        vec_today = vec_today_df.mean().values.astype(float)

    except Exception as e:
        return {"matches": [], "status": f"error: {e}", "library_size": len(PATTERN_LIBRARY)}

    # Cosine similarity against all library entries
    def cosine_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    scores = []
    for date_str, entry in PATTERN_LIBRARY.items():
        try:
            sim = cosine_sim(vec_today, entry["vector"])
            scores.append({
                "date": date_str,
                "similarity": round(sim * 100, 1),
                "eod_change": entry["eod_change"],
                "day_range": entry["day_range"]
            })
        except Exception:
            continue

    top3 = sorted(scores, key=lambda x: x["similarity"], reverse=True)[:3]
    return {
        "matches": top3,
        "status": "ok",
        "library_size": len(PATTERN_LIBRARY)
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
async def get_candles(email: str = Query(None), strategy: str = Query("243A"), limit: int = Query(3000), current_user=Depends(get_current_user)):
    if not current_user or current_user.email != (email or "").strip().lower():
        raise HTTPException(status_code=401, detail="Authentication required")
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
            # Process only recent candles (tail limit) for sub-second response times!
            df_recent = session.candles_df.tail(max(500, limit)).copy()
            df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'], format='mixed')
            import pytz
            ist_tz = pytz.timezone("Asia/Kolkata")
            df_recent['time_epoch'] = df_recent['timestamp'].apply(lambda x: int(ist_tz.localize(x).timestamp()) if x.tzinfo is None else int(x.astimezone(ist_tz).timestamp()))
            
            seen_epochs = set()
            for _, row in df_recent.iterrows():
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
                
            # Precompute strategy signals ONLY on the recent tail portion (sub-50ms!)
            live_markers = get_strategy_signals_for_chart(df_recent, strategy)
            
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
        
        # Limit the candles dataset dynamically based on client request (default: 3000)
        limit = max(100, min(100000, limit))
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
    from backend_engine.config import TRADING_MODE
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
    if not db_ok or ws_status.get("stale", True):
        status = "unhealthy"
        
    return {
        "status": status,
        "database": "connected" if db_ok else "disconnected",
        "market_feed": "connected" if ws_status.get("connected") else "disconnected",
        "feed_stale": ws_status.get("stale"),
        "kill_switch_active": ks_active,
        "trading_mode": TRADING_MODE,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/api/killswitch/toggle")
async def toggle_killswitch(active: bool = Form(...)):
    """Triggers or resets the emergency halt switch."""
    from backend_engine.kill_switch import set_kill_switch_state
    set_kill_switch_state(active)
    return {"status": "success", "kill_switch_active": active}

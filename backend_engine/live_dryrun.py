# PASSKEY: rushit2712
import asyncio
import os
import sys
import datetime
import requests
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add parent directory to path to allow correct absolute/relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend_engine.strategies import get_strategy
from backend_engine.candle_builder import CandleBuilder, INDEX_TOKEN
from backend_engine.trade_logger import TradeLogger
from backend_engine.paper_trade_engine import PaperTradeEngine
from backend_engine.websocket_handler import WSHandler
from backend_engine.angel_ws_handler import AngelOneWSHandler
from backend_engine.risk_manager import RiskManager
from backend_engine.config import FIXED_SL, PYRAMIDING_LIMIT

# Global registries
active_sessions = {}  # user_id -> UserSession
restart_checker_task = None

import json

# Cache for Nifty Future Token to avoid repeated scrip master downloads
cached_future_token = None
cached_future_date = None

def get_cached_future_token():
    global cached_future_token, cached_future_date
    today = datetime.date.today()
    if cached_future_token is not None and cached_future_date == today:
        return cached_future_token
        
    # Check disk cache first to bypass download if we already resolved it today!
    cache_path = "backend_engine/resolved_future_token.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                if cached_data.get("date") == str(today):
                    cached_future_token = cached_data.get("token")
                    cached_future_date = today
                    print(f"[ScripMaster] Loaded cached Nifty Future token from disk: {cached_future_token}")
                    return cached_future_token
        except Exception as e:
            print(f"[ScripMaster] Error reading disk cache: {e}")
            
    try:
        print("[ScripMaster] Fetching scrip master to resolve Nifty Future token...")
        url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            instruments = resp.json()
            nifty_futs = [
                i for i in instruments
                if i.get("name") == "NIFTY"
                and i.get("instrumenttype") == "FUTIDX"
                and i.get("exch_seg") == "NFO"
            ]
            if nifty_futs:
                nifty_futs.sort(key=lambda i: datetime.datetime.strptime(i["expiry"], "%d%b%Y"))
                cached_future_token = nifty_futs[0]["token"]
                cached_future_date = today
                
                # Write to disk cache
                try:
                    with open(cache_path, "w") as f:
                        json.dump({"date": str(today), "token": cached_future_token}, f)
                except Exception as e:
                    print(f"[ScripMaster] Error writing disk cache: {e}")
                    
                print(f"[ScripMaster] Resolved front-month Nifty Future token: {cached_future_token}")
                return cached_future_token
    except Exception as e:
        print(f"[ScripMaster] Error fetching future token: {e}")
    return None

class DemoMarketSimulator:
    def __init__(self, tick_queue, trade_logger=None):
        self.tick_queue = tick_queue
        self.trade_logger = trade_logger
        self.is_running = False
        self.conn_status = {INDEX_TOKEN: "live"}
        
    def log(self, msg):
        if self.trade_logger:
            self.trade_logger.log_activity(msg)
            
    async def connect_and_stream(self):
        self.is_running = True
        self.log("[DEMO] Starting Demo Market Simulator (data replay/random walk)...")
        
        base_price = 24600.0
        try:
            import pandas as pd
            if os.path.exists("backend_engine/old data.csv"):
                df = pd.read_csv("backend_engine/old data.csv")
                if not df.empty:
                    base_price = float(df.iloc[-1]['close'])
        except Exception:
            pass
            
        import random
        from backend_engine.angel_ws_handler import CONSTITUENT_TOKENS
        
        while self.is_running:
            try:
                # Random walk simulation
                change = random.normalvariate(0, 1.5)
                base_price += change
                base_price = round(base_price, 2)
                
                # Nifty Spot tick (Token 26000 or INDEX_TOKEN)
                tick = {
                    "ScripCode": INDEX_TOKEN,
                    "LTP": base_price,
                    "Volume64": 0,
                    "Time": datetime.datetime.now(),
                    "RecType": "A"
                }
                await self.tick_queue.put(tick)
                
                # Constituent stock volume simulation
                fake_vol = random.randint(1000, 5000)
                vol_tick = {
                    "ScripCode": random.choice(CONSTITUENT_TOKENS),
                    "LTP": base_price / 100,
                    "Volume64": fake_vol,
                    "Time": datetime.datetime.now(),
                    "RecType": "d"
                }
                await self.tick_queue.put(vol_tick)
                
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log(f"[DEMO] Error in simulator loop: {e}")
                await asyncio.sleep(5)
                
    def stop(self):
        self.is_running = False

class UserSession:
    def __init__(self, user_id, credentials, strategy_name="243A"):
        self.user_id = user_id
        self.credentials = credentials
        self.strategy_name = strategy_name
        self.email = credentials.get("email", "")
        self.is_demo = self.email in ("demo@gmail.com", "developer@gmail.com")
        self.user_dir = "data"
        os.makedirs(self.user_dir, exist_ok=True)
        
        self.candle_data_path = "backend_engine/old data.csv"
        self.active_positions_path = "data/active_positions.json"
        
        self.trade_logger = TradeLogger(user_dir=self.user_dir)
        self.risk_manager = RiskManager()
        self.paper_trade_engine = PaperTradeEngine(
            ws_handler=None, option_ltp_cache=None,
            trade_logger=self.trade_logger,
            state_path=self.active_positions_path,
            user_id=self.user_id
        )
        self.candles_df = None
        self.ws_handler = None
        self.candle_builder = None
        self.future_token = None
        self.smart_connect = None
        
        self.index_ltp = 0.0
        self.last_index_time = None
        
        self.sync_in_progress = False
        self.warmup_in_progress = False
        self.warmup_total = 0
        self.warmup_current = 0
        self.warmup_progress = "0/0"
        self.warmup_time_remaining = 0
        
        self.system_running = False
        self.tasks = []

    def load_mixed_csv(self, file_path: str) -> pd.DataFrame:
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            
        headers = [h.strip().lower() for h in lines[0].split(",")]
        has_volume = "volume" in headers
        
        for idx in range(1, len(lines)):
            line = lines[idx].strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 5:
                try:
                    row_dict = {
                        "timestamp": parts[0].strip(),
                        "open": float(parts[1].strip()),
                        "high": float(parts[2].strip()),
                        "low": float(parts[3].strip()),
                        "close": float(parts[4].strip()),
                        "volume": float(parts[5].strip()) if (has_volume and len(parts) >= 6) else 0.0
                    }
                    rows.append(row_dict)
                except ValueError:
                    continue
        return pd.DataFrame(rows)

    def bootstrap_candles(self):
        """Warms up the user's candle database."""
        if os.path.exists(self.candle_data_path):
            try:
                self.candles_df = self.load_mixed_csv(self.candle_data_path)
                if len(self.candles_df) >= 400:
                    self.candles_df['timestamp'] = pd.to_datetime(self.candles_df['timestamp'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
                    self.trade_logger.log_activity(f"Candle CSV loaded with {len(self.candles_df)} rows.")
                    return
            except Exception as e:
                self.trade_logger.log_activity(f"Error bootstrapping from {self.candle_data_path}: {e}")
                pass

        warmup_file = "backend_engine/old data.csv"
        if os.path.exists(warmup_file):
            self.trade_logger.log_activity(f"Bootstrapping candles from warmup file: {warmup_file} ...")
            df_hist = self.load_mixed_csv(warmup_file)
            df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
            self.candles_df = df_hist
            self.trade_logger.log_activity(f"Loaded bootstrap candle file with {len(df_hist)} records.")
        else:
            self.candles_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            self.trade_logger.log_activity("Warning: No historical file found. Features will be warm-up delayed.")

    def fill_historical_gap(self, smart_connect=None):
        """
        Queries missing 5-minute Nifty Spot candles from Angel One API,
        downloads historical constituent volumes for the top 50 stocks,
        sums their volumes, and appends the contiguous candles to the database.
        """
        if self.candles_df is None or self.candles_df.empty:
            return
            
        self.sync_in_progress = True
        try:
            try:
                # Parse the last timestamp in our database
                last_timestamp = pd.to_datetime(self.candles_df.iloc[-1]['timestamp'])
                if last_timestamp.tzinfo is not None:
                    last_timestamp = last_timestamp.tz_localize(None)
            except Exception:
                return
                
            now = datetime.datetime.now()
            
            # Check if the gap is larger than 10 minutes (to avoid unnecessary fetches)
            if (now - last_timestamp).total_seconds() < 600:
                self.trade_logger.log_activity("Database is up to date. No historical gap found.")
                return
                
            self.trade_logger.log_activity(f"Calculating missing candles since {last_timestamp} up to {now}...")
            
            from backend_engine.config import DEMO_MODE
            if DEMO_MODE:
                self.trade_logger.log_activity("[DEMO] Bypassing historical gap filler sync in Demo Mode.")
                return
    
            sc = smart_connect
            if sc is None:
                # Only attempt developer credential fallback if DEMO_MODE is False
                self.trade_logger.log_activity("No active user session found. Logging in with default developer credentials...")
                try:
                    import pyotp
                    from SmartApi import SmartConnect
                    dev_api_key = os.getenv("DEV_API_KEY", "")
                    dev_client_id = os.getenv("DEV_CLIENT_ID", "")
                    dev_password = os.getenv("DEV_PASSWORD", "")
                    dev_totp_secret = os.getenv("DEV_TOTP_SECRET", "")
                    
                    if not (dev_api_key and dev_client_id):
                        self.trade_logger.log_activity("Developer credentials not configured in environment variables. Cannot sync missing candles.")
                        return
                        
                    sc = SmartConnect(api_key=dev_api_key)
                    totp = pyotp.TOTP(dev_totp_secret).now()
                    data = sc.generateSession(dev_client_id, dev_password, totp)
                    if data.get('status') != True:
                        self.trade_logger.log_activity("Developer credentials authentication failed. Cannot sync missing candles.")
                        return
                except Exception as e:
                    self.trade_logger.log_activity(f"Error authenticating with developer credentials: {e}")
                    return
                    
            # Fetch the missing candles from last_timestamp to now
            gap_candles = []
            curr_start = last_timestamp
            
            from backend_engine.websocket_handler import CONSTITUENT_TOKENS
            import time
            
            while curr_start < now:
                curr_end = min(curr_start + datetime.timedelta(days=30), now)
                from_str = curr_start.strftime("%Y-%m-%d %H:%M")
                to_str = curr_end.strftime("%Y-%m-%d %H:%M")
                
                historicParam = {
                    "exchange": "NSE",
                    "symboltoken": "99926000",
                    "interval": "FIVE_MINUTE",
                    "fromdate": from_str,
                    "todate": to_str
                }
                
                spot_data = []
                try:
                    for attempt in range(3):
                        try:
                            res = sc.getCandleData(historicParam)
                            if res and res.get('status') == True and res.get('data'):
                                spot_data = res['data']
                                break
                            else:
                                time.sleep(2.0)
                        except Exception:
                            time.sleep(2.0)
                except Exception as e:
                    self.trade_logger.log_activity(f"Error fetching Nifty Spot history: {e}")
                    break
                    
                if not spot_data:
                    curr_start = curr_end + datetime.timedelta(days=1)
                    continue
                    
                # Fetch constituent volume data for the same period
                volume_by_time = {}
                self.trade_logger.log_activity(f"Downloading constituent volumes for {len(spot_data)} gap candles...")
                
                for idx, token in enumerate(CONSTITUENT_TOKENS):
                    stockParam = {
                        "exchange": "NSE",
                        "symboltoken": str(token),
                        "interval": "FIVE_MINUTE",
                        "fromdate": from_str,
                        "todate": to_str
                    }
                    try:
                        stock_candles = []
                        for attempt in range(3):
                            try:
                                res = sc.getCandleData(stockParam)
                                if res and res.get('status') == True and res.get('data'):
                                    stock_candles = res['data']
                                    break
                                else:
                                    time.sleep(2.0)
                            except Exception:
                                time.sleep(2.0)
                                    
                        for item in stock_candles:
                            dt_val = pd.to_datetime(item[0])
                            if dt_val.tzinfo is not None:
                                dt_val = dt_val.tz_localize(None)
                            ts_str = dt_val.strftime('%Y-%m-%d %H:%M:%S')
                            volume_by_time[ts_str] = volume_by_time.get(ts_str, 0) + int(item[5])
                    except Exception:
                        pass
                    time.sleep(0.35) # keep requests spaced out under the 3 TPS history API rate limit
                    
                # Merge Spot OHLC with summed stock volumes
                for item in spot_data:
                    dt_parsed = pd.to_datetime(item[0])
                    if dt_parsed.tzinfo is not None:
                        dt_parsed = dt_parsed.tz_localize(None)
                        
                    if dt_parsed > last_timestamp:
                        ts_str = dt_parsed.strftime('%Y-%m-%d %H:%M:%S')
                        summed_vol = float(volume_by_time.get(ts_str, 0))
                        gap_candles.append({
                            "timestamp": ts_str,
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": summed_vol
                        })
                        
                curr_start = curr_end + datetime.timedelta(days=1)
                time.sleep(0.5)
                
            if gap_candles:
                self.trade_logger.log_activity(f"Downloaded {len(gap_candles)} missing candles with constituent volume sums.")
                df_gap = pd.DataFrame(gap_candles)
                self.candles_df = pd.concat([self.candles_df, df_gap], ignore_index=True)
                self.candles_df = self.candles_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                self.candles_df.to_csv(self.candle_data_path, index=False)
                self.trade_logger.log_activity("Historical database updated with missing candles successfully.")
            else:
                self.trade_logger.log_activity("No new missing candles found to sync.")
                
            # Sync strategy signals for both models!
            sync_model_signals(self, "243A")
            sync_model_signals(self, "LONGPING")
        finally:
            self.sync_in_progress = False

    async def start(self):
        if self.system_running:
            return
        self.system_running = True
        
        # Use asyncio.to_thread to run heavy synchronous loading operations off the main event loop
        await asyncio.to_thread(self.bootstrap_candles)
        
        if self.is_demo:
            self.trade_logger.log_activity("[DEMO] Starting in Demo Mode. Displaying static cached candles & signals.")
            return
            
        self.future_token = await asyncio.to_thread(get_cached_future_token)
        
        # 2. Init websocket queue
        tick_queue = asyncio.Queue()
        
        from backend_engine.config import DEMO_MODE, TRADING_MODE
        
        if DEMO_MODE or TRADING_MODE == "DEMO":
            self.trade_logger.log_activity("[DEMO] Starting in Demo Mode. Replaying historical data...")
            self.ws_handler = DemoMarketSimulator(tick_queue, trade_logger=self.trade_logger)
            asyncio.create_task(asyncio.to_thread(self.fill_historical_gap, None))
        else:
            api_key = self.credentials.get("api_key", "")
            client_id = self.credentials.get("client_id", "")
            password = self.credentials.get("password", "")
            totp_secret = self.credentials.get("totp_secret", "")
            
            if api_key and client_id:
                self.trade_logger.log_activity("Starting in Angel One Mode. Authenticating...")
                import pyotp
                from SmartApi import SmartConnect
                
                try:
                    smart_connect = SmartConnect(api_key=api_key)
                    totp = pyotp.TOTP(totp_secret).now()
                    data = await asyncio.to_thread(smart_connect.generateSession, client_id, password, totp)
                    if data.get('status') == True:
                        self.smart_connect = smart_connect
                        jwt_token = data['data']['jwtToken']
                        feed_token = await asyncio.to_thread(smart_connect.getfeedToken)
                        
                        self.trade_logger.log_activity("Launching historical gap sync task in background thread...")
                        asyncio.create_task(asyncio.to_thread(self.fill_historical_gap, smart_connect))
                            
                        self.ws_handler = AngelOneWSHandler(
                            tick_queue, api_key, client_id, jwt_token, feed_token, 
                            trade_logger=self.trade_logger, future_token=None
                        )
                        self.trade_logger.log_activity("Angel One authentication successful. Live feed started.")
                    else:
                        err_msg = data.get('message', 'Session generation failed')
                        self.trade_logger.log_activity(f"Angel One authentication failed: {err_msg}. Syncing candles with default credentials and falling back to custom feed.")
                        asyncio.create_task(asyncio.to_thread(self.fill_historical_gap, None))
                        self.ws_handler = WSHandler(tick_queue, trade_logger=self.trade_logger)
                except Exception as e:
                    self.trade_logger.log_activity(f"Error during Angel One auth: {e}. Syncing candles with default credentials and falling back to custom feed.")
                    asyncio.create_task(asyncio.to_thread(self.fill_historical_gap, None))
                    self.ws_handler = WSHandler(tick_queue, trade_logger=self.trade_logger)
            else:
                self.trade_logger.log_activity("Starting in Custom Feed Mode. Syncing candles with default credentials.")
                asyncio.create_task(asyncio.to_thread(self.fill_historical_gap, None))
                self.ws_handler = WSHandler(tick_queue, trade_logger=self.trade_logger)
            
        self.paper_trade_engine.ws_handler = self.ws_handler
        self.candle_builder = CandleBuilder(
            on_candle_completed_cb=self.on_candle_completed,
            trade_logger=self.trade_logger,
            candle_data_path=self.candle_data_path,
            future_token=None
        )
        
        # Start worker tasks
        self.tasks.append(asyncio.create_task(self.ws_handler.connect_and_stream()))
        self.tasks.append(asyncio.create_task(self.tick_consumer(tick_queue)))
        self.tasks.append(asyncio.create_task(self.force_flush_timer()))
        self.tasks.append(asyncio.create_task(self.monitor_loop()))

    def on_candle_completed(self, timestamp, o, h, l, c, vol):
        asyncio.create_task(self.on_candle_completed_async(timestamp, o, h, l, c, vol))

    async def on_candle_completed_async(self, timestamp, o, h, l, c, vol):
        # 1. Append the new candle to our database
        new_row = pd.DataFrame([{
            "timestamp": timestamp,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol
        }])
        self.candles_df = pd.concat([self.candles_df, new_row], ignore_index=True)
            
        # 2. Evaluate Strategy
        strategy = get_strategy(self.strategy_name)
        in_pos = len(self.paper_trade_engine.active_positions) > 0
        try:
            result = strategy.predict(self.candles_df, in_position=in_pos)
            signal = result.get('signal', 0)
            metrics = result.get('metrics', {})
            # Save this live completed candle prediction to our persistent cache
            from backend_engine.signal_cacher import save_predictions_batch
            save_predictions_batch([{
                "timestamp": timestamp,
                "strategy": self.strategy_name,
                "signal": signal,
                "metrics": metrics
            }])
        except Exception as e:
            self.trade_logger.log_activity(f"Error during strategy predict: {e}")
            return
            
        # Format predictions dict for logging
        predictions = {
            'hmm_regime_name': metrics.get('hmm_regime', 'Unknown'),
            'gbm_prob_buy': metrics.get('gbm_prob_buy', 0.5 if signal == 1 else 0.1),
            'gbm_prob_sell': metrics.get('gbm_prob_sell', 0.5 if signal == -1 else 0.1),
            'tcn_predicted': 'BUY' if signal == 1 else ('SELL' if signal == -1 else 'HOLD'),
            'tcn_prob_buy': metrics.get('tcn_prob_buy', 0.5 if signal == 1 else 0.1),
            'tcn_prob_sell': metrics.get('tcn_prob_sell', 0.5 if signal == -1 else 0.1)
        }
        hmm_regime = metrics.get('hmm_regime', 'Unknown')
        
        self.trade_logger.log_activity(
            f"[{self.strategy_name} Candle Completed] Close: {c:.2f} | Signal: {signal} | Metrics: {metrics}"
        )
        
        # 3. Position Management
        dt_obj = pd.to_datetime(timestamp, format='mixed')
        entry_time = dt_obj + datetime.timedelta(minutes=5)
        entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M:%S')
        
        active_positions_snapshot = list(self.paper_trade_engine.active_positions)
        reverse_pos_type = None
        exited_this_bar = False
        
        # 3a. Check candle exits (EOD force exit, or strategy reversions)
        eod_minute = 10 # Default for 243A
        for pos in active_positions_snapshot:
            is_eod = False
            if self.strategy_name != "LONGPING":
                if (entry_time.hour == 15 and entry_time.minute >= eod_minute) or entry_time.hour > 15:
                    is_eod = True
                
            exit_reason = None
            if is_eod:
                exit_reason = "EOD"
            elif self.strategy_name == "LONGPING":
                # Check linear regression slope at completed candle close
                if len(self.candles_df) >= 20:
                    import numpy as np
                    y = self.candles_df['close'].tail(20).values
                    x = np.arange(20)
                    x_mean = x.mean()
                    x_dev = x - x_mean
                    x_var = (x_dev**2).sum()
                    slope_val = np.dot(x_dev, y) / x_var if x_var > 0 else 0.0
                else:
                    slope_val = 0.0
                    
                if pos["position_type"] == "LONG" and slope_val < -0.001:
                    exit_reason = "Longpine Trend Exit"
            else: # 243A
                if signal == -1 and pos["position_type"] == "LONG":
                    exit_reason = "REV"
                elif signal == 1 and pos["position_type"] == "SHORT":
                    exit_reason = "REV"
                    
            if exit_reason:
                self.trade_logger.log_activity(f"[CANDLE EXIT] Exit triggered: {exit_reason} at Nifty {c:.2f}")
                from backend_engine.execution_engine import execution_engine
                success, status = execution_engine.execute_order(
                    user_session=self,
                    strategy=self.strategy_name,
                    symbol="NIFTY",
                    side="CLOSE",
                    quantity=pos.get("option_lots", 1) * pos.get("lot_size", 65),
                    price=c,
                    current_time=entry_time_str,
                    entry_reason=exit_reason,
                    hmm_regime=None
                )
                rev_type = None
                if success:
                    exited_this_bar = True
                    should_reverse = False
                    if pos["position_type"] == "LONG" and signal == -1:
                        should_reverse = True
                    elif pos["position_type"] == "SHORT" and signal == 1:
                        should_reverse = True
                    if should_reverse and not (entry_time.hour == 15 and entry_time.minute >= 10):
                        rev_type = "SHORT" if pos["position_type"] == "LONG" else "LONG"
                
                if rev_type and self.strategy_name != "LONGPING":
                    reverse_pos_type = rev_type

        # 3b. Check Entry
        pos_type = None
        if len(self.paper_trade_engine.active_positions) < PYRAMIDING_LIMIT and not exited_this_bar:
            entry_reason = f"{self.strategy_name} entry signal"
            
            if reverse_pos_type:
                pos_type = reverse_pos_type
                entry_reason = "Reverse Entry"
            elif signal != 0:
                allow_entry = True
                if entry_time.hour == 9 and entry_time.minute == 15:
                    allow_entry = False
                if self.strategy_name != "LONGPING":
                    if (entry_time.hour == 15 and entry_time.minute >= eod_minute) or entry_time.hour > 15:
                        allow_entry = False
                    
                if allow_entry:
                    if signal == 1:
                        pos_type = "LONG"
                    elif signal == -1 and self.strategy_name != "LONGPING":
                        pos_type = "SHORT"
                        
            if pos_type:
                from backend_engine.execution_engine import execution_engine
                execution_engine.execute_order(
                    user_session=self,
                    strategy=self.strategy_name,
                    symbol="NIFTY",
                    side=pos_type,
                    quantity=65,
                    price=c,
                    current_time=entry_time_str,
                    entry_reason=entry_reason,
                    hmm_regime=hmm_regime
                )
                
        # Write the executed signals to the new row
        last_idx = len(self.candles_df) - 1
        if self.strategy_name == "LONGPING":
            if exited_this_bar:
                self.candles_df.at[last_idx, 'signal_longping'] = "SELL"
            elif pos_type == "LONG":
                self.candles_df.at[last_idx, 'signal_longping'] = "BUY"
        elif self.strategy_name == "243A":
            if exited_this_bar:
                self.candles_df.at[last_idx, 'signal_243a'] = "EXIT"
            elif pos_type == "LONG":
                self.candles_df.at[last_idx, 'signal_243a'] = "BUY"
            elif pos_type == "SHORT":
                self.candles_df.at[last_idx, 'signal_243a'] = "SELL"
                
        # Write completed candles to CSV file
        self.candles_df.to_csv(self.candle_data_path, index=False)
                
        # Log candle metrics
        ohlcv_dict = {"open": o, "high": h, "low": l, "close": c, "volume": vol}
        self.trade_logger.log_candle_metrics(timestamp, ohlcv_dict, predictions, signal, "BUY" if signal == 1 else ("SELL" if signal == -1 else "nan"))

    async def tick_consumer(self, queue):
        self.trade_logger.log_activity("Tick consumer started.")
        from backend_engine.health_monitor import monitor as health_monitor
        while True:
            try:
                tick = await queue.get()
                health_monitor.record_tick()
                scrip = int(tick["ScripCode"])
                rec_type = tick["RecType"]
                
                # 1. Update Spot Price LTP
                if scrip == INDEX_TOKEN and rec_type in ("A", "H") and tick["LTP"] is not None:
                    self.index_ltp = tick["LTP"]
                    self.last_index_time = tick["Time"]
                    
                    # Check SL/TP in real-time
                    active_positions_snapshot = list(self.paper_trade_engine.active_positions)
                    for pos in active_positions_snapshot:
                        pos_type = pos["position_type"]
                        entry_nifty = pos["entry_nifty_price"]
                        sl_nifty = pos["sl_nifty_price"]
                        tp_nifty = pos["tp_nifty_price"]
                        
                        exit_reason = None
                        if pos_type == "LONG":
                            if self.index_ltp <= sl_nifty:
                                exit_reason = "SL"
                            elif self.index_ltp >= tp_nifty:
                                exit_reason = "TP"
                        else:
                            if self.index_ltp >= sl_nifty:
                                exit_reason = "SL"
                            elif self.index_ltp <= tp_nifty:
                                exit_reason = "TP"
                                
                        if exit_reason:
                            self.trade_logger.log_activity(
                                f"[REAL-TIME TICK EXIT] Spot Index {self.index_ltp:.2f} touched {exit_reason} limit. Exiting..."
                            )
                            tick_time_str = tick["Time"].strftime('%Y-%m-%d %H:%M:%S')
                            from backend_engine.execution_engine import execution_engine
                            execution_engine.execute_order(
                                user_session=self,
                                strategy=self.strategy_name,
                                symbol="NIFTY",
                                side="CLOSE",
                                quantity=pos.get("option_lots", 1) * pos.get("lot_size", 65),
                                price=self.index_ltp,
                                current_time=tick_time_str,
                                entry_reason=exit_reason,
                                hmm_regime=None
                            )
                            
                # 2. Pass ticks to candle builder
                self.candle_builder.process_tick(tick)
                queue.task_done()
            except Exception as e:
                self.trade_logger.log_activity(f"Error in tick consumer: {e}")

    async def force_flush_timer(self):
        while True:
            await asyncio.sleep(10)
            try:
                if self.candle_builder:
                    self.candle_builder.force_flush_stale(30)
            except Exception as e:
                self.trade_logger.log_activity(f"Timer Error: {e}")

    async def monitor_loop(self):
        while True:
            await asyncio.sleep(60)
            try:
                current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                realized = sum(t.get("pnl", 0.0) for t in self.paper_trade_engine.trades)
                unrealized = 0.0
                for pos in self.paper_trade_engine.active_positions:
                    pos_type = pos["position_type"]
                    entry_nifty = pos["entry_nifty_price"]
                    qty = pos.get("lot_size", 75) * pos.get("option_lots", 1)
                    pnl_pts = (self.index_ltp - entry_nifty) if pos_type == "LONG" else (entry_nifty - self.index_ltp)
                    unrealized += pnl_pts * qty
                
                self.trade_logger.log_activity(f"LTP: {self.index_ltp:.2f} | Active Positions: {len(self.paper_trade_engine.active_positions)} | Today PnL: {realized+unrealized:+.2f}")
            except Exception:
                pass

    def stop(self):
        self.system_running = False
        if self.ws_handler:
            try:
                self.ws_handler.stop()
            except Exception:
                pass
        for t in self.tasks:
            try:
                t.cancel()
            except Exception:
                pass
        self.tasks = []

def sync_model_signals(session, strategy_name):
    candles_df = session.candles_df
    trade_logger = session.trade_logger
    
    if candles_df is None or candles_df.empty:
        return
        
    last_candle_ts = pd.to_datetime(candles_df['timestamp'].iloc[-1])
    
    from backend_engine.signal_cacher import load_cached_predictions, save_predictions_batch
    cache_df = load_cached_predictions()
    
    strategy_cache = cache_df[cache_df['strategy'] == strategy_name]
    
    if strategy_cache.empty:
        # If no cache exists, default to last candle minus 2 days to calculate gap
        last_signal_ts = last_candle_ts - datetime.timedelta(days=2)
    else:
        last_signal_ts = pd.to_datetime(strategy_cache['timestamp'].iloc[-1])
        
    missing_indices = candles_df[pd.to_datetime(candles_df['timestamp']) > last_signal_ts].index
    msg = f"[Model Sync] Strategy: {strategy_name} | Last Signal: {last_signal_ts} | Last Candle: {last_candle_ts} | Missing candles: {len(missing_indices)}"
    if trade_logger:
        trade_logger.log_activity(msg)
    else:
        print(msg)
        
    if len(missing_indices) > 0:
        msg = f"[Model Sync] Syncing missing signals for strategy: {strategy_name}..."
        if trade_logger:
            trade_logger.log_activity(msg)
        else:
            print(msg)
            
        session.warmup_in_progress = True
        session.warmup_total = len(missing_indices)
        session.warmup_current = 0
        session.warmup_progress = f"0/{session.warmup_total}"
        session.warmup_time_remaining = int(session.warmup_total * 0.15)
        
        from backend_engine.strategies import get_strategy
        strategy = get_strategy(strategy_name)
        
        predictions_to_save = []
        for idx in missing_indices:
            lookback = candles_df.iloc[:idx+1].copy()
            try:
                result = strategy.predict(lookback, in_position=False)
                signal = result.get('signal', 0)
                metrics = result.get('metrics', {})
                predictions_to_save.append({
                    "timestamp": candles_df['timestamp'].iloc[idx],
                    "strategy": strategy_name,
                    "signal": signal,
                    "metrics": metrics
                })
            except Exception as e:
                err_msg = f"[Model Sync] Error syncing signal at idx {idx}: {e}"
                if trade_logger:
                    trade_logger.log_activity(err_msg)
                else:
                    print(err_msg)
            
            session.warmup_current += 1
            session.warmup_progress = f"{session.warmup_current}/{session.warmup_total}"
            session.warmup_time_remaining = int((session.warmup_total - session.warmup_current) * 0.15)
                    
        if predictions_to_save:
            save_predictions_batch(predictions_to_save)
            msg = f"[Model Sync] Completed! Saved {len(predictions_to_save)} missing signals to cache."
            if trade_logger:
                trade_logger.log_activity(msg)
            else:
                print(msg)
                
        session.warmup_in_progress = False

def start_user_system(user_id, credentials, strategy_name="243A"):
    if active_sessions:
        # Re-use existing running session to avoid duplicate streams & CPU usage
        active_uid = list(active_sessions.keys())[0]
        session = active_sessions[active_uid]
        print(f"[Session Manager] Active session already running for user {active_uid}. Reusing for user {user_id}.")
        active_sessions[user_id] = session
        return session
        
    session = UserSession(user_id, credentials, strategy_name)
    active_sessions[user_id] = session
    asyncio.create_task(session.start())
    return session

def stop_user_system(user_id):
    if user_id in active_sessions:
        session = active_sessions[user_id]
        del active_sessions[user_id]
        # Only stop the session if no other active users are using it
        if not any(s == session for s in active_sessions.values()):
            session.stop()
            print(f"[Session Manager] All users logged out. Stopped shared session.")

async def daily_restart_checker_loop():
    """Daily check to restart all user systems cleanly at 9:00 AM IST."""
    last_reset_date = None
    while True:
        try:
            now = datetime.datetime.now()
            if now.hour == 9 and now.minute == 0 and now.date() != last_reset_date:
                last_reset_date = now.date()
                print("[Daily Restart] Running daily reset checker for all users...")
                from users_db import get_all_users
                users = get_all_users()
                for user in users:
                    user_id = user["id"]
                    if user_id in active_sessions:
                        print(f"[Daily Restart] Restarting session for user {user['email']}...")
                        credentials = {
                            "api_key": user["api_key"],
                            "client_id": user["client_id"],
                            "password": user["password"],
                            "totp_secret": user["totp_secret"]
                        }
                        strategy_name = active_sessions[user_id].strategy_name
                        stop_user_system(user_id)
                        await asyncio.sleep(2)
                        start_user_system(user_id, credentials, strategy_name)
        except Exception as e:
            print(f"Error in daily restart checker: {e}")
        await asyncio.sleep(30)

def start_background_system():
    # Deprecated single-user launcher wrapper
    global restart_checker_task
    if restart_checker_task is None or restart_checker_task.done():
        restart_checker_task = asyncio.create_task(daily_restart_checker_loop())

def stop_background_system():
    # Cancel all sessions
    for user_id in list(active_sessions.keys()):
        stop_user_system(user_id)


def check_and_sync_missing(candle_data_path):
    import pandas as pd
    import datetime as dt
    import os
    if not os.path.exists(candle_data_path):
        return True
        
    try:
        df = pd.read_csv(candle_data_path)
        if df.empty:
            return True
        last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_localize(None)
            
        now = dt.datetime.now()
        
        # Check if the gap is larger than 10 minutes (to see if we have missing candles)
        # We only check regular trading days (Mon-Fri) and during or after market hours
        if now.weekday() < 5:
            market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            if now > market_start:
                gap_seconds = (now - last_ts).total_seconds()
                if gap_seconds > 600:
                    return True
        else:
            # On weekends, if the last candle in DB is older than Friday 15:30
            pass
            
    except Exception:
        return True
        
    return False

def sync_last_72_candles(sc, candle_data_path, email=None):
    import datetime as dt
    import pandas as pd
    import time
    from backend_engine.websocket_handler import CONSTITUENT_TOKENS
    
    # 72 candles at 5m interval is 6 hours of trading time.
    # To cover weekends and non-trading hours, we go back 3 days.
    to_dt = dt.datetime.now()
    from_dt = to_dt - dt.timedelta(days=3)
    
    from_str = from_dt.strftime("%Y-%m-%d %H:%M")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M")
    
    from backend_engine.config import DEMO_MODE
    if DEMO_MODE:
        print("[DEMO] Manual 72-candle sync bypassed in Demo Mode.")
        return True

    # Authenticate with default developer key if no active user session
    if sc is None:
        try:
            import pyotp
            from SmartApi import SmartConnect
            dev_api_key = os.getenv("DEV_API_KEY", "")
            dev_client_id = os.getenv("DEV_CLIENT_ID", "")
            dev_password = os.getenv("DEV_PASSWORD", "")
            dev_totp_secret = os.getenv("DEV_TOTP_SECRET", "")
            
            if not (dev_api_key and dev_client_id):
                return False
                
            sc = SmartConnect(api_key=dev_api_key)
            totp = pyotp.TOTP(dev_totp_secret).now()
            data = sc.generateSession(dev_client_id, dev_password, totp)
            if data.get('status') != True:
                return False
        except Exception:
            return False
            
    # 1. Fetch Nifty Spot candles
    params = {
        "exchange": "NSE",
        "symboltoken": "99926000",
        "interval": "FIVE_MINUTE",
        "fromdate": from_str,
        "todate": to_str
    }
    
    spot_data = []
    for attempt in range(3):
        try:
            res = sc.getCandleData(params)
            if res and res.get('status') == True and res.get('data'):
                spot_data = res['data']
                break
            else:
                time.sleep(2.0)
        except Exception:
            time.sleep(2.0)
            
    if not spot_data:
        return False
        
    # Get last 72 candles
    spot_data = spot_data[-72:]
    
    # 2. Fetch constituent stock volumes for the same time period
    volume_by_time = {}
    for idx, token in enumerate(CONSTITUENT_TOKENS):
        stock_params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": "FIVE_MINUTE",
            "fromdate": from_str,
            "todate": to_str
        }
        try:
            stock_candles = []
            for attempt in range(3):
                try:
                    res = sc.getCandleData(stock_params)
                    if res and res.get('status') == True and res.get('data'):
                        stock_candles = res['data']
                        break
                    else:
                        time.sleep(2.0)
                except Exception:
                    time.sleep(2.0)
                    
            for item in stock_candles:
                dt_val = pd.to_datetime(item[0])
                if dt_val.tzinfo is not None:
                    dt_val = dt_val.tz_localize(None)
                ts_str = dt_val.strftime('%Y-%m-%d %H:%M:%S')
                volume_by_time[ts_str] = volume_by_time.get(ts_str, 0) + int(item[5])
        except Exception:
            pass
        time.sleep(0.35) # strict sleep to prevent hitting rate limit
        
    # 3. Read existing dataset
    df_old = pd.read_csv(candle_data_path)
    df_old['timestamp'] = pd.to_datetime(df_old['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 4. Merge new/updated candles (ensure no duplicates, replace if already exists)
    new_rows = []
    for item in spot_data:
        dt_parsed = pd.to_datetime(item[0])
        if dt_parsed.tzinfo is not None:
            dt_parsed = dt_parsed.tz_localize(None)
        ts_str = dt_parsed.strftime('%Y-%m-%d %H:%M:%S')
        
        summed_vol = float(volume_by_time.get(ts_str, 0))
        
        df_old = df_old[df_old['timestamp'] != ts_str]
        
        new_rows.append({
            "timestamp": ts_str,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": summed_vol
        })
        
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    df_combined.to_csv(candle_data_path, index=False)
    
    # 5. Precalculate and cache predictions for these 72 candles (saving runtime!)
    df_updated = pd.read_csv(candle_data_path)
    df_updated['timestamp'] = pd.to_datetime(df_updated['timestamp'])
    
    from backend_engine.strategies import get_strategy
    strategy = get_strategy("243A")
    
    start_idx = max(150, len(df_updated) - 72)
    new_predictions = []
    
    for idx in range(start_idx, len(df_updated)):
        lookback = df_updated.iloc[max(0, idx - 149) : idx + 1].reset_index(drop=True)
        timestamp = lookback.iloc[-1]['timestamp']
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
        try:
            pred = strategy.predict(lookback)
            new_predictions.append({
                "timestamp": timestamp,
                "strategy": "243A",
                "signal": pred.get("signal", 0),
                "metrics": pred.get("metrics", {})
            })
        except Exception:
            pass
            
    if new_predictions:
        from backend_engine.signal_cacher import save_predictions_batch
        save_predictions_batch(new_predictions)
        
    return True

# PASSKEY: rushit2712
import os
import sys
import time
import json
import threading
import datetime
import pytz
from collections import deque, defaultdict
import pandas as pd
import numpy as np
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from dotenv import load_dotenv

load_dotenv()

# Import strategy config
from backend_engine.config import (
    GBM_THRESHOLD, TCN_THRESHOLD, SL_POINTS, TSL_POINTS,
    TSL_ONLY_IN_PROFIT, FORCE_EXIT_HOUR, FORCE_EXIT_MINUTE
)

# Import model inference engine
import importlib
live_pred_mod = importlib.import_module("243A.live_prediction_engine")
LivePredictionEngine = live_pred_mod.LivePredictionEngine

from backend_engine.historical_signal_engine import (
    download_and_sync_candles,
    generate_model_signals_for_candles,
    apply_constituent_volume_sum
)
from backend_engine.simulate_risk_rules import is_smart_time, SmartThermalDissipationIntegerSizer
from backend_engine.pattern_match_engine import PatternMatchEngine

CONSTITUENT_NAMES = {
    "10999": "MARUTI", "1232": "GRASIM", "13538": "TECHM", "1333": "HDFCBANK",
    "1394": "HINDUNILVR", "10604": "BHARTIARTL", "11536": "TCS", "1363": "HINDALCO",
    "1660": "ITC", "11630": "NTPC", "14977": "POWERGRID", "15083": "ADANIPORTS",
    "157": "APOLLOHOSP", "1594": "INFY", "16675": "BAJAJFINSV", "11483": "LT",
    "3506": "TITAN", "4963": "ICICIBANK", "1922": "KOTAKBANK", "11195": "INDIGO",
    "11723": "JSWSTEEL", "2031": "M&M", "21808": "SBILIFE", "25": "ADANIENT",
    "2475": "ONGC", "4306": "SHRIRAMFIN", "881": "DRREDDY", "3045": "SBIN",
    "3432": "TATACONSUM", "3456": "TMPV", "383": "BEL", "5097": "ETERNAL",
    "17963": "NESTLEIND", "11532": "ULTRACEMCO", "16669": "BAJAJ-AUTO", "1964": "TRENT",
    "236": "ASIANPAINT", "694": "CIPLA", "20374": "COALINDIA", "22377": "MAXHEALTH",
    "317": "BAJFINANCE", "3351": "SUNPHARMA", "3499": "TATASTEEL", "3787": "WIPRO",
    "5900": "AXISBANK", "467": "HDFCLIFE", "7229": "HCLTECH", "910": "EICHERMOT",
    "18143": "JIOFIN", "2885": "RELIANCE"
}
CONSTITUENT_TOKENS = [int(k) for k in CONSTITUENT_NAMES.keys()]

IST = pytz.timezone("Asia/Kolkata")

def to_chart_epoch(dt):
    """
    Converts a timestamp string or datetime object in Indian Standard Time (IST)
    into a Unix epoch timestamp that Lightweight Charts will render directly as IST clock time
    (strictly between 09:15 and 15:30).
    """
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    if hasattr(dt, 'tz_localize') and dt.tzinfo is not None:
        dt = dt.tz_convert('Asia/Kolkata').tz_localize(None)
    elif hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        dt = dt.astimezone(IST).replace(tzinfo=None)
    # Treat the IST wall-clock representation as UTC epoch so chart displays exact 09:15 - 15:30
    return int(dt.replace(tzinfo=pytz.UTC).timestamp())

class LiveFeedService:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.api_key = os.getenv("ANGEL_API_KEY", "")
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "")
        self.password = os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

        self.smart_connect = None
        self.ws = None
        self.is_running = False
        self.connection_status = "offline"  # offline, connecting, live, error
        self.last_tick_time = None
        self.tick_count = 0

        # High-frequency buffer for Tick Data Board (latest 1000 ticks)
        self.ticks_buffer = deque(maxlen=1000)

        # In-memory Candle store
        self.candles_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self.current_bucket = None
        self.current_candle = None
        self.pending_entry = None  # Holds signal from candle T to execute at open of candle T+1

        # Trading & Signal state
        self.signals = []         # List of signal dicts
        self.active_positions = []  # List of up to 4 concurrent active positions
        self.closed_trades = []    # Completed trades history
        self.trade_counter = 0
        self.thermal_sizer = SmartThermalDissipationIntegerSizer()

        # Markers for Lightweight Charts
        self.chart_markers = []
        self._historical_markers = []

        # Latest ML predictions and features
        self.latest_prediction = {
            "gbm_predicted": 0,
            "gbm_prob_buy": 0.5,
            "gbm_prob_sell": 0.5,
            "tcn_predicted": "HOLD",
            "tcn_prob_buy": 0.33,
            "tcn_prob_sell": 0.33,
            "hmm_regime": "compression",
            "consensus": "HOLD",
            "signal": 0,
            "timestamp": "--"
        }
        self.feature_contributions = {}

        # Last known LTP for direction calculation
        self.last_ltp = None
        self.index_ltp = 0.0
        self.day_open = 0.0
        self.day_high = 0.0
        self.day_low = 0.0
        self.prev_close = 0.0
        self.close_5d = 0.0

        # 15-Minute Pattern Match Direction Engine (18 Years 2008-2025 Match)
        self.pattern_engine = PatternMatchEngine.get_instance()

        # Nifty 50 50-Constituent volume sum tracking & Market Breadth stats
        self.token_day_volume = {}       # token_id -> latest cumulative day volume
        self.constituent_volume_sum = 0  # Total summed volume across all 50 stocks today
        self.constituent_stats = {
            str(k): {
                "name": v,
                "ltp": 0.0,
                "base_p": 0.0,
                "change": 0.0,
                "change_pct": 0.0
            }
            for k, v in CONSTITUENT_NAMES.items()
        }

        # Model Engine
        self.prediction_engine = None
        self._init_engine()

        # Warmup with historical data
        self._warmup_history()

        # Preload historical signals and entries (last 3 months, EOD exits only)
        self._load_historical_markers()

    def _init_engine(self):
        try:
            print("[LiveFeedService] Initializing 3 ML Models (LightGBM, TCN, HMM)...")
            self.prediction_engine = LivePredictionEngine()
            print("[LiveFeedService] ML Models initialized successfully!")
        except Exception as e:
            print(f"[LiveFeedService] Error initializing ML engine: {e}")

    def _warmup_history(self):
        """Loads and syncs 5-min candles strictly within the last 3 months and Indian market hours (09:15 to 15:30)."""
        try:
            print("[LiveFeedService] Syncing past 3 months (90 days) candles from Angel One / local cache...")
            df_synced = download_and_sync_candles(
                api_key=self.api_key,
                client_id=self.client_id,
                password=self.password,
                totp_secret=self.totp_secret,
                days=90
            )
            if not df_synced.empty:
                self.candles_df = df_synced.reset_index(drop=True)
                self.candles_df["timestamp"] = pd.to_datetime(self.candles_df["timestamp"])
                self.index_ltp = float(self.candles_df.iloc[-1]["close"])
                self.last_ltp = self.index_ltp
                today_date = self.candles_df.iloc[-1]["timestamp"].date()
                today_candles = self.candles_df[self.candles_df["timestamp"].dt.date == today_date]
                if not today_candles.empty:
                    self.day_open = float(today_candles.iloc[0]["open"])
                else:
                    self.day_open = self.index_ltp

                # Calculate prev_close and close_5d for Pattern Match Engine
                unique_dates = sorted(self.candles_df["timestamp"].dt.date.unique())
                if len(unique_dates) >= 2:
                    prev_d = unique_dates[-2]
                    prev_c = self.candles_df[self.candles_df["timestamp"].dt.date == prev_d]
                    if not prev_c.empty:
                        self.prev_close = float(prev_c.iloc[-1]["close"])
                if len(unique_dates) >= 6:
                    d_5 = unique_dates[-6]
                    c_5 = self.candles_df[self.candles_df["timestamp"].dt.date == d_5]
                    if not c_5.empty:
                        self.close_5d = float(c_5.iloc[-1]["close"])

                # Feed today's 5-min candles into pattern engine (temp 15m in-memory aggregation)
                if not today_candles.empty and self.pattern_engine:
                    macro_info = {
                        "prev_close": self.prev_close,
                        "close_5d": self.close_5d
                    }
                    for _, row in today_candles.iterrows():
                        c_dict = {
                            "timestamp": row["timestamp"],
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row.get("volume", 0.0))
                        }
                        self.pattern_engine.on_5min_candle_closed(c_dict, macro_info)
                    
                    # Evaluate tick invalidation against latest LTP
                    self.pattern_engine.on_tick(self.index_ltp, datetime.datetime.now())

                print(f"[LiveFeedService] Warmup loaded {len(self.candles_df)} candles for past 3 months. Latest close: {self.index_ltp}, Day Open: {self.day_open}")
                self._evaluate_strategy_on_candles(self.candles_df)
                if not self.candles_df.empty:
                    last_row = self.candles_df.iloc[-1]
                    last_ts = pd.to_datetime(last_row["timestamp"])
                    self.current_bucket = self.get_5min_bucket(last_ts)
                    self.current_candle = {
                        "time": to_chart_epoch(self.current_bucket),
                        "bucket": self.current_bucket,
                        "timestamp": self.current_bucket.strftime("%Y-%m-%d %H:%M:%S"),
                        "open": float(last_row["open"]),
                        "high": float(last_row["high"]),
                        "low": float(last_row["low"]),
                        "close": float(last_row["close"]),
                        "volume": float(last_row.get("volume", 0.0)),
                        "ticks": 1
                    }
        except Exception as e:
            print(f"[LiveFeedService] Error in _warmup_history: {e}")

    def _load_historical_markers(self):
        """
        Loads historical signals, entry points (Signal+1), and all EXITS with Total Net PnL.
        If signals/markers are missing, runs the 3 ML models to generate them dynamically!
        """
        sig_csv = "backend_engine/model signal.csv"
        markers_json = "backend_engine/model_markers.json"
        need_generate = True

        if os.path.exists(markers_json) and os.path.exists(sig_csv):
            try:
                with open(markers_json, "r", encoding="utf-8") as f:
                    cached_markers = json.load(f)
                df_sig = pd.read_csv(sig_csv)
                if cached_markers and not df_sig.empty:
                    df_sig['Entry Time'] = pd.to_datetime(df_sig['Entry Time'])
                    latest_sig_time = df_sig['Entry Time'].max()
                    if (datetime.datetime.now() - latest_sig_time).days <= 3:
                        need_generate = False
                        # Filter strictly to past 3 months (90 days)
                        now = datetime.datetime.now()
                        cutoff = now - datetime.timedelta(days=90)
                        cutoff_epoch = to_chart_epoch(cutoff)
                        self._historical_markers = [m for m in cached_markers if m.get("time", 0) >= cutoff_epoch]
                        
                        df_sig = df_sig[df_sig['Entry Time'] >= cutoff].copy()
                        all_trades = []
                        cum_net_pnl = 0.0
                        for _, row in df_sig.iterrows():
                            entry_dt = row["Entry Time"]
                            is_smart = is_smart_time(entry_dt.hour, entry_dt.day_name())
                            lot_size = float(row.get("Lot Size", 1.0))
                            is_entered = is_smart and (lot_size > 0)
                            
                            pnl_1qty = float(row.get("1 QTY PnL", 0.0))
                            sim_pnl = float(row.get("Sim PnL", pnl_1qty * lot_size)) if is_entered else 0.0
                            temp_val = float(row.get("Temperature", 1.0))
                            if is_entered:
                                cum_net_pnl = round(cum_net_pnl + sim_pnl, 2)
                                
                            all_trades.append({
                                "id": len(all_trades) + 1,
                                "direction": str(row.get("Direction", "BUY")).upper(),
                                "entry_time": str(row.get("Entry Time")),
                                "exit_time": str(row.get("Exit Time")),
                                "entry_price": float(row.get("Nifty Enter Price", 0.0)),
                                "exit_price": float(row.get("Nifty Exit Price", 0.0)),
                                "1_qty_pnl": pnl_1qty if is_entered else 0.0,
                                "lot_size": lot_size if is_entered else 0.0,
                                "pnl_points": sim_pnl,
                                "cumulative_net_pnl": cum_net_pnl,
                                "exit_reason": str(row.get("Exit Reason", "EOD")).upper() if is_entered else f"{str(row.get('Exit Reason', 'EOD')).upper()} (TIME FILTERED)",
                                "temperature": temp_val,
                                "entered": is_entered
                            })
                        self.closed_trades = all_trades
                        if all_trades:
                            self.thermal_sizer.T = all_trades[-1]["temperature"]
                        print(f"[LiveFeedService] Loaded {len(self._historical_markers)} markers and {len(self.closed_trades)} trades ({sum(1 for t in all_trades if t['entered'])} entered, Net PnL: {cum_net_pnl:+.2f} pts, Thermal T={self.thermal_sizer.T:.2f}).")
                        return
            except Exception as e:
                print(f"[LiveFeedService] Error reading markers cache: {e}")
                need_generate = True

        if need_generate and not self.candles_df.empty:
            print("[LiveFeedService] Signals missing or outdated. Passing candles through 3 ML models (LightGBM, TCN, HMM)...")
            trades, markers = generate_model_signals_for_candles(self.candles_df)
            self._historical_markers = markers
            self.closed_trades = trades
            print(f"[LiveFeedService] Dynamic signal generation completed: {len(trades)} trades, {len(markers)} markers.")
            return

        try:
            df_sig = pd.read_csv(sig_csv)
            df_sig['Entry Time'] = pd.to_datetime(df_sig['Entry Time'])
            df_sig['Exit Time'] = pd.to_datetime(df_sig['Exit Time'])

            # Filter strictly to past 3 months (90 days)
            now = datetime.datetime.now()
            cutoff = now - datetime.timedelta(days=90)
            df_sig = df_sig[df_sig['Entry Time'] >= cutoff].copy()

            # Enforce max 4 concurrent signals / active trades at any single time
            # If 4 trades are active, no new trade can enter until at least one exits (if 1st is exit than new enter)
            MAX_CONCURRENT_SIGNALS = 4
            selected_trades = []
            active_slots = []

            for _, row in df_sig.sort_values("Entry Time").iterrows():
                entry_dt = row["Entry Time"]
                # Free up slots where trade has already exited before or at this entry_dt
                active_slots = [ex_t for ex_t in active_slots if ex_t > entry_dt]

                if len(active_slots) < MAX_CONCURRENT_SIGNALS:
                    selected_trades.append(row)
                    active_slots.append(row["Exit Time"])

            df_trades = pd.DataFrame(selected_trades)
            markers = []
            seen_candles = set()
            cum_net_pnl = 0.0

            exit_colors = {
                "EOD": "#f59e0b",
                "TP": "#10b981",
                "SL": "#ef4444",
                "TSL": "#a855f7"
            }
            exit_shapes = {
                "EOD": "square",
                "TP": "circle",
                "SL": "cross",
                "TSL": "square"
            }

            for _, row in df_trades.iterrows():
                direction = str(row.get("Direction", "BUY")).upper()
                entry_dt = row["Entry Time"]
                exit_dt = row["Exit Time"]
                entry_price = float(row.get("Nifty Enter Price", 0.0))
                exit_price = float(row.get("Nifty Exit Price", 0.0))
                lot_size = float(row.get("Lot Size", 1.0))
                pnl_1qty = float(row.get("1 QTY PnL", 0.0))
                sim_pnl = float(row.get("Sim PnL", pnl_1qty * lot_size))
                exit_reason = str(row.get("Exit Reason", "EOD")).upper()
                temp_val = float(row.get("Temperature", 1.0))

                # Signal was on Candle T (5 min before Entry Time)
                sig_dt = entry_dt - pd.Timedelta(minutes=5)

                t_sig = to_chart_epoch(sig_dt)
                t_entry = to_chart_epoch(entry_dt)
                t_exit = to_chart_epoch(exit_dt)

                # 1. ALWAYS Display Signal Marker on Candle T
                sig_key = (t_sig, "signal")
                if sig_key not in seen_candles:
                    seen_candles.add(sig_key)
                    markers.append({
                        "time": t_sig,
                        "position": "belowBar" if direction == "BUY" else "aboveBar",
                        "color": "#10b981" if direction == "BUY" else "#ef4444",
                        "shape": "arrowUp" if direction == "BUY" else "arrowDown",
                        "text": "BUY" if direction == "BUY" else "SELL",
                        "detail": f"{direction} SIGNAL on Candle T @ {sig_dt.strftime('%H:%M')}",
                        "size": 1
                    })

                # Check Specific Time Filter from simulate_risk_rules.py
                entry_hour = entry_dt.hour
                entry_day = entry_dt.day_name()
                is_entered = is_smart_time(entry_hour, entry_day) and (lot_size > 0)

                if is_entered:
                    cum_net_pnl = round(cum_net_pnl + sim_pnl, 2)

                    # 2. Entry Point Marker on Candle T+1 (Only if allowed by time filter)
                    entry_key = (t_entry, f"entry_{len(self.closed_trades)}")
                    if entry_key not in seen_candles:
                        seen_candles.add(entry_key)
                        markers.append({
                            "time": t_entry,
                            "position": "belowBar" if direction == "BUY" else "aboveBar",
                            "color": "#06b6d4",
                            "shape": "circle",
                            "text": "ENTRY",
                            "detail": f"ENTRY POINT (T+1): {direction} @ {entry_price:.1f} ({entry_dt.strftime('%H:%M')})",
                            "size": 1
                        })

                    # 3. Exit Marker with Total Net PnL
                    exit_key = (t_exit, f"exit_{len(self.closed_trades)}")
                    if exit_key not in seen_candles:
                        seen_candles.add(exit_key)
                        pnl_str = f"+{sim_pnl:.1f}" if sim_pnl > 0 else f"{sim_pnl:.1f}"
                        cum_str = f"+{cum_net_pnl:.1f}" if cum_net_pnl > 0 else f"{cum_net_pnl:.1f}"
                        markers.append({
                            "time": t_exit,
                            "position": "aboveBar",
                            "color": exit_colors.get(exit_reason, "#f59e0b"),
                            "shape": exit_shapes.get(exit_reason, "square"),
                            "text": exit_reason,
                            "detail": f"{exit_reason} EXIT: {direction} @ {exit_price:.1f} | {int(lot_size)} Lot(s) | Trade PnL: {pnl_str} pts | Total Net PnL: {cum_str} pts",
                            "size": 1
                        })

                    self.closed_trades.append({
                        "id": len(self.closed_trades) + 1,
                        "direction": direction,
                        "signal_time": sig_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "1_qty_pnl": pnl_1qty,
                        "lot_size": lot_size,
                        "pnl_points": sim_pnl,
                        "cumulative_net_pnl": cum_net_pnl,
                        "exit_reason": exit_reason,
                        "temperature": temp_val,
                        "entered": True
                    })
                else:
                    # Filtered trade (not entered due to time filter): PnL is strictly 0.0, but counts as 1 trade slot!
                    self.closed_trades.append({
                        "id": len(self.closed_trades) + 1,
                        "direction": direction,
                        "signal_time": sig_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "1_qty_pnl": 0.0,
                        "lot_size": 0.0,
                        "pnl_points": 0.0,
                        "cumulative_net_pnl": cum_net_pnl,
                        "exit_reason": f"{exit_reason} (TIME FILTERED)",
                        "temperature": temp_val,
                        "entered": False
                    })

            if self.closed_trades:
                self.thermal_sizer.T = self.closed_trades[-1]["temperature"]
            markers.sort(key=lambda x: x["time"])
            self._historical_markers = markers
            print(f"[LiveFeedService] Loaded {len(markers)} markers and {len(self.closed_trades)} trades (max 4 concurrent slots, Thermal T={self.thermal_sizer.T:.2f}).")
        except Exception as e:
            print(f"[LiveFeedService] Error loading historical markers: {e}")

    def _evaluate_strategy_on_candles(self, df_candles):
        """Runs the 3 models on the latest candles window to get current regime and probabilities."""
        if self.prediction_engine is None or len(df_candles) < 150:
            return 0, "HOLD", "--"

        try:
            window_df = df_candles.iloc[-500:].copy().reset_index(drop=True)
            if (window_df['volume'] == 0).all():
                window_df['volume'] = 1000000.0

            preds = self.prediction_engine.predict_latest(window_df)
            hmm_regime = preds.get("hmm_regime_name", "compression")
            gbm_prob_buy = float(preds.get("gbm_prob_buy", 0.5))
            gbm_prob_sell = float(preds.get("gbm_prob_sell", 0.5))
            tcn_pred = preds.get("tcn_predicted", "HOLD")
            tcn_prob_buy = float(preds.get("tcn_prob_buy", 0.33))
            tcn_prob_sell = float(preds.get("tcn_prob_sell", 0.33))

            gbm_buy = (preds.get("gbm_predicted") == 1) and (gbm_prob_buy >= GBM_THRESHOLD)
            gbm_sell = (preds.get("gbm_predicted") == 0) and (gbm_prob_sell >= GBM_THRESHOLD)
            tcn_buy = (tcn_pred == 'BUY') and (tcn_prob_buy >= TCN_THRESHOLD)
            tcn_sell = (tcn_pred == 'SELL') and (tcn_prob_sell >= TCN_THRESHOLD)

            buy_consensus = gbm_buy and tcn_buy
            sell_consensus = gbm_sell and tcn_sell

            blocked_regimes_buy = {'compression', 'expansiondown', 'distributiondown', 'markdown'}
            blocked_regimes_sell = {'compression', 'expansionup', 'distributionup', 'markup'}

            signal = 0
            consensus_str = "HOLD"
            if buy_consensus and not sell_consensus and (hmm_regime not in blocked_regimes_buy):
                signal = 1
                consensus_str = "BUY"
            elif sell_consensus and not buy_consensus and (hmm_regime not in blocked_regimes_sell):
                signal = -1
                consensus_str = "SELL"

            last_ts = str(df_candles.iloc[-1]["timestamp"])
            self.latest_prediction = {
                "gbm_predicted": preds.get("gbm_predicted", 0),
                "gbm_prob_buy": round(gbm_prob_buy, 4),
                "gbm_prob_sell": round(gbm_prob_sell, 4),
                "tcn_predicted": tcn_pred,
                "tcn_prob_buy": round(tcn_prob_buy, 4),
                "tcn_prob_sell": round(tcn_prob_sell, 4),
                "hmm_regime": hmm_regime,
                "consensus": consensus_str,
                "signal": signal,
                "timestamp": last_ts
            }
            self.feature_contributions = preds.get("feature_contributions", {})
            return signal, consensus_str, last_ts

        except Exception as e:
            print(f"[LiveFeedService] Prediction evaluation error: {e}")
            return 0, "HOLD", "--"

    def get_5min_bucket(self, dt):
        minute = (dt.minute // 5) * 5
        return dt.replace(minute=minute, second=0, microsecond=0)

    def process_tick(self, tick_time, ltp, volume=0, token="99926000", name="NIFTY 50", force_process=False):
        """Central tick processor called for every incoming WebSocket tick."""
        # Only process ticks during market hours (09:15 to 15:30 IST) unless force_process is True
        t_val = tick_time.hour * 100 + tick_time.minute
        # Allow pre-market or live market, but discard off-market ticks unless force_process is set
        if not force_process and (t_val < 900 or t_val > 1535):
            return

        self.tick_count += 1
        self.last_tick_time = tick_time
        self.index_ltp = ltp

        direction = "same"
        change_pts = 0.0
        if self.last_ltp is not None:
            change_pts = round(ltp - self.last_ltp, 2)
            if change_pts > 0:
                direction = "up"
            elif change_pts < 0:
                direction = "down"
        self.last_ltp = ltp

        tick_entry = {
            "id": self.tick_count,
            "time": tick_time.strftime("%H:%M:%S.%f")[:-3],
            "datetime": tick_time.strftime("%Y-%m-%d %H:%M:%S"),
            "token": str(token),
            "name": name,
            "ltp": round(ltp, 2),
            "change": change_pts,
            "direction": direction,
            "volume": volume
        }
        self.ticks_buffer.append(tick_entry)

        # 1. Evaluate Live Active Trade Exits against this tick!
        self._evaluate_live_trade_exit(ltp, tick_time)

        # 1b. Evaluate Pattern Match Engine Stop Loss (80-pt SL invalidation)
        if hasattr(self, "pattern_engine") and self.pattern_engine is not None:
            self.pattern_engine.on_tick(ltp, tick_time)

        # 2. Update or advance 5-minute candle
        bucket = self.get_5min_bucket(tick_time)
        if self.current_bucket is None or bucket != self.current_bucket:
            if self.current_candle is not None:
                self._finalize_candle(self.current_candle)

            self.current_bucket = bucket
            bucket_str = bucket.strftime("%Y-%m-%d %H:%M:%S")
            t_epoch = to_chart_epoch(bucket)
            self.current_candle = {
                "time": t_epoch,
                "bucket": bucket,
                "timestamp": bucket_str,
                "open": ltp,
                "high": ltp,
                "low": ltp,
                "close": ltp,
                "volume": volume,
                "ticks": 1
            }

            # Candle T + 1: EXECUTE ENTRY AT OPEN!
            if self.pending_entry is not None:
                self._execute_pending_entry(self.pending_entry, ltp, bucket_str, bucket)
                self.pending_entry = None

        else:
            c = self.current_candle
            c["time"] = to_chart_epoch(bucket)
            c["high"] = max(c["high"], ltp)
            c["low"] = min(c["low"], ltp)
            c["close"] = ltp
            c["volume"] += volume
            c["ticks"] += 1

    def _finalize_candle(self, candle):
        """Closes a 5-minute candle, appends to store, and runs 3 ML models for signal consensus."""
        c_vol = float(candle.get("volume", 0.0))
        if c_vol <= 0:
            temp_df = pd.DataFrame([{
                "timestamp": candle["timestamp"],
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": 0.0
            }])
            temp_df = apply_constituent_volume_sum(temp_df)
            c_vol = float(temp_df.iloc[0]["volume"])
            candle["volume"] = c_vol

        c_ts = pd.to_datetime(candle["timestamp"])
        new_row = pd.DataFrame([{
            "timestamp": c_ts,
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": c_vol
        }])
        
        # Prevent duplicate row creation if timestamp matches last row
        if not self.candles_df.empty:
            last_ts = self.candles_df.iloc[-1]["timestamp"]
            if str(last_ts) == str(c_ts):
                idx = self.candles_df.index[-1]
                self.candles_df.at[idx, "open"] = float(candle["open"])
                self.candles_df.at[idx, "high"] = float(candle["high"])
                self.candles_df.at[idx, "low"] = float(candle["low"])
                self.candles_df.at[idx, "close"] = float(candle["close"])
                self.candles_df.at[idx, "volume"] = c_vol
            else:
                self.candles_df = pd.concat([self.candles_df, new_row], ignore_index=True)
        else:
            self.candles_df = pd.concat([self.candles_df, new_row], ignore_index=True)

        print(f"[Candle Closed] {candle['timestamp']} | O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']} Constituent Vol:{candle['volume']:,} Ticks:{candle['ticks']}")

        # Feed 5-min candle to Pattern Match Engine (temp 15m in-memory aggregation)
        if hasattr(self, "pattern_engine") and self.pattern_engine is not None:
            macro_info = {
                "prev_close": self.prev_close,
                "close_5d": getattr(self, "close_5d", 0.0)
            }
            self.pattern_engine.on_5min_candle_closed(candle, macro_info)

        signal, consensus_str, last_ts = self._evaluate_strategy_on_candles(self.candles_df)

        if signal != 0:
            # Concurrency limit: Maximum 4 concurrent active positions
            if len(self.active_positions) < 4 and self.pending_entry is None:
                print(f"🚨 [SIGNAL GENERATED on Candle T] {consensus_str} at {candle['timestamp']} (Close: {candle['close']})")
                t_epoch = to_chart_epoch(candle["timestamp"])

                sig_marker = {
                    "time": t_epoch,
                    "position": "belowBar" if signal == 1 else "aboveBar",
                    "color": "#10b981" if signal == 1 else "#ef4444",
                    "shape": "arrowUp" if signal == 1 else "arrowDown",
                    "text": "BUY" if signal == 1 else "SELL",
                    "detail": f"{consensus_str} SIGNAL on Candle T (Price: {candle['close']:.1f})",
                    "size": 1
                }
                self.chart_markers.append(sig_marker)

                sig_info = {
                    "id": len(self.signals) + 1,
                    "time": candle["timestamp"],
                    "type": consensus_str,
                    "candle_time": candle["timestamp"],
                    "close": candle["close"],
                    "regime": self.latest_prediction["hmm_regime"],
                    "gbm_prob": self.latest_prediction["gbm_prob_buy"] if signal == 1 else self.latest_prediction["gbm_prob_sell"],
                    "tcn_prob": self.latest_prediction["tcn_prob_buy"] if signal == 1 else self.latest_prediction["tcn_prob_sell"],
                    "status": "WAITING FOR ENTRY (T+1 Candle Open)"
                }
                self.signals.append(sig_info)

                # Schedule ENTRY on next candle (Candle T+1) at open
                self.pending_entry = {
                    "direction": consensus_str,
                    "signal_candle_time": candle["timestamp"],
                    "regime": self.latest_prediction["hmm_regime"],
                    "signal_id": sig_info["id"]
                }

    @property
    def active_position(self):
        return self.active_positions[-1] if self.active_positions else None

    @active_position.setter
    def active_position(self, val):
        if val is None:
            self.active_positions = []
        else:
            self.active_positions.append(val)

    @property
    def total_net_pnl(self):
        return round(sum(t.get("pnl_points", 0.0) for t in self.closed_trades if t.get("entered", True)), 2)

    def _execute_pending_entry(self, pending, open_price, bucket_str, bucket_dt):
        """Executes entry on Candle T+1 at open price and marks ENTRY POINT on chart and board."""
        direction = pending["direction"]
        regime = pending.get("regime", "compression")

        # Specific Time Filter from simulate_risk_rules.py
        entry_hour = bucket_dt.hour
        entry_day = bucket_dt.strftime("%A")
        is_entered = is_smart_time(entry_hour, entry_day)
        lot_size = float(self.thermal_sizer.get_lot_size(entry_hour, entry_day, direction, open_price)) if is_entered else 0.0

        self.trade_counter += 1

        if is_entered:
            print(f"🎯 [ENTRY POINT EXECUTED on Candle T+1] {direction} @ {open_price} ({int(lot_size)} Lot(s), T={self.thermal_sizer.T:.2f}) at {bucket_str}")
            t_epoch = to_chart_epoch(bucket_dt)

            entry_marker = {
                "time": t_epoch,
                "position": "belowBar" if direction == "BUY" else "aboveBar",
                "color": "#06b6d4",
                "shape": "circle",
                "text": "ENTRY",
                "detail": f"ENTRY POINT (T+1): {direction} @ {open_price:.1f} ({int(lot_size)} Lot(s))",
                "size": 1
            }
            self.chart_markers.append(entry_marker)
            for s in self.signals:
                if s["id"] == pending.get("signal_id"):
                    s["status"] = f"ENTERED @ {open_price:.1f} ({int(lot_size)} Lot(s))"
                    s["entry_time"] = bucket_str
                    s["entry_price"] = open_price
                    s["lot_size"] = lot_size
        else:
            print(f"🕒 [TIME FILTER - SLOT USED] {direction} at {bucket_str} ({entry_day} {entry_hour:02d}:00) - Occupies 1 trade slot, 0 Net PnL")
            for s in self.signals:
                if s["id"] == pending.get("signal_id"):
                    s["status"] = f"TIME FILTERED (SLOT USED) ({entry_day} {entry_hour:02d}:00)"
                    s["entry_time"] = bucket_str
                    s["entry_price"] = open_price
                    s["lot_size"] = 0.0

        if regime in ['markup', 'markdown', 'expansionup', 'expansiondown']:
            tp_pts = 200.0
        else:
            tp_pts = 100.0

        sl_level = open_price - SL_POINTS if direction == "BUY" else open_price + SL_POINTS
        tp_level = open_price + tp_pts if direction == "BUY" else open_price - tp_pts

        pos_entry = {
            "id": self.trade_counter,
            "direction": direction,
            "entry_time": bucket_str,
            "entry_price": open_price,
            "current_price": open_price,
            "high_since_entry": open_price,
            "low_since_entry": open_price,
            "sl_level": sl_level,
            "tp_level": tp_level,
            "tsl_points": TSL_POINTS,
            "pnl_points": 0.0,
            "regime": regime,
            "signal_candle_time": pending["signal_candle_time"],
            "lot_size": lot_size,
            "temperature": round(self.thermal_sizer.T, 2),
            "entered": is_entered
        }
        self.active_positions.append(pos_entry)

    def _evaluate_live_trade_exit(self, ltp, tick_time):
        """Checks all active positions on every tick against SL, TP, TSL, EOD exit."""
        if not self.active_positions:
            return

        remaining_positions = []
        for pos in self.active_positions:
            direction = pos["direction"]
            entry_price = pos["entry_price"]
            is_entered = pos.get("entered", True)

            if direction == "BUY":
                pos["high_since_entry"] = max(pos["high_since_entry"], ltp)
                pos["low_since_entry"] = min(pos["low_since_entry"], ltp)
                pos["pnl_points"] = round(ltp - entry_price, 2)
            else:
                pos["high_since_entry"] = max(pos["high_since_entry"], ltp)
                pos["low_since_entry"] = min(pos["low_since_entry"], ltp)
                pos["pnl_points"] = round(entry_price - ltp, 2)

            pos["current_price"] = ltp

            exited = False
            exit_reason = ""
            exit_price = ltp

            # 1. Stop Loss (SL)
            if direction == "BUY" and ltp <= pos["sl_level"]:
                exited = True
                exit_reason = "SL"
                exit_price = pos["sl_level"]
            elif direction == "SELL" and ltp >= pos["sl_level"]:
                exited = True
                exit_reason = "SL"
                exit_price = pos["sl_level"]

            # 2. Take Profit (TP)
            if not exited:
                if direction == "BUY" and ltp >= pos["tp_level"]:
                    exited = True
                    exit_reason = "TP"
                    exit_price = pos["tp_level"]
                elif direction == "SELL" and ltp <= pos["tp_level"]:
                    exited = True
                    exit_reason = "TP"
                    exit_price = pos["tp_level"]

            # 3. Trailing Stop Loss (TSL)
            if not exited and TSL_POINTS is not None:
                if direction == "BUY":
                    tsl_trigger = pos["high_since_entry"] - TSL_POINTS
                    if ltp <= tsl_trigger:
                        if not TSL_ONLY_IN_PROFIT or tsl_trigger > entry_price:
                            exited = True
                            exit_reason = "TSL"
                            exit_price = tsl_trigger
                else:
                    tsl_trigger = pos["low_since_entry"] + TSL_POINTS
                    if ltp >= tsl_trigger:
                        if not TSL_ONLY_IN_PROFIT or tsl_trigger < entry_price:
                            exited = True
                            exit_reason = "TSL"
                            exit_price = tsl_trigger

            # 4. EOD Force Exit at 15:10 IST
            if not exited:
                if tick_time.hour == FORCE_EXIT_HOUR and tick_time.minute >= FORCE_EXIT_MINUTE:
                    exited = True
                    exit_reason = "EOD"
                    exit_price = ltp

            if exited:
                exit_time_str = tick_time.strftime("%Y-%m-%d %H:%M:%S")

                if is_entered:
                    lot_size = float(pos.get("lot_size", 1.0))
                    pnl_1qty = round((exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price), 2)
                    pnl_final = round(pnl_1qty * lot_size, 2)

                    # Update thermal dissipation sizer
                    self.thermal_sizer.record_outcome(pnl_1qty, entry_price)
                    temp_after = round(self.thermal_sizer.T, 2)

                    cum_net_pnl = sum(t.get("pnl_points", 0.0) for t in self.closed_trades if t.get("entered", True)) + pnl_final
                    cum_net_pnl = round(cum_net_pnl, 2)

                    print(f"🛑 [LIVE EXIT HIT] {exit_reason} for {direction} @ {exit_price:.1f} | {int(lot_size)} Lot(s) | Trade PnL: {pnl_final:+.1f} pts | Total Net PnL: {cum_net_pnl:+.1f} pts (T={temp_after:.2f})")

                    t_epoch = to_chart_epoch(tick_time)
                    pnl_str = f"+{pnl_final:.1f}" if pnl_final > 0 else f"{pnl_final:.1f}"
                    cum_str = f"+{cum_net_pnl:.1f}" if cum_net_pnl > 0 else f"{cum_net_pnl:.1f}"

                    exit_colors = {
                        "EOD": "#f59e0b",
                        "TP": "#10b981",
                        "SL": "#ef4444",
                        "TSL": "#a855f7"
                    }
                    exit_shapes = {
                        "EOD": "square",
                        "TP": "circle",
                        "SL": "cross",
                        "TSL": "square"
                    }

                    exit_marker = {
                        "time": t_epoch,
                        "position": "aboveBar",
                        "color": exit_colors.get(exit_reason, "#f59e0b"),
                        "shape": exit_shapes.get(exit_reason, "square"),
                        "text": exit_reason,
                        "detail": f"{exit_reason} EXIT: {direction} @ {exit_price:.1f} | {int(lot_size)} Lot(s) | Trade PnL: {pnl_str} pts | Total Net PnL: {cum_str} pts",
                        "size": 1
                    }
                    self.chart_markers.append(exit_marker)

                    closed_entry = {
                        "id": pos["id"],
                        "direction": direction,
                        "entry_time": pos["entry_time"],
                        "exit_time": exit_time_str,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "1_qty_pnl": pnl_1qty,
                        "lot_size": lot_size,
                        "pnl_points": pnl_final,
                        "cumulative_net_pnl": cum_net_pnl,
                        "exit_reason": exit_reason,
                        "temperature": temp_after,
                        "entered": True
                    }
                    self.closed_trades.append(closed_entry)
                else:
                    # Filtered trade: strictly 0.0 pts PnL!
                    closed_entry = {
                        "id": pos["id"],
                        "direction": direction,
                        "entry_time": pos["entry_time"],
                        "exit_time": exit_time_str,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "1_qty_pnl": 0.0,
                        "lot_size": 0.0,
                        "pnl_points": 0.0,
                        "cumulative_net_pnl": self.total_net_pnl,
                        "exit_reason": f"{exit_reason} (TIME FILTERED)",
                        "temperature": round(self.thermal_sizer.T, 2),
                        "entered": False
                    }
                    self.closed_trades.append(closed_entry)
            else:
                remaining_positions.append(pos)

        self.active_positions = remaining_positions

    def get_chart_data(self, limit=1000):
        """
        Returns candles and markers formatted strictly for past 3 months and Indian Market Hours (09:15 to 15:30).
        Timestamps display accurately between 9 to 4!
        """
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(days=90)

        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.candles_df["timestamp"]):
            self.candles_df["timestamp"] = pd.to_datetime(self.candles_df["timestamp"], format="mixed")

        # Filter strictly past 3 months
        df_slice = self.candles_df[self.candles_df["timestamp"] >= cutoff].copy()

        # Filter strictly Indian Market Hours (09:15 to 15:30)
        df_slice["time_val"] = df_slice["timestamp"].dt.hour * 100 + df_slice["timestamp"].dt.minute
        df_slice = df_slice[(df_slice["time_val"] >= 915) & (df_slice["time_val"] <= 1530)].copy()

        records = []
        for _, row in df_slice.iterrows():
            ts = row["timestamp"]
            t_epoch = to_chart_epoch(ts)
            records.append({
                "time": t_epoch,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0))
            })

        # Format current forming candle
        forming_candle = None
        if self.current_candle is not None:
            c = self.current_candle
            t_epoch = to_chart_epoch(c["bucket"])
            forming_candle = {
                "time": t_epoch,
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"]),
                "ticks": int(c["ticks"])
            }

        # Combine historical markers and live markers (filtered to cutoff)
        # Deduplicate strictly by (time, shape) to avoid duplicate markers on same candle
        cutoff_epoch = to_chart_epoch(cutoff)
        all_markers_map = {}
        for m in self._historical_markers:
            if m["time"] >= cutoff_epoch:
                all_markers_map[(m["time"], m.get("shape", ""))] = m
        for m in self.chart_markers:
            if m["time"] >= cutoff_epoch:
                all_markers_map[(m["time"], m.get("shape", ""))] = m

        markers = list(all_markers_map.values())
        markers.sort(key=lambda x: x["time"])

        return {
            "candles": records,
            "forming_candle": forming_candle,
            "markers": markers,
            "pattern_line": self.get_pattern_line_state()
        }

    def get_market_breadth(self):
        """Calculates live advance/decline stats and top movers across 50 constituents."""
        advances = 0
        declines = 0
        unchanged = 0
        movers = []

        for token, stat in self.constituent_stats.items():
            chg = stat.get("change", 0.0)
            chg_pct = stat.get("change_pct", 0.0)
            ltp = stat.get("ltp", 0.0)
            if chg > 0:
                advances += 1
            elif chg < 0:
                declines += 1
            else:
                unchanged += 1

            if ltp > 0:
                movers.append({
                    "name": stat["name"],
                    "ltp": ltp,
                    "change": chg,
                    "change_pct": chg_pct
                })

        total = len(self.constituent_stats) or 50
        active_total = advances + declines + unchanged
        adv_pct = round((advances / active_total) * 100, 1) if active_total > 0 else 50.0
        dec_pct = round((declines / active_total) * 100, 1) if active_total > 0 else 50.0

        movers.sort(key=lambda x: x["change_pct"], reverse=True)
        top_gainers = [m for m in movers if m["change_pct"] > 0][:4]
        top_losers = [m for m in movers if m["change_pct"] < 0][-4:][::-1]

        if advances >= 32:
            sentiment = "STRONGLY BULLISH"
            sentiment_color = "#10b981"
        elif advances >= 27:
            sentiment = "MODERATELY BULLISH"
            sentiment_color = "#34d399"
        elif declines >= 32:
            sentiment = "STRONGLY BEARISH"
            sentiment_color = "#ef4444"
        elif declines >= 27:
            sentiment = "MODERATELY BEARISH"
            sentiment_color = "#fb7185"
        else:
            sentiment = "NEUTRAL / BALANCED"
            sentiment_color = "#94a3b8"

        return {
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "total": total,
            "adv_pct": adv_pct,
            "dec_pct": dec_pct,
            "sentiment": sentiment,
            "sentiment_color": sentiment_color,
            "top_gainers": top_gainers,
            "top_losers": top_losers
        }

    def get_thermal_sizer_state(self):
        """Returns the dynamic thermodynamic state of the Thermal Dissipation Lot Sizer."""
        T = round(self.thermal_sizer.T, 2)
        lots = int(round(T))
        lots = max(1, min(4, lots))
        if T > 1.25:
            status = "HEATING (+α PnL WIN RUN)"
            status_color = "#f97316"
        elif T < 0.95:
            status = "COOLING (-β ΔT DISSIPATION)"
            status_color = "#06b6d4"
        else:
            status = "AMBIENT STABLE"
            status_color = "#10b981"

        return {
            "temperature": T,
            "t_min": self.thermal_sizer.T_min,
            "t_max": self.thermal_sizer.T_max,
            "t_ambient": self.thermal_sizer.T_ambient,
            "alpha": self.thermal_sizer.alpha,
            "beta": self.thermal_sizer.beta,
            "recommended_lots": lots,
            "recommended_qty": lots * 65,
            "status": status,
            "status_color": status_color
        }

    def get_performance_metrics(self):
        """Calculates institutional risk & performance metrics for strictly entered trades only."""
        # Strictly consider only trades that were actually entered (exclude unentered/time-filtered trades)
        entered_trades = [
            t for t in self.closed_trades
            if t.get("entered", False) is True
            and t.get("lot_size", 0.0) > 0
            and abs(t.get("pnl_points", 0.0)) > 1e-6
            and "TIME FILTERED" not in str(t.get("exit_reason", "")).upper()
        ]

        wins = [t for t in entered_trades if t.get("pnl_points", 0.0) > 0]
        losses = [t for t in entered_trades if t.get("pnl_points", 0.0) < 0]

        win_count = len(wins)
        loss_count = len(losses)
        total_trades = win_count + loss_count
        filtered_trades = len(self.closed_trades) - total_trades

        if total_trades == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "max_dd_pts": 0.0,
                "max_dd_inr": 0.0,
                "recovery_factor": 0.0,
                "avg_lot_size": 1.0,
                "net_pnl_pts": 0.0,
                "net_pnl_inr": 0.0,
                "filtered_skipped": filtered_trades
            }

        win_rate = round((win_count / total_trades) * 100, 1)

        gross_profit = round(sum(t.get("pnl_points", 0.0) for t in wins), 2)
        gross_loss = round(abs(sum(t.get("pnl_points", 0.0) for t in losses)), 2)

        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = 99.99
        else:
            profit_factor = 0.0

        # Calculate Max Drawdown across strictly entered trades
        peak = 0.0
        max_dd = 0.0
        running_equity = 0.0

        for t in entered_trades:
            running_equity += t.get("pnl_points", 0.0)
            if running_equity > peak:
                peak = running_equity
            dd = peak - running_equity
            if dd > max_dd:
                max_dd = dd

        max_dd_pts = round(max_dd, 2)
        max_dd_inr = round(max_dd_pts * 65.0, 2)

        net_pnl_pts = round(running_equity, 2)
        net_pnl_inr = round(net_pnl_pts * 65.0, 2)

        if max_dd_pts > 0:
            recovery_factor = round(net_pnl_pts / max_dd_pts, 2)
        elif net_pnl_pts > 0:
            recovery_factor = 99.99
        else:
            recovery_factor = 0.0

        avg_lot = round(sum(t.get("lot_size", 1.0) for t in entered_trades) / total_trades, 2)

        return {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "max_dd_pts": max_dd_pts,
            "max_dd_inr": max_dd_inr,
            "recovery_factor": recovery_factor,
            "avg_lot_size": avg_lot,
            "net_pnl_pts": net_pnl_pts,
            "net_pnl_inr": net_pnl_inr,
            "filtered_skipped": filtered_trades
        }

    def get_pattern_line_state(self):
        """Returns the current 15M historical pattern projection line state."""
        if hasattr(self, "pattern_engine") and self.pattern_engine is not None:
            return self.pattern_engine.get_state()
        return {
            "active": False,
            "drawn_today": False,
            "direction": "NONE",
            "entry_time": None,
            "entry_time_str": None,
            "entry_price": 0.0,
            "target_time": None,
            "target_time_str": None,
            "target_price": 0.0,
            "spike_pct": 0.0,
            "target_price_2": 0.0,
            "angle_pct_2": 0.0,
            "flatline_price": 0.0,
            "diff_score": 0.0,
            "consensus": "0/5",
            "invalidated": False,
            "invalidation_reason": None,
            "status_text": "Pattern Engine Initializing"
        }

    def start_streaming(self):
        """Connects to Angel One SmartWebSocketV2 and streams ticks indefinitely."""
        if self.is_running:
            return

        self.is_running = True
        t = threading.Thread(target=self._stream_worker, daemon=True)
        t.start()

        tf = threading.Thread(target=self._flusher_worker, daemon=True)
        tf.start()

        ts = threading.Thread(target=self._simulated_tick_worker, daemon=True)
        ts.start()

    def _simulated_tick_worker(self):
        """Fallback synthetic tick generator thread when live WebSocket is offline or without API keys."""
        import random
        sim_time = None
        while self.is_running:
            # If live WebSocket is active, yield to real ticks
            if self.connection_status == "live":
                time.sleep(2)
                continue

            self.connection_status = "simulated"
            if sim_time is None:
                if not self.candles_df.empty and "timestamp" in self.candles_df.columns:
                    last_ts = self.candles_df.iloc[-1]["timestamp"]
                    if isinstance(last_ts, str):
                        sim_time = pd.to_datetime(last_ts)
                    else:
                        sim_time = last_ts
                else:
                    sim_time = datetime.datetime.now()

            # Advance simulated time by 5-10 seconds per tick loop
            sim_time += datetime.timedelta(seconds=random.randint(5, 10))

            # Keep simulated time within 09:15 to 15:30 IST market hours
            if sim_time.hour > 15 or (sim_time.hour == 15 and sim_time.minute > 30):
                sim_time = sim_time.replace(hour=9, minute=15, second=0) + datetime.timedelta(days=1)
            elif sim_time.hour < 9 or (sim_time.hour == 9 and sim_time.minute < 15):
                sim_time = sim_time.replace(hour=9, minute=15, second=0)

            # Generate realistic micro fluctuation around current LTP
            base_p = self.index_ltp if (self.index_ltp and self.index_ltp > 0) else 23372.4
            noise = random.choice([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
            sim_ltp = round(base_p + noise, 2)
            sim_vol = random.randint(100, 3000)

            # Process simulated tick with force_process=True
            try:
                self.process_tick(sim_time, sim_ltp, volume=sim_vol, force_process=True)
            except Exception as ex:
                pass
            time.sleep(1.0)

    def _stream_worker(self):
        """Worker thread connecting to WebSocket with auto-reconnection."""
        while self.is_running:
            try:
                self.connection_status = "connecting"
                print("[LiveFeedService] Generating Angel One session for WebSocket...")
                self.smart_connect = SmartConnect(api_key=self.api_key)
                totp = pyotp.TOTP(self.totp_secret).now()
                sess = self.smart_connect.generateSession(self.client_id, self.password, totp)
                if not sess.get("status"):
                    raise RuntimeError(sess.get("message", "Session failed"))

                jwt_token = sess["data"]["jwtToken"]
                feed_token = self.smart_connect.getfeedToken()
                print("[LiveFeedService] Session created. Connecting SmartWebSocketV2...")

                sws = SmartWebSocketV2(jwt_token, self.api_key, self.client_id, feed_token)

                def on_open(wsapp):
                    self.connection_status = "live"
                    print(f"[LiveFeedService] WebSocket connected! Subscribing to NIFTY 50 Spot + {len(CONSTITUENT_NAMES)} Constituent Stocks in Quote Mode (mode=2)...")
                    all_tokens = ["99926000", "26000"] + list(CONSTITUENT_NAMES.keys())
                    token_list = [
                        {"exchangeType": 1, "tokens": all_tokens}
                    ]
                    sws.subscribe(correlation_id="live_stream_sub", mode=2, token_list=token_list)

                def on_data(wsapp, msg):
                    if not msg or not isinstance(msg, dict):
                        return
                    token = str(msg.get("token", ""))
                    ltp_raw = msg.get("last_traded_price")
                    if ltp_raw is None:
                        return
                    ltp = float(ltp_raw) / 100.0
                    vol = int(msg.get("volume_trade_for_the_day", 0) or 0)

                    tick_time = datetime.datetime.now()
                    exch_ts = msg.get("exchange_timestamp")
                    if exch_ts:
                        try:
                            if exch_ts > 10**12:
                                tick_time = datetime.datetime.fromtimestamp(exch_ts / 1000.0)
                            else:
                                tick_time = datetime.datetime.fromtimestamp(exch_ts)
                        except Exception:
                            pass

                    if token in ["99926000", "26000"]:
                        name = "NIFTY 50"
                        self.process_tick(tick_time, ltp, 0, token=token, name=name)
                    elif token in CONSTITUENT_NAMES:
                        name = CONSTITUENT_NAMES[token]
                        day_vol = int(msg.get("volume_trade_for_the_day", 0) or 0)
                        prev_vol = self.token_day_volume.get(token, None)

                        # Extract price change
                        open_raw = msg.get("open_price_of_the_day")
                        close_raw = msg.get("close_price")
                        open_p = float(open_raw) / 100.0 if open_raw else 0.0
                        prev_close = float(close_raw) / 100.0 if close_raw else 0.0

                        base_p = prev_close if prev_close > 0 else open_p
                        if base_p > 0:
                            stock_change = round(ltp - base_p, 2)
                            stock_change_pct = round((stock_change / base_p) * 100.0, 2)
                        else:
                            curr_stat = self.constituent_stats.get(token, {})
                            base_p = curr_stat.get("base_p", 0.0)
                            if base_p <= 0:
                                base_p = ltp
                            stock_change = round(ltp - base_p, 2)
                            stock_change_pct = round(((ltp - base_p) / base_p) * 100.0, 2) if base_p > 0 else 0.0

                        self.constituent_stats[token] = {
                            "name": name,
                            "ltp": round(ltp, 2),
                            "base_p": base_p,
                            "change": stock_change,
                            "change_pct": stock_change_pct
                        }

                        dir_str = "up" if stock_change > 0 else ("down" if stock_change < 0 else "same")

                        if prev_vol is not None and day_vol >= prev_vol:
                            delta_vol = day_vol - prev_vol
                        else:
                            delta_vol = int(msg.get("last_traded_quantity", 0) or 0)
                        self.token_day_volume[token] = day_vol

                        if delta_vol > 0:
                            self.constituent_volume_sum += delta_vol
                            if self.current_candle is not None:
                                self.current_candle["volume"] += delta_vol
                                self.current_candle["ticks"] += 1

                        self.ticks_buffer.append({
                            "id": self.tick_count + 1,
                            "time": tick_time.strftime("%H:%M:%S.%f")[:-3],
                            "datetime": tick_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "token": str(token),
                            "name": name,
                            "ltp": round(ltp, 2),
                            "change": stock_change,
                            "direction": dir_str,
                            "volume": delta_vol if delta_vol > 0 else (msg.get("last_traded_quantity") or 0)
                        })

                def on_error(wsapp, error):
                    self.connection_status = "error"
                    print(f"[LiveFeedService] WebSocket error: {error}")

                def on_close(wsapp, code, reason):
                    self.connection_status = "offline"
                    print(f"[LiveFeedService] WebSocket closed: code={code}, reason={reason}")

                sws.on_open = on_open
                sws.on_data = on_data
                sws.on_error = on_error
                sws.on_close = on_close

                self.ws = sws
                sws.connect()

            except Exception as e:
                self.connection_status = "error"
                print(f"[LiveFeedService] Connection loop error: {e}")

            if self.is_running:
                print("[LiveFeedService] Reconnecting in 5 seconds...")
                time.sleep(5)

    def _flusher_worker(self):
        """Force flushes 5-minute candle if time passes bucket boundary even without tick."""
        while self.is_running:
            time.sleep(2)
            try:
                now = datetime.datetime.now()
                now_bucket = self.get_5min_bucket(now)
                if self.current_bucket is not None and now_bucket > self.current_bucket:
                    if self.current_candle is not None:
                        self._finalize_candle(self.current_candle)
                        self.current_bucket = now_bucket
                        bucket_str = now_bucket.strftime("%Y-%m-%d %H:%M:%S")
                        last_c = self.current_candle["close"]
                        self.current_candle = {
                            "bucket": now_bucket,
                            "timestamp": bucket_str,
                            "open": last_c,
                            "high": last_c,
                            "low": last_c,
                            "close": last_c,
                            "volume": 0,
                            "ticks": 0
                        }
            except Exception as e:
                pass

    def stop(self):
        self.is_running = False
        self.connection_status = "offline"
        if self.ws:
            try:
                self.ws.close()
            except:
                pass

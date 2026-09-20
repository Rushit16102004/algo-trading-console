# PASSKEY: rushit2712
"""
================================================================================
15-MINUTE HISTORICAL PATTERN MATCHING DIRECTION ENGINE
================================================================================
Based on 243A/read_rule.me and 243A/every_day_model.py
Data: 243A/2008 to 2025 data 15 min.csv (18 years: 99,205 patterns)

Key Rules Enforced:
1. Operates on 15-Minute Candles (Pure OHLC).
2. Live 5-minute candles are accumulated into 15-minute buckets in a temporary
   in-memory cache only (zero disk clutter, no DB mutation).
3. 16-Dimensional Feature Vector:
   - 3-candle sequence normalized OHLC (12 values relative to Open of t-2)
   - 5-Day Trend % ((Close - Close_5d) / Close_5d * 100)
   - Opening Gap % ((Day_Open - Prev_Close) / Prev_Close * 100)
   - Intraday Session Position ((Close - Day_Low) / (Day_High - Day_Low + 1e-4))
   - Candle Body Vector ((Close - Open) / (High - Low + 1e-4))
4. Similarity Search: Query top 5 nearest historical neighbors using cKDTree.
5. Consensus: At least 4 of 5 (80%) historical neighbors must agree on direction.
6. Evaluated only between 09:45 AM and 12:30 PM (Morning Window).
7. Projection Line:
   - Line drawn once per day based on matched days' average spike %.
   - Target price = Entry Price * (1 + avg_spike%) for BUY, (1 - avg_spike%) for SELL.
   - Start: (Entry Time, Entry Price), End: (15:00, Target Price).
8. 80-Point Stop Loss Invalidation:
   - If BUY and market drops 80 pts below entry price -> Line removed immediately.
   - If SELL and market rises 80 pts above entry price -> Line removed immediately.
9. Strictly drawn only ONCE per day (no second line drawn on the same day).
================================================================================
"""

import os
import time
import datetime
import threading
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import pytz

CACHE_FILE = "backend_engine/pattern_15m_cache.npz"
HIST_CSV = "243A/2008 to 2025 data 15 min.csv"

def to_chart_epoch(dt):
    """Converts IST datetime into unix epoch matching Lightweight Charts time scale."""
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    if hasattr(dt, 'tz_localize') and dt.tzinfo is not None:
        dt = dt.tz_convert('Asia/Kolkata').tz_localize(None)
    elif hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        dt = dt.astimezone(pytz.timezone("Asia/Kolkata")).replace(tzinfo=None)
    return int(dt.replace(tzinfo=pytz.UTC).timestamp())


class PatternMatchEngine:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.tree = None
        self.h_mat = None
        self.up_spikes = None
        self.dn_spikes = None
        self.eod_dirs = None
        self.is_initialized = False

        # In-memory session state (15-min temp cache)
        self.current_session_date = None
        self.session_15m_candles = []  # [{timestamp, date_str, time_hm, open, high, low, close, volume}]
        self.acc_5m_candles = []       # Temporary buffer to roll 5-min into 15-min
        self.day_open = None
        self.day_high = None
        self.day_low = None
        self.prev_close = None
        self.close_5d = None

        # Daily Projection Line State
        self.drawn_today = False
        self.line_state = {
            "active": False,
            "drawn_today": False,
            "is_prev_day": False,
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
            "status_text": "Awaiting 09:45 AM Evaluation Window"
        }

        # Initialize historical database in a background thread or synchronously
        self._load_or_build_cache()

    def _load_or_build_cache(self):
        """Loads precomputed patterns from fast .npz cache or builds from CSV."""
        t0 = time.time()
        if os.path.exists(CACHE_FILE):
            try:
                print(f"[PatternMatchEngine] Loading precomputed patterns from {CACHE_FILE}...")
                data = np.load(CACHE_FILE, allow_pickle=True)
                self.h_mat = data["h_mat"]
                self.up_spikes = data["up_spikes"]
                self.dn_spikes = data["dn_spikes"]
                self.eod_dirs = data["eod_dirs"]
                self.tree = cKDTree(self.h_mat)
                self.is_initialized = True
                print(f"[PatternMatchEngine] Loaded {len(self.h_mat):,} patterns and built KD-Tree in {time.time() - t0:.2f}s.")
                return
            except Exception as e:
                print(f"[PatternMatchEngine] Failed loading cache: {e}. Rebuilding...")

        # Build from HIST_CSV
        if not os.path.exists(HIST_CSV):
            print(f"[PatternMatchEngine] Historical file {HIST_CSV} not found!")
            return

        print(f"[PatternMatchEngine] Building 15-min pattern library from {HIST_CSV}...")
        df = pd.read_csv(HIST_CSV)
        df["dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("dt").reset_index(drop=True)
        df["date"] = df["dt"].dt.date
        df["time_hm"] = df["dt"].dt.strftime("%H:%M")

        # Daily metrics
        daily_open = df.groupby("date")["open"].first().rename("day_open")
        daily_close = df.groupby("date")["close"].last().rename("day_close")
        daily_df = pd.DataFrame({"date": df["date"].drop_duplicates().tolist()})
        daily_df = daily_df.merge(daily_open, on="date", how="left").merge(daily_close, on="date", how="left")
        daily_df["prev_close"] = daily_df["day_close"].shift(1)
        daily_df["close_5d"] = daily_df["day_close"].shift(5)

        df = df.merge(daily_df[["date", "day_open", "prev_close", "close_5d"]], on="date", how="left")
        df["trend_5d"] = ((df["close"] - df["close_5d"]) / df["close_5d"]) * 100.0
        df["gap_pct"] = ((df["day_open"] - df["prev_close"]) / df["prev_close"]) * 100.0
        df["high_day"] = df.groupby("date")["high"].cummax()
        df["low_day"] = df.groupby("date")["low"].cummin()
        df["position"] = (df["close"] - df["low_day"]) / (df["high_day"] - df["low_day"] + 1e-4)
        df["body_vector"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-4)

        df_valid = df.dropna(subset=["trend_5d", "gap_pct", "position", "body_vector"]).reset_index(drop=True)

        vecs = []
        up_spikes = []
        dn_spikes = []
        eod_dirs = []

        for _, grp in df_valid.groupby("date"):
            grp = grp.reset_index(drop=True)
            n = len(grp)
            if n < 3:
                continue
            highs = grp["high"].to_numpy()
            lows = grp["low"].to_numpy()
            closes = grp["close"].to_numpy()
            opens = grp["open"].to_numpy()
            eod_c = closes[-1]

            for i in range(2, n):
                sub_o = opens[i-2:i+1]
                sub_h = highs[i-2:i+1]
                sub_l = lows[i-2:i+1]
                sub_c = closes[i-2:i+1]
                base_p = float(sub_o[0])

                o_norm = (sub_o - base_p) / base_p * 100.0
                h_norm = (sub_h - base_p) / base_p * 100.0
                l_norm = (sub_l - base_p) / base_p * 100.0
                c_norm = (sub_c - base_p) / base_p * 100.0

                row_t = grp.iloc[i]
                vec = np.concatenate([
                    o_norm, h_norm, l_norm, c_norm,
                    [row_t["trend_5d"], row_t["gap_pct"], row_t["position"], row_t["body_vector"]]
                ])

                cur_c = closes[i]
                if i + 1 < n:
                    forward_highs = highs[i+1:]
                    forward_lows = lows[i+1:]
                    max_h = float(np.max(forward_highs))
                    min_l = float(np.min(forward_lows))
                else:
                    max_h = cur_c
                    min_l = cur_c

                up_spike = (max_h - cur_c) / cur_c * 100.0
                dn_spike = (cur_c - min_l) / cur_c * 100.0
                e_dir = 1 if eod_c > cur_c else (-1 if eod_c < cur_c else 0)

                vecs.append(vec)
                up_spikes.append(up_spike)
                dn_spikes.append(dn_spike)
                eod_dirs.append(e_dir)

        self.h_mat = np.array(vecs, dtype=np.float64)
        self.up_spikes = np.array(up_spikes, dtype=np.float32)
        self.dn_spikes = np.array(dn_spikes, dtype=np.float32)
        self.eod_dirs = np.array(eod_dirs, dtype=np.int8)

        # Save to fast cache file
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            np.savez_compressed(
                CACHE_FILE,
                h_mat=self.h_mat,
                up_spikes=self.up_spikes,
                dn_spikes=self.dn_spikes,
                eod_dirs=self.eod_dirs
            )
            print(f"[PatternMatchEngine] Saved fast cache to {CACHE_FILE}.")
        except Exception as e:
            print(f"[PatternMatchEngine] Could not save cache: {e}")

        self.tree = cKDTree(self.h_mat)
        self.is_initialized = True
        print(f"[PatternMatchEngine] Built {len(self.h_mat):,} historical patterns in {time.time() - t0:.2f}s.")

    def reset_day_session(self, date_str):
        """Resets in-memory accumulator and daily line state for a new trading day."""
        self.current_session_date = date_str
        self.session_15m_candles.clear()
        self.acc_5m_candles.clear()
        self.day_open = None
        self.day_high = None
        self.day_low = None
        self.drawn_today = False

        # If previous day had an active line (and was not invalidated by 80-pt SL),
        # keep it visible on the chart until today's NEW signal comes!
        if self.line_state.get("active") and not self.line_state.get("invalidated"):
            self.line_state["is_prev_day"] = True
            prev_dir = self.line_state.get("direction", "BUY")
            self.line_state["status_text"] = f"Previous Day ({prev_dir}) Line Active | Awaiting Today's Signal (09:45-12:30)"
        else:
            self.line_state = {
                "active": False,
                "drawn_today": False,
                "is_prev_day": False,
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
                "status_text": "Awaiting 09:45 AM Evaluation Window"
            }

    def on_5min_candle_closed(self, candle_5m, daily_macro_info=None):
        """
        Receives closed 5-minute candle and rolls into 15-minute bucket in memory.
        candle_5m format: {timestamp, open, high, low, close, volume, ...}
        """
        if not self.is_initialized:
            return

        ts = pd.to_datetime(candle_5m["timestamp"])
        date_str = ts.strftime("%Y-%m-%d")
        time_hm = ts.strftime("%H:%M")

        # Day reset
        if self.current_session_date != date_str:
            self.reset_day_session(date_str)

        # Update macro context if provided
        if daily_macro_info:
            self.prev_close = daily_macro_info.get("prev_close", self.prev_close)
            self.close_5d = daily_macro_info.get("close_5d", self.close_5d)

        # Update session day boundaries
        c_open = float(candle_5m["open"])
        c_high = float(candle_5m["high"])
        c_low = float(candle_5m["low"])
        c_close = float(candle_5m["close"])

        if self.day_open is None:
            self.day_open = c_open
        self.day_high = max(self.day_high or c_high, c_high)
        self.day_low = min(self.day_low or c_low, c_low)

        self.acc_5m_candles.append(candle_5m)

        # 3 consecutive 5-min candles complete one 15-min candle
        if len(self.acc_5m_candles) >= 3:
            df_5m = pd.DataFrame(self.acc_5m_candles)
            candle_15m = {
                "timestamp": ts,
                "date_str": date_str,
                "time_hm": time_hm,
                "open": float(df_5m["open"].iloc[0]),
                "high": float(df_5m["high"].max()),
                "low": float(df_5m["low"].min()),
                "close": float(df_5m["close"].iloc[-1]),
                "volume": float(df_5m["volume"].sum())
            }
            self.session_15m_candles.append(candle_15m)
            self.acc_5m_candles.clear()

            print(f"[PatternMatchEngine] Completed 15m Candle #{len(self.session_15m_candles)} at {time_hm}: "
                  f"O:{candle_15m['open']:.1f} H:{candle_15m['high']:.1f} L:{candle_15m['low']:.1f} C:{candle_15m['close']:.1f}")

            # Evaluate pattern matching if within window
            self._evaluate_current_session(time_hm, candle_15m)

    def _evaluate_current_session(self, current_time_hm, latest_candle_15m):
        """
        Evaluates pattern match if time is between 09:45 AM and 12:30 PM
        and line has not already been drawn today.
        """
        # Rule: Drawn strictly ONCE per day
        if self.drawn_today:
            return

        # Rule: Evaluated between 09:45 AM and 12:30 PM
        if not ("09:45" <= current_time_hm <= "12:30"):
            return

        # Rule: Requires at least 3 completed 15-minute candles of the day (t-2, t-1, t)
        if len(self.session_15m_candles) < 3:
            return

        # Extract sequence
        sub = self.session_15m_candles[-3:]
        base_p = float(sub[0]["open"])
        if base_p <= 0:
            return

        opens = np.array([c["open"] for c in sub], dtype=np.float64)
        highs = np.array([c["high"] for c in sub], dtype=np.float64)
        lows = np.array([c["low"] for c in sub], dtype=np.float64)
        closes = np.array([c["close"] for c in sub], dtype=np.float64)

        o_norm = (opens - base_p) / base_p * 100.0
        h_norm = (highs - base_p) / base_p * 100.0
        l_norm = (lows - base_p) / base_p * 100.0
        c_norm = (closes - base_p) / base_p * 100.0

        cur_close = float(sub[-1]["close"])
        cur_open = float(sub[-1]["open"])
        cur_high = float(sub[-1]["high"])
        cur_low = float(sub[-1]["low"])

        # Condition 2: 5-Day Trend %
        if self.close_5d and self.close_5d > 0:
            trend_5d = ((cur_close - self.close_5d) / self.close_5d) * 100.0
        else:
            trend_5d = 0.0

        # Condition 3: Opening Gap %
        if self.prev_close and self.prev_close > 0 and self.day_open:
            gap_pct = ((self.day_open - self.prev_close) / self.prev_close) * 100.0
        else:
            gap_pct = 0.0

        # Condition 4: Intraday Session Position
        day_h = self.day_high if self.day_high else cur_high
        day_l = self.day_low if self.day_low else cur_low
        position = (cur_close - day_l) / (day_h - day_l + 1e-4)

        # Condition 5: Candle Body Vector
        body_vector = (cur_close - cur_open) / (cur_high - cur_low + 1e-4)

        vec = np.concatenate([
            o_norm, h_norm, l_norm, c_norm,
            [trend_5d, gap_pct, position, body_vector]
        ])

        # Query KD-Tree for 5 nearest historical neighbors
        dists, indices = self.tree.query(vec.reshape(1, -1), k=5)
        dists = dists[0]
        indices = indices[0]

        best_dist = float(dists[0])

        # Check Consensus Rule: at least 4 of 5 (80%) must agree
        neigh_dirs = self.eod_dirs[indices]
        up_count = int(np.sum(neigh_dirs == 1))
        dn_count = int(np.sum(neigh_dirs == -1))

        direction = None
        if up_count >= 4:
            direction = "BUY"
        elif dn_count >= 4:
            direction = "SELL"

        print(f"[PatternMatchEngine] Evaluation at {current_time_hm}: Consensus UP:{up_count}/5, DN:{dn_count}/5 | Best Dist: {best_dist:.3f}")

        if direction is not None:
            # Line 1: Compute Average Spike % from matched days, drawn 20% below full target
            if direction == "BUY":
                avg_spike = float(np.mean(self.up_spikes[indices]))
                raw_target_p = round(cur_close * (1.0 + avg_spike / 100.0), 2)
                raw_diff = raw_target_p - cur_close
                # Line drawn 20% below target: base + (|target - base| * 0.80)
                target_p = round(cur_close + (raw_diff * 0.80), 2)
                adj_spike_pct = round(((target_p - cur_close) / cur_close) * 100.0, 2)

                # Line 2: 0.05 angle from breakout flatline
                target_p2 = round(cur_close * (1.0 + 0.05 / 100.0), 2)
                angle_pct2 = 0.05
            else:
                avg_spike = float(np.mean(self.dn_spikes[indices]))
                raw_target_p = round(cur_close * (1.0 - avg_spike / 100.0), 2)
                raw_diff = cur_close - raw_target_p
                # Line drawn 20% below target: base - (|base - target| * 0.80)
                target_p = round(cur_close - (raw_diff * 0.80), 2)
                adj_spike_pct = round(((target_p - cur_close) / cur_close) * 100.0, 2)

                # Line 2: 0.05 angle from breakout flatline
                target_p2 = round(cur_close * (1.0 - 0.05 / 100.0), 2)
                angle_pct2 = -0.05

            entry_dt = sub[-1]["timestamp"]
            entry_epoch = to_chart_epoch(entry_dt)
            
            # EOD Target time (3:00 PM / 15:00)
            target_dt = entry_dt.replace(hour=15, minute=0, second=0, microsecond=0)
            target_epoch = to_chart_epoch(target_dt)

            # Rule: When new signal comes for line draw, previous day line is removed!
            if self.line_state.get("is_prev_day"):
                print(f"[PatternMatchEngine] New signal ({direction}) confirmed for today {self.current_session_date} at {current_time_hm}! Removing previous day line and drawing new lines.")

            self.drawn_today = True
            self.line_state = {
                "active": True,
                "drawn_today": True,
                "is_prev_day": False,
                "direction": direction,
                "entry_time": entry_epoch,
                "entry_time_str": current_time_hm,
                "entry_price": cur_close,
                "target_time": target_epoch,
                "target_time_str": "15:00",
                "target_price": target_p,
                "raw_target_price": raw_target_p,
                "spike_pct": adj_spike_pct,
                "raw_spike_pct": round(avg_spike, 2),
                # Line 2: 0.05 Angle from Breakout Flatline
                "target_price_2": target_p2,
                "angle_pct_2": angle_pct2,
                # Flatline Breakout Reference:
                "flatline_price": cur_close,
                "diff_score": round(best_dist, 3),
                "consensus": f"{up_count if direction == 'BUY' else dn_count}/5 {direction}",
                "invalidated": False,
                "invalidation_reason": None,
                "status_text": f"Pattern Match: {direction} L1: {target_p:.1f} ({adj_spike_pct:+.2f}%) | L2 (0.05°): {target_p2:.1f} | Score: {best_dist:.3f}"
            }

            print(f"[PatternMatchEngine 2 LINES DRAWN] {direction} @ {cur_close:.2f} -> L1 (20% below): {target_p:.2f} (raw: {raw_target_p:.2f}), L2 (0.05°): {target_p2:.2f} | Consensus: {self.line_state['consensus']}")

    def on_tick(self, ltp, tick_time):
        """
        Evaluates 80-Point Stop Loss Invalidation on every live tick.
        If market drops 80 pts below BUY entry or rises 80 pts above SELL entry,
        the line is immediately removed.
        """
        if not self.line_state.get("active") or self.line_state.get("invalidated"):
            return

        # Do not check 80-pt SL against previous day's completed line with today's opening ticks
        if self.line_state.get("is_prev_day"):
            return

        direction = self.line_state.get("direction")
        entry_p = self.line_state.get("entry_price", 0.0)

        if entry_p <= 0:
            return

        # 80-Point Stop Loss Check
        if direction == "BUY" and ltp <= (entry_p - 80.0):
            print(f"[PatternMatchEngine INVALIDATED] BUY Line Removed: LTP {ltp:.2f} dropped 80 pts below Entry {entry_p:.2f}")
            self.line_state["active"] = False
            self.line_state["invalidated"] = True
            self.line_state["invalidation_reason"] = f"SL HIT: LTP {ltp:.2f} <= {entry_p - 80:.2f} (-80 pts drop)"
            self.line_state["status_text"] = "Line Removed: 80-Point Stop Loss Hit"

        elif direction == "SELL" and ltp >= (entry_p + 80.0):
            print(f"[PatternMatchEngine INVALIDATED] SELL Line Removed: LTP {ltp:.2f} rose 80 pts above Entry {entry_p:.2f}")
            self.line_state["active"] = False
            self.line_state["invalidated"] = True
            self.line_state["invalidation_reason"] = f"SL HIT: LTP {ltp:.2f} >= {entry_p + 80:.2f} (+80 pts rise)"
            self.line_state["status_text"] = "Line Removed: 80-Point Stop Loss Hit"

    def get_state(self):
        """Returns the current pattern line state for API serialization."""
        return dict(self.line_state)

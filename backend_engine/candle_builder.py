# PASSKEY: rushit2712
import os
import csv
import datetime
import pandas as pd
from backend_engine.config import CANDLE_DATA_PATH

INDEX_TOKEN = 26000
CONSTITUENT_TOKENS = [
    25, 15083, 157, 236, 5900, 16669, 16675, 383, 10604, 694,
    20374, 881, 910, 1232, 7229, 1333, 467, 1363, 1394, 4963,
    11195, 1594, 1660, 11723, 1922, 2031, 10999, 22377, 17963,
    2475, 14977, 2885, 21808, 3045, 3351, 3432, 3499, 13538,
    3506, 1964, 3787, 11483, 11630, 11536, 11532, 5097, 18143,
    4306, 3456, 317
]
TOKEN_SET = set(CONSTITUENT_TOKENS)

class CandleBuilder:
    def __init__(self, on_candle_completed_cb=None, trade_logger=None, candle_data_path=None, future_token=None):
        self.on_candle_completed_cb = on_candle_completed_cb
        self.trade_logger = trade_logger
        self.candle_data_path = candle_data_path if candle_data_path is not None else CANDLE_DATA_PATH
        self.future_token = int(future_token) if future_token is not None else None
        
        # OHLC State
        self.index_ohlc = {}      # bucket -> {open, high, low, close}
        self.index_bucket = None  # current index bucket
        
        # Volume State
        self.last_volume = {}     # scrip -> cum vol at start of current bucket
        self.last_bucket = {}     # scrip -> current open bucket
        self.latest_seen = {}     # scrip -> most recent cum vol
        
        # Pending lists
        self.pending_ohlc = {}    # bucket -> {open, high, low, close}
        self.pending_volume = {}  # bucket -> {scrip: volume, ...}
        
        self.setup_csv()

    def log(self, message):
        """Helper to write to trade logger if available, otherwise print."""
        if self.trade_logger:
            self.trade_logger.log_activity(message)
        else:
            print(message)

    def setup_csv(self):
        """Initializes the CSV output file with appropriate headers."""
        if not os.path.exists(self.candle_data_path):
            os.makedirs(os.path.dirname(self.candle_data_path), exist_ok=True)
            with open(self.candle_data_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "open", "high", "low", "close", "volume"]
                )

    def get_5min_bucket(self, dt):
        """Floors datetime to the nearest 5-minute interval."""
        floored = (dt.minute // 5) * 5
        return dt.replace(minute=floored, second=0, microsecond=0)

    def process_tick(self, tick):
        """Processes a single tick from the WebSocket handler."""
        scrip = int(tick["ScripCode"])
        rec_type = tick["RecType"]
        tick_time = tick["Time"]
        bucket = self.get_5min_bucket(tick_time)

        # 1. Index OHLC Tick (RecType 'A' or 'H')
        if scrip == INDEX_TOKEN and rec_type in ("A", "H") and tick["LTP"] is not None:
            self.handle_index_ohlc(tick["LTP"], bucket)
            
        # 2a. Dynamic Future Volume Tick (if future_token is enabled)
        elif self.future_token and scrip == self.future_token and rec_type == "d" and tick["Volume64"] is not None:
            self.handle_volume(scrip, tick["Volume64"], bucket)
            
        # 2b. Fallback Constituent Stock Vol Tick
        elif not self.future_token and scrip in TOKEN_SET and rec_type == "d" and tick["Volume64"] is not None:
            self.handle_volume(scrip, tick["Volume64"], bucket)

    def handle_index_ohlc(self, ltp, bucket):
        if self.index_bucket is None:
            self.index_bucket = bucket
            self.index_ohlc[bucket] = {
                "open": ltp, "high": ltp, "low": ltp, "close": ltp
            }
            self.log(f"📈 Index first tick: {ltp} in bucket {bucket}")
            return

        if bucket != self.index_bucket:
            # Current bucket has changed, close the old one
            closed_bucket = self.index_bucket
            closed = dict(self.index_ohlc[closed_bucket])
            
            self.index_bucket = bucket
            self.index_ohlc[bucket] = {
                "open": ltp, "high": ltp, "low": ltp, "close": ltp
            }
            
            if closed_bucket in self.index_ohlc and closed_bucket != bucket:
                del self.index_ohlc[closed_bucket]
                
            self.pending_ohlc[closed_bucket] = closed
            self.try_flush(closed_bucket)
            return

        # Update running candle
        state = self.index_ohlc[bucket]
        state["high"] = max(state["high"], ltp)
        state["low"] = min(state["low"], ltp)
        state["close"] = ltp

    def handle_volume(self, scrip, vol, bucket):
        self.latest_seen[scrip] = vol
        
        if scrip not in self.last_bucket:
            self.last_bucket[scrip] = bucket
            self.last_volume[scrip] = vol
            return

        if bucket != self.last_bucket[scrip]:
            five_min_vol = max(0, vol - self.last_volume[scrip])
            closed_bucket = self.last_bucket[scrip]
            
            self.last_bucket[scrip] = bucket
            self.last_volume[scrip] = vol
            
            if closed_bucket not in self.pending_volume:
                self.pending_volume[closed_bucket] = {}
            self.pending_volume[closed_bucket][scrip] = five_min_vol
            
            self.try_flush(closed_bucket)

    def try_flush(self, bucket):
        """Check if index OHLC and volume are ready for a bucket."""
        if bucket not in self.pending_ohlc:
            return
            
        if self.future_token:
            # For Future volume, check if the single future token's volume tick for that bucket has been recorded
            if bucket not in self.pending_volume:
                # Retrieve last volume from cache as fallback
                vol_now = self.latest_seen.get(self.future_token, 0)
                vol_base = self.last_volume.get(self.future_token, vol_now)
                five_min_vol = max(0, vol_now - vol_base)
                self.pending_volume[bucket] = {self.future_token: five_min_vol}
            self._write_candle(bucket)
        else:
            # Real-time constituent stocks volume sum logic
            # Instead of waiting for all constituent ticks, we populate any missing stock volumes
            # from the last known values to flush the candle immediately without any delay.
            if bucket not in self.pending_volume:
                self.pending_volume[bucket] = {}
                
            now = datetime.datetime.now()
            now_bucket = self.get_5min_bucket(now)
            
            for scrip in CONSTITUENT_TOKENS:
                if scrip in self.pending_volume[bucket]:
                    continue
                vol_now = self.latest_seen.get(scrip, 0)
                vol_base = self.last_volume.get(scrip, vol_now)
                five_min_vol = max(0, vol_now - vol_base)
                
                # Store in pending
                self.pending_volume[bucket][scrip] = five_min_vol
                
                # Update baseline
                self.last_bucket[scrip] = now_bucket
                self.last_volume[scrip] = vol_now
                
            self._write_candle(bucket)

    def _write_candle(self, bucket):
        # Discard any candle starting before 09:15 AM
        if bucket.time() < datetime.time(9, 15):
            self.log(f"[CandleBuilder] Discarding pre-market candle at {bucket}")
            if bucket in self.pending_ohlc:
                del self.pending_ohlc[bucket]
            if bucket in self.pending_volume:
                del self.pending_volume[bucket]
            return

        ohlc = self.pending_ohlc[bucket]
        combined_vol = sum(self.pending_volume[bucket].values())
        
        # Format timestamp matching train dataset
        ts = bucket.strftime('%Y-%m-%d %H:%M:%S')
        
        self.log(f"[CandleBuilder] Completed 5-Min Candle: {ts} | O: {ohlc['open']:.2f} | H: {ohlc['high']:.2f} | L: {ohlc['low']:.2f} | C: {ohlc['close']:.2f} | Vol: {combined_vol:,}")
        
        # Write to local CSV
        with open(self.candle_data_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts,
                ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"],
                combined_vol
            ])
            
        # Clean up pending
        if bucket in self.pending_ohlc:
            del self.pending_ohlc[bucket]
        if bucket in self.pending_volume:
            del self.pending_volume[bucket]
            
        # Fire callback
        if self.on_candle_completed_cb:
            self.on_candle_completed_cb(ts, ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], combined_vol)

    def force_flush_stale(self, age_seconds=30):
        """Force flush any pending buckets that have been incomplete for too long."""
        now = datetime.datetime.now()
        all_buckets = sorted(
            list(set(list(self.pending_ohlc.keys()) + list(self.pending_volume.keys())))
        )
        
        for bucket in all_buckets:
            bucket_end = bucket + datetime.timedelta(minutes=5)
            if (now - bucket_end).total_seconds() < age_seconds:
                continue
                
            if self.future_token:
                # Force flush future volume
                if bucket not in self.pending_volume:
                    vol_now = self.latest_seen.get(self.future_token, 0)
                    vol_base = self.last_volume.get(self.future_token, vol_now)
                    five_min_vol = max(0, vol_now - vol_base)
                    self.pending_volume[bucket] = {self.future_token: five_min_vol}
            else:
                # Force flush constituent volumes
                if bucket in self.pending_volume:
                    now_bucket = self.get_5min_bucket(now)
                    for scrip in CONSTITUENT_TOKENS:
                        if scrip in self.pending_volume[bucket]:
                            continue
                        vol_now = self.latest_seen.get(scrip, 0)
                        vol_base = self.last_volume.get(scrip, vol_now)
                        five_min_vol = max(0, vol_now - vol_base)
                        
                        self.pending_volume[bucket][scrip] = five_min_vol
                        self.last_bucket[scrip] = now_bucket
                        self.last_volume[scrip] = vol_now
                        
            if bucket in self.pending_ohlc and bucket in self.pending_volume:
                self.log(f"[CandleBuilder] Force flushing bucket {bucket}...")
                self._write_candle(bucket)
            else:
                if bucket in self.pending_ohlc:
                    del self.pending_ohlc[bucket]
                if bucket in self.pending_volume:
                    del self.pending_volume[bucket]

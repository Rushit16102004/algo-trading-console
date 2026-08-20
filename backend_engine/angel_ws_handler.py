# PASSKEY: rushit2712
import asyncio
import datetime
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from backend_engine.websocket_handler import CONSTITUENT_TOKENS, INDEX_TOKEN

class AngelOneWSHandler:
    def __init__(self, tick_queue: asyncio.Queue, api_key, client_id, jwt_token, feed_token, trade_logger=None, future_token=None):
        self.tick_queue = tick_queue
        self.api_key = api_key
        self.client_id = client_id
        self.jwt_token = jwt_token
        self.feed_token = feed_token
        self.trade_logger = trade_logger
        self.future_token = int(future_token) if future_token is not None else None
        self.ws = None
        self.is_running = False
        self.conn_status = {"status": "offline"}
        self.active_option_tokens = set()
        self.loop = None

    def log(self, message):
        if self.trade_logger:
            self.trade_logger.log_activity(message)
        else:
            now = datetime.datetime.now()
            ts = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts} [AngelOne WS] {message}")

    async def subscribe_options(self, tokens):
        # Options subscription is completely disabled as we trade nifty spot directly
        pass

    def on_open(self, wsapp):
        self.conn_status["status"] = "live"
        self.log("[LIVE] Connected to Angel One Market Feed WebSocket.")
        
        # 1. Subscribe to Nifty 50 Spot Index (Token 99926000 on exchangeType 1)
        nifty_token_list = [
            {
                "exchangeType": 1,
                "tokens": ["99926000"]
            }
        ]
        self.ws.subscribe(correlation_id="nifty_spot_sub", mode=1, token_list=nifty_token_list)
        
        # 2. Subscribe to Volume Source
        if self.future_token:
            # Subscribe to the resolved Nifty Future contract (ExchangeType 2: NSE_FO) with mode 2 (Quote) to get volume updates
            self.log(f"Subscribing to Nifty Future for volume: token {self.future_token}")
            future_list = [
                {
                    "exchangeType": 2,
                    "tokens": [str(self.future_token)]
                }
            ]
            self.ws.subscribe(correlation_id="future_vol_sub", mode=2, token_list=future_list)
        else:
            # Fallback to subscribing to 50 constituent stocks
            self.log("No Nifty Future token provided. Falling back to subscribing to 50 constituent stocks.")
            constituent_list = [
                {
                    "exchangeType": 1,
                    "tokens": [str(t) for t in CONSTITUENT_TOKENS]
                }
            ]
            self.ws.subscribe(correlation_id="constituents_sub", mode=2, token_list=constituent_list)

    def on_data(self, wsapp, message):
        """Callback for incoming data packet."""
        if not message or not isinstance(message, dict):
            return
            
        token = message.get("token")
        if not token:
            return
            
        mapped_tick = self.map_tick(message)
        if mapped_tick and self.loop:
            try:
                self.loop.call_soon_threadsafe(self.tick_queue.put_nowait, mapped_tick)
            except Exception:
                pass

    def map_tick(self, message):
        """Maps Angel One tick structure to the internal format expected by the CandleBuilder."""
        token = message.get("token")
        scrip_code = int(token)
        
        # Map Nifty Spot back to 26000 internally
        if scrip_code == 99926000:
            scrip_code = 26000
            
        is_index = (scrip_code == 26000)
        is_future = (self.future_token is not None and scrip_code == self.future_token)
        is_constituent = scrip_code in CONSTITUENT_TOKENS or is_future
        
        ltp_val = None
        vol_val = None
        rec_type = "A" # Default to LTP tick
        
        if "last_traded_price" in message:
            ltp_val = float(message["last_traded_price"]) / 100.0
            
        if "volume_trade_for_the_day" in message:
            vol_val = int(message["volume_trade_for_the_day"])
            if is_constituent:
                rec_type = "d" # Volume tick
                
        # Guard clauses:
        if is_constituent and vol_val is None:
            return None
        if (is_index or not is_constituent) and ltp_val is None:
            return None
            
        # Parse time
        tick_time = datetime.datetime.now()
        exch_ts = message.get("exchange_timestamp")
        if exch_ts:
            try:
                if isinstance(exch_ts, (int, float)):
                    if exch_ts > 10**12:
                        tick_time = datetime.datetime.fromtimestamp(exch_ts / 1000.0)
                    else:
                        tick_time = datetime.datetime.fromtimestamp(exch_ts)
            except Exception:
                pass
                
        return {
            "ScripCode": scrip_code,
            "Time": tick_time,
            "RecType": rec_type,
            "LTP": ltp_val,
            "Volume64": vol_val
        }

    def on_error(self, wsapp, error):
        self.conn_status["status"] = "error"
        self.log(f"WebSocket Error: {error}")

    def on_close(self, wsapp, close_status_code, close_msg):
        self.conn_status["status"] = "offline"
        self.log(f"WebSocket connection closed. Code: {close_status_code}, Message: {close_msg}")

    async def connect_and_stream(self):
        """Maintain the single centralized Angel One stream with reconnects."""
        self.is_running = True
        self.loop = asyncio.get_running_loop()

        while self.is_running:
            try:
                self.conn_status["status"] = "connecting"
                self.ws = SmartWebSocketV2(self.jwt_token, self.api_key, self.client_id, self.feed_token)
                self.ws.on_open = self.on_open
                self.ws.on_data = self.on_data
                self.ws.on_error = self.on_error
                self.ws.on_close = self.on_close

                self.log("Connecting to Angel One feed stream in background executor...")
                await self.loop.run_in_executor(None, self.ws.connect)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.conn_status["status"] = "error"
                self.log(f"Angel One feed connection failed: {exc}")

            if self.is_running:
                self.conn_status["status"] = "connecting"
                await asyncio.sleep(5)

    def stop(self):
        self.is_running = False
        self.conn_status["status"] = "offline"
        if self.ws:
            self.log("Stopping Angel One feed WebSocket client...")
            try:
                self.ws.close()
            except Exception as e:
                self.log(f"Error while closing Angel One connection: {e}")

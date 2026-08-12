import time

class HealthMonitor:
    def __init__(self):
        self.last_tick_time = 0.0
        self.last_candle_time = 0.0
        self.reconnect_count = 0
        self.latency_ms = 0
        self.ws_connected = False
        
    def record_tick(self):
        self.last_tick_time = time.time()
        self.ws_connected = True
        
    def record_candle(self):
        self.last_candle_time = time.time()
        
    def record_reconnect(self):
        self.reconnect_count += 1
        
    def set_connected(self, connected: bool):
        self.ws_connected = connected
        
    def set_latency(self, latency_ms: int):
        self.latency_ms = latency_ms
        
    def is_feed_stale(self, timeout_seconds: int = 30) -> bool:
        """Returns True if WebSocket feed has not received any ticks for timeout_seconds."""
        # For DEMO mode or simulator, the feed is generated locally so it is never stale
        from backend_engine.config import DEMO_MODE
        if DEMO_MODE:
            return False
            
        if not self.ws_connected:
            return True
        if self.last_tick_time == 0.0:
            return True
        return (time.time() - self.last_tick_time) > timeout_seconds
        
    def get_status(self) -> dict:
        stale = self.is_feed_stale()
        return {
            "connected": self.ws_connected,
            "stale": stale,
            "last_tick_elapsed": round(time.time() - self.last_tick_time, 1) if self.last_tick_time > 0 else -1,
            "latency_ms": self.latency_ms,
            "reconnect_count": self.reconnect_count
        }

monitor = HealthMonitor()

import time
from datetime import datetime, timedelta
from typing import Optional

class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 90000,
        min_delay: float = 0.1
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.min_delay = min_delay
        
        self.request_timestamps = []
        self.token_usage = []
        self.last_request_time = None
    
    def wait_if_needed(self, tokens: Optional[int] = None):
        """Wait if necessary to stay within rate limits."""
        current_time = datetime.now()
        
        minute_ago = current_time - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        self.token_usage = [(ts, tokens) for ts, tokens in self.token_usage if ts > minute_ago]
        
        while len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = (self.request_timestamps[0] - minute_ago).total_seconds()
            time.sleep(max(sleep_time, self.min_delay))
            current_time = datetime.now()
            minute_ago = current_time - timedelta(minutes=1)
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        if tokens is not None:
            total_tokens = sum(t for _, t in self.token_usage)
            while total_tokens + tokens > self.tokens_per_minute:
                sleep_time = (self.token_usage[0][0] - minute_ago).total_seconds()
                time.sleep(max(sleep_time, self.min_delay))
                current_time = datetime.now()
                minute_ago = current_time - timedelta(minutes=1)
                self.token_usage = [(ts, t) for ts, t in self.token_usage if ts > minute_ago]
                total_tokens = sum(t for _, t in self.token_usage)
        
        if self.last_request_time is not None:
            time_since_last = (current_time - self.last_request_time).total_seconds()
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
        
        self.request_timestamps.append(current_time)
        if tokens is not None:
            self.token_usage.append((current_time, tokens))
        self.last_request_time = current_time
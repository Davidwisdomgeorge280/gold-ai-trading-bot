from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from .config import CONFIG
from .data import fetch_market_data
from .features import compute_indicators
from .regime import classify_regime
from .strategy import generate_trade_decision, score_signal


class LivePrototype:
    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = interval_seconds
        self.last_run = None

    def tick(self):
        df = fetch_market_data(symbol=CONFIG.symbol, period="2d", interval=CONFIG.data_interval)
        data = compute_indicators(df)
        data = classify_regime(data)
        data = score_signal(data, CONFIG.min_confidence)
        latest = data.iloc[-1]

        decision, confidence, reason = generate_trade_decision(latest, CONFIG.allowed_regimes)
        now = datetime.utcnow().isoformat(timespec="seconds")
        print(f"[{now}] signal={decision} confidence={confidence:.2f} reason={reason} regime={latest.get('regime', 'unknown')}")
        self.last_run = now

    def run_forever(self):
        while True:
            self.tick()
            time.sleep(self.interval_seconds)


if __name__ == "__main__":
    runner = LivePrototype(interval_seconds=300)
    runner.run_forever()

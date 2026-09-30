from __future__ import annotations

import time
from datetime import datetime

from .config import CONFIG
from .data import fetch_market_data
from .regime import classify_regime
from .strategy import generate_trade_decision, score_signal


class LiveRunner:
    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = interval_seconds

    def tick(self) -> None:
        df = fetch_market_data(symbol=CONFIG.symbol, period="7d", interval=CONFIG.data_interval)
        data = classify_regime(df)
        data = score_signal(data, CONFIG.min_confidence)
        latest = data.iloc[-1]
        signal, confidence, reason = generate_trade_decision(
            latest,
            CONFIG.allowed_regimes,
            min_confidence=CONFIG.min_confidence,
            threshold=CONFIG.signal_threshold,
        )
        print(f"[{datetime.utcnow().isoformat(timespec='seconds')}] signal={signal} confidence={confidence:.2f} regime={latest.get('regime')} reason={reason}")

    def run_forever(self) -> None:
        while True:
            self.tick()
            time.sleep(self.interval_seconds)


if __name__ == "__main__":
    runner = LiveRunner(interval_seconds=300)
    runner.run_forever()

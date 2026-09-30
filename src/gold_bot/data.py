from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import CONFIG, ensure_data_dir


def fetch_market_data(symbol: str = None, period: str = "1y", interval: str = None, cache_path: str | None = None) -> pd.DataFrame:
    symbol = symbol or CONFIG.symbol
    interval = interval or CONFIG.data_interval
    cache_path = cache_path or str(Path(ensure_data_dir()) / f"{symbol.replace('/', '_')}_{interval}.csv")

    csv_path = Path(cache_path)
    if csv_path.exists():
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if not data.empty:
            return data

    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No data returned for {symbol} at interval {interval}.")

    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    data = data[["open", "high", "low", "close", "volume"]].dropna()
    data.index.name = "timestamp"
    data.to_csv(csv_path)
    return data

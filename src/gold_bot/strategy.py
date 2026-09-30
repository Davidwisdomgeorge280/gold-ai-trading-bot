from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def compute_bollinger(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    rolling_mean = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std(ddof=0)
    upper = rolling_mean + k * rolling_std
    lower = rolling_mean - k * rolling_std
    return rolling_mean, upper, lower


def trend_strength(df: pd.DataFrame) -> pd.Series:
    short = compute_ema(df["close"], 8)
    long = compute_ema(df["close"], 21)
    trend = short - long
    std = df["close"].rolling(20).std(ddof=0).replace(0, np.nan)
    return trend / std


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ema_fast"] = compute_ema(out["close"], 8)
    out["ema_slow"] = compute_ema(out["close"], 21)
    out["ema_trend"] = compute_ema(out["close"], 50)
    out["rsi"] = compute_rsi(out["close"], 14)
    macd, macd_signal, macd_hist = compute_macd(out["close"], 12, 26, 9)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["atr"] = compute_atr(out, 14)
    rolling_mean, upper, lower = compute_bollinger(out)
    out["bb_mid"] = rolling_mean
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["volatility_ratio"] = (out["atr"] / out["close"].rolling(50).mean()).replace(0, np.nan)
    out["trend_strength"] = trend_strength(out)
    out["range_strength"] = (out["close"] - out["bb_mid"]).abs() / out["bb_upper"].sub(out["bb_lower"]).replace(0, np.nan)

    trend_up = (out["ema_fast"] > out["ema_slow"]) & (out["ema_slow"] > out["ema_trend"])
    trend_down = (out["ema_fast"] < out["ema_slow"]) & (out["ema_slow"] < out["ema_trend"])
    atr_threshold = out["atr"].rolling(50).quantile(0.8)
    range_threshold = out["range_strength"].rolling(50).quantile(0.35)
    low_vol_threshold = out["atr"].rolling(50).quantile(0.35)

    breakout_long = (out["close"] > out["bb_upper"]) & (out["atr"] > atr_threshold)
    breakout_short = (out["close"] < out["bb_lower"]) & (out["atr"] > atr_threshold)
    range_condition = (out["range_strength"] < range_threshold) & (out["rsi"].between(45, 55))
    low_vol_condition = (out["atr"] < low_vol_threshold) & (out["volatility_ratio"] < out["volatility_ratio"].rolling(50).quantile(0.5))

    out["regime"] = "neutral"
    out.loc[trend_up | trend_down, "regime"] = "trend"
    out.loc[breakout_long | breakout_short, "regime"] = "breakout"
    out.loc[range_condition, "regime"] = "range"
    out.loc[low_vol_condition, "regime"] = "low_vol"

    out["regime_confidence"] = 0.55
    out.loc[out["regime"] == "trend", "regime_confidence"] = np.clip(np.abs(out["trend_strength"]) * 2.5, 0.55, 0.95)
    out.loc[out["regime"] == "breakout", "regime_confidence"] = 0.8
    out.loc[out["regime"] == "range", "regime_confidence"] = 0.7
    out.loc[out["regime"] == "low_vol", "regime_confidence"] = 0.6

    return out

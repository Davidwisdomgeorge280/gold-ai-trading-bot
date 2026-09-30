from __future__ import annotations

import numpy as np
import pandas as pd


def score_signal(df: pd.DataFrame, min_confidence: float = 0.55) -> pd.DataFrame:
    out = df.copy()

    trend_signal = np.where(
        out["ema_fast"] > out["ema_slow"], 1.0, -1.0
    )
    momentum_signal = np.where(out["rsi"] > 60, 1.0, np.where(out["rsi"] < 40, -1.0, 0.0))
    macd_signal = np.where(out["macd"] > out["macd_signal"], 1.0, -1.0)
    breakout_signal = np.where(out["close"] > out["bb_upper"], 1.0, np.where(out["close"] < out["bb_lower"], -1.0, 0.0))

    weighted = (
        0.35 * trend_signal +
        0.25 * momentum_signal +
        0.20 * macd_signal +
        0.20 * breakout_signal
    )

    out["signal_strength"] = weighted
    out["signal_confidence"] = np.clip(np.abs(weighted) / 1.0, 0.0, 1.0)
    out["signal"] = np.where(out["signal_strength"] > 0.45, "buy",
                            np.where(out["signal_strength"] < -0.45, "sell", "hold"))

    out["signal_confidence"] = out["signal_confidence"].fillna(0.0)
    out["valid_signal"] = out["signal_confidence"] >= min_confidence
    return out


def generate_trade_decision(row: pd.Series, allowed_regimes: tuple[str, ...]) -> tuple[str, float, str]:
    regime = row.get("regime", "neutral")
    signal = row.get("signal", "hold")
    confidence = float(row.get("signal_confidence", 0.0))

    if regime not in allowed_regimes:
        return "hold", 0.0, "regime not allowed"

    if signal == "buy" and confidence >= 0.55:
        return "buy", confidence, "strong buy signal"
    if signal == "sell" and confidence >= 0.55:
        return "sell", confidence, "strong sell signal"
    return "hold", 0.0, "confidence below threshold"

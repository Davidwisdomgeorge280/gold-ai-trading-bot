from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    trend_score = np.where(
        (out["ema_fast"] > out["ema_slow"]) & (out["ema_slow"] > out["ema_trend"]),
        1.0,
        np.where(
            (out["ema_fast"] < out["ema_slow"]) & (out["ema_slow"] < out["ema_trend"]),
            -1.0,
            0.0,
        ),
    )

    volatility_score = np.where(out["volatility_ratio"] > out["volatility_ratio"].rolling(50).quantile(0.8), 1.0, 0.0)
    range_score = np.where(
        out["range_strength"] < out["range_strength"].rolling(50).quantile(0.4),
        1.0,
        0.0,
    )

    out["trend_score"] = trend_score
    out["volatility_score"] = volatility_score
    out["range_score"] = range_score

    conds = []
    choices = []

    conds.append((trend_score == 1.0) & (out["rsi"] > 55) & (out["macd_hist"] > 0))
    choices.append("trend")

    conds.append((trend_score == -1.0) & (out["rsi"] < 45) & (out["macd_hist"] < 0))
    choices.append("trend")

    conds.append((volatility_score == 1.0) & (out["atr"] > out["atr"].rolling(50).median()))
    choices.append("breakout")

    conds.append((range_score == 1.0) & (out["rsi"].between(45, 55)))
    choices.append("range")

    conds.append((volatility_score == 0.0) & (out["atr"] < out["atr"].rolling(50).quantile(0.35)))
    choices.append("low_vol")

    out["regime"] = "neutral"
    for cond, choice in zip(conds, choices):
        out.loc[cond, "regime"] = choice

    if out["regime"].eq("neutral").all():
        out["regime"] = "neutral"

    out["regime_confidence"] = out["regime"].map({
        "trend": 0.7,
        "breakout": 0.8,
        "range": 0.65,
        "low_vol": 0.55,
        "neutral": 0.4,
    })

    return out

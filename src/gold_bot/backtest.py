from __future__ import annotations

import numpy as np
import pandas as pd


def indicator_block(row: pd.Series) -> dict[str, float]:
    return {
        "trend": 1.0 if row["ema_fast"] > row["ema_slow"] else -1.0,
        "momentum": 1.0 if row["rsi"] > 60 else (-1.0 if row["rsi"] < 40 else 0.0),
        "macd": 1.0 if row["macd"] > row["macd_signal"] else -1.0,
        "breakout": 1.0 if row["close"] > row["bb_upper"] else (-1.0 if row["close"] < row["bb_lower"] else 0.0),
        "structure": 1.0 if row["close"] > row["ema_fast"] else -1.0,
    }


def build_ensemble_score(row: pd.Series) -> tuple[float, float, str, str]:
    block = indicator_block(row)
    weights = {
        "trend": 0.30,
        "momentum": 0.25,
        "macd": 0.20,
        "breakout": 0.15,
        "structure": 0.10,
    }

    raw_score = (
        weights["trend"] * block["trend"]
        + weights["momentum"] * block["momentum"]
        + weights["macd"] * block["macd"]
        + weights["breakout"] * block["breakout"]
        + weights["structure"] * block["structure"]
    )

    ensemble = float(np.clip(raw_score, -1.0, 1.0))
    if ensemble > 0.45:
        return ensemble, abs(ensemble), "buy", "multi-model bullish agreement"
    if ensemble < -0.45:
        return ensemble, abs(ensemble), "sell", "multi-model bearish agreement"
    return ensemble, abs(ensemble), "hold", "insufficient model agreement"


def score_signal(df: pd.DataFrame, min_confidence: float = 0.55) -> pd.DataFrame:
    out = df.copy()
    out["signal_strength"] = 0.0
    out["signal_confidence"] = 0.0
    out["signal"] = "hold"
    out["signal_reason"] = "no signal"

    for idx, row in out.iterrows():
        ensemble, confidence, direction, reason = build_ensemble_score(row)
        out.at[idx, "signal_strength"] = ensemble
        out.at[idx, "signal_confidence"] = confidence
        out.at[idx, "signal"] = direction
        out.at[idx, "signal_reason"] = reason

    out["valid_signal"] = out["signal_confidence"] >= min_confidence
    return out


def generate_trade_decision(row: pd.Series, allowed_regimes: tuple[str, ...], min_confidence: float = 0.55, threshold: float = 0.42) -> tuple[str, float, str]:
    regime = row.get("regime", "neutral")
    signal = str(row.get("signal", "hold")).lower()
    confidence = float(row.get("signal_confidence", 0.0))
    strength = float(row.get("signal_strength", 0.0))

    if regime not in allowed_regimes:
        return "hold", 0.0, "regime not allowed"

    if signal == "buy" and confidence >= min_confidence and strength > threshold:
        return "buy", confidence, "bullish ensemble with strong agreement"
    if signal == "sell" and confidence >= min_confidence and strength < -threshold:
        return "sell", confidence, "bearish ensemble with strong agreement"

    return "hold", 0.0, "confidence or agreement below threshold"


def evaluate_trades(trade_df: pd.DataFrame) -> dict[str, float]:
    if trade_df.empty:
        return {
            "total_trades": 0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    pnl = trade_df["pnl"].astype(float)
    wins = pnl > 0
    losses = pnl < 0
    gross_profit = pnl[wins].sum() if wins.any() else 0.0
    gross_loss = abs(pnl[losses].sum()) if losses.any() else 0.0

    metrics = {
        "total_trades": int(len(trade_df)),
        "net_pnl": float(pnl.sum()),
        "win_rate": float(wins.mean()) if len(trade_df) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_trade": float(pnl.mean()),
        "max_drawdown": float(abs(pnl[pnl < 0].min())) if (pnl < 0).any() else 0.0,
    }

    returns = pnl / max(abs(pnl).sum(), 1e-8)
    if len(returns) > 1:
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)
        metrics["sharpe"] = float(mean_ret / std_ret) if std_ret > 0 else 0.0
    else:
        metrics["sharpe"] = 0.0

    negative_returns = returns[returns < 0]
    metrics["sortino"] = float(negative_returns.mean() / negative_returns.std(ddof=1)) if len(negative_returns) > 1 and negative_returns.std(ddof=1) > 0 else 0.0

    return metrics

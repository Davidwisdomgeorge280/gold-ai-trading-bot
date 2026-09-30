from __future__ import annotations

import math


def calc_stop_loss(entry: float, atr_value: float, side: str, atr_mult: float = 1.2) -> float:
    if side == "long":
        return entry - atr_value * atr_mult
    if side == "short":
        return entry + atr_value * atr_mult
    raise ValueError(f"Unsupported side: {side}")


def calc_take_profit(entry: float, atr_value: float, side: str, atr_mult: float = 2.0) -> float:
    if side == "long":
        return entry + atr_value * atr_mult
    if side == "short":
        return entry - atr_value * atr_mult
    raise ValueError(f"Unsupported side: {side}")


def risk_per_trade(account_balance: float, risk_pct: float, entry: float, stop: float, symbol_price: float | None = None) -> float:
    risk_amount = account_balance * risk_pct
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0

    units = risk_amount / risk_per_unit
    if symbol_price is not None:
        units = min(units, account_balance * 0.25 / symbol_price)
    return units


def position_size(account_balance: float, risk_pct: float, entry: float, stop: float, leverage_cap: float, symbol_price: float | None = None) -> float:
    raw_units = risk_per_trade(account_balance, risk_pct, entry, stop, symbol_price)
    max_units = (account_balance * leverage_cap / max(entry, 1e-6)) if symbol_price is None else (account_balance * leverage_cap / symbol_price)
    return min(raw_units, max_units)


def max_trades_today(trades_today: int, max_trades: int) -> bool:
    return trades_today >= max_trades

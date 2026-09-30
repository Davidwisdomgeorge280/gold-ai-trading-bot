from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .data import fetch_market_data
from .features import compute_indicators
from .regime import classify_regime
from .risk import calc_stop_loss, calc_take_profit, position_size
from .strategy import generate_trade_decision, score_signal


class PaperTrader:
    def __init__(self, df: pd.DataFrame, config=None):
        self.config = config or CONFIG
        self.df = df.copy().reset_index().rename(columns={"index": "timestamp"})
        self.trades = []
        self.daily_trade_count = 0
        self.current_day = None

    def _reset_daily_trade_count_if_needed(self, date_str: str):
        if self.current_day != date_str:
            self.current_day = date_str
            self.daily_trade_count = 0

    def run(self) -> pd.DataFrame:
        data = self.df.copy()
        data = compute_indicators(data)
        data = classify_regime(data)
        data = score_signal(data, self.config.min_confidence)

        equity = self.config.account_balance
        peak_equity = equity
        max_drawdown = 0.0
        trade_log = []
        position = None
        realized_pnl = 0.0

        for idx, row in data.iterrows():
            current_date = str(pd.to_datetime(row["timestamp"]).date())
            self._reset_daily_trade_count_if_needed(current_date)

            if position is not None:
                entry = position["entry"]
                side = position["side"]
                stop = position["stop"]
                tp = position["take_profit"]
                units = position["units"]

                last_close = float(row["close"])
                if side == "buy":
                    if last_close <= stop or last_close >= tp:
                        pnl = (last_close - entry) * units
                        realized_pnl += pnl
                        trade_log.append({
                            "entry": entry,
                            "exit": last_close,
                            "side": side,
                            "pnl": pnl,
                            "date": current_date,
                        })
                        position = None
                else:
                    if last_close >= stop or last_close <= tp:
                        pnl = (entry - last_close) * units
                        realized_pnl += pnl
                        trade_log.append({
                            "entry": entry,
                            "exit": last_close,
                            "side": side,
                            "pnl": pnl,
                            "date": current_date,
                        })
                        position = None

            if position is None and self.daily_trade_count < self.config.max_trades_per_day:
                signal, conf, reason = generate_trade_decision(row, self.config.allowed_regimes)
                if signal in {"buy", "sell"}:
                    side = signal
                    atr_value = float(row.get("atr", 0.0))
                    if atr_value <= 0:
                        continue

                    entry = float(row["close"])
                    stop = calc_stop_loss(entry, atr_value, side, self.config.stop_loss_atr_mult)
                    take_profit = calc_take_profit(entry, atr_value, side, self.config.take_profit_atr_mult)
                    units = position_size(
                        account_balance=equity,
                        risk_pct=self.config.risk_per_trade,
                        entry=entry,
                        stop=stop,
                        leverage_cap=self.config.leverage_cap,
                        symbol_price=entry,
                    )
                    if units <= 0:
                        continue

                    position = {
                        "entry": entry,
                        "side": side,
                        "stop": stop,
                        "take_profit": take_profit,
                        "units": units,
                        "date": current_date,
                    }
                    self.daily_trade_count += 1

            equity = self.config.account_balance + realized_pnl
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        trade_df = pd.DataFrame(trade_log)
        if trade_df.empty:
            trade_df = pd.DataFrame(columns=["entry", "exit", "side", "pnl", "date"])

        return trade_df


def run_backtest() -> pd.DataFrame:
    df = fetch_market_data(symbol=CONFIG.symbol, period="1y", interval=CONFIG.data_interval)
    trader = PaperTrader(df, CONFIG)
    results = trader.run()
    return results


if __name__ == "__main__":
    result = run_backtest()
    print(result.head(20))
    if not result.empty:
        print(f"Trades: {len(result)}")
        print(f"Net PnL: {result['pnl'].sum():.2f}")
        print(f"Win rate: {((result['pnl'] > 0).mean() * 100):.2f}%")

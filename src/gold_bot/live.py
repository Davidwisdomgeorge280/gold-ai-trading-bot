from __future__ import annotations

from typing import Any

import pandas as pd

from .config import CONFIG
from .data import fetch_market_data
from .regime import classify_regime
from .risk import calc_stop_loss, calc_take_profit, position_size
from .strategy import evaluate_trades, generate_trade_decision, score_signal


class GoldBacktester:
    def __init__(self, df: pd.DataFrame, config: Any = None):
        self.config = config or CONFIG
        self.data = df.copy().reset_index().rename(columns={"index": "timestamp"})
        self.trades = []
        self.daily_trade_count = 0
        self.current_day = None
        self.daily_loss = 0.0
        self.equity = self.config.account_balance
        self.peak_equity = self.config.account_balance
        self.max_drawdown = 0.0

    def _next_day(self, dt: pd.Timestamp) -> None:
        date_key = str(dt.date())
        if self.current_day != date_key:
            self.current_day = date_key
            self.daily_trade_count = 0
            self.daily_loss = 0.0

    def _apply_trade_exit(self, position: dict[str, Any], row: pd.Series) -> None:
        last_close = float(row["close"])
        side = position["side"]
        entry = position["entry"]
        units = position["units"]

        if side == "buy":
            if last_close <= position["stop"] or last_close >= position["take_profit"]:
                pnl = (last_close - entry) * units
                self.trades.append({
                    "entry": entry,
                    "exit": last_close,
                    "side": side,
                    "pnl": pnl,
                    "date": str(pd.Timestamp(row["timestamp"]).date()),
                })
                self.equity += pnl
                self.daily_loss += pnl
                self.peak_equity = max(self.peak_equity, self.equity)
                self.max_drawdown = max(self.max_drawdown, (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0)
                return
        else:
            if last_close >= position["stop"] or last_close <= position["take_profit"]:
                pnl = (entry - last_close) * units
                self.trades.append({
                    "entry": entry,
                    "exit": last_close,
                    "side": side,
                    "pnl": pnl,
                    "date": str(pd.Timestamp(row["timestamp"]).date()),
                })
                self.equity += pnl
                self.daily_loss += pnl
                self.peak_equity = max(self.peak_equity, self.equity)
                self.max_drawdown = max(self.max_drawdown, (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0)
                return

    def run(self) -> tuple[pd.DataFrame, dict[str, float]]:
        processed = self.data.copy()
        processed = classify_regime(processed)
        processed = score_signal(processed, self.config.min_confidence)

        position = None
        for _, row in processed.iterrows():
            self._next_day(pd.Timestamp(row["timestamp"]))

            if position is not None:
                self._apply_trade_exit(position, row)
                position = None

            if self.daily_loss <= -self.config.account_balance * self.config.max_daily_loss_pct:
                continue

            if self.peak_equity > 0 and (self.peak_equity - self.equity) / self.peak_equity > self.config.max_drawdown_pct:
                break

            if self.daily_trade_count >= self.config.max_trades_per_day:
                continue

            signal, conf, reason = generate_trade_decision(
                row,
                self.config.allowed_regimes,
                min_confidence=self.config.min_confidence,
                threshold=self.config.signal_threshold,
            )

            if signal not in {"buy", "sell"}:
                continue

            atr_value = float(row.get("atr", 0.0))
            if atr_value <= 0:
                continue

            entry = float(row["close"])
            side = signal
            stop = calc_stop_loss(entry, atr_value, side, self.config.stop_loss_atr_mult)
            take_profit = calc_take_profit(entry, atr_value, side, self.config.take_profit_atr_mult)
            units = position_size(
                account_balance=self.equity,
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
            }
            self.daily_trade_count += 1

        trade_df = pd.DataFrame(self.trades)
        metrics = evaluate_trades(trade_df)
        return trade_df, metrics


def run_backtest() -> tuple[pd.DataFrame, dict[str, float]]:
    df = fetch_market_data(symbol=CONFIG.symbol, period="1y", interval=CONFIG.data_interval)
    backtester = GoldBacktester(df, CONFIG)
    return backtester.run()


if __name__ == "__main__":
    trades, metrics = run_backtest()
    print(trades.head(10))
    print(metrics)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BotConfig:
    symbol: str = "XAUUSD=X"
    data_interval: str = "1h"
    max_trades_per_day: int = 3
    risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 0.04
    max_drawdown_pct: float = 0.12
    account_balance: float = 10000.0
    leverage_cap: float = 5.0
    min_confidence: float = 0.55
    take_profit_atr_mult: float = 2.2
    stop_loss_atr_mult: float = 1.3
    warmup_period: int = 100
    fee_pct: float = 0.0002
    allowed_regimes: tuple[str, ...] = ("trend", "breakout", "range")
    data_dir: str = "./data"
    risk_mode: str = "balanced"
    signal_threshold: float = 0.42
    session_monitor: bool = True
    mobile_dashboard: bool = True


CONFIG = BotConfig()


def ensure_data_dir() -> Path:
    path = Path(CONFIG.data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

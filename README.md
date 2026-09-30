# Gold AI Trading Bot

A modular, research-first gold (XAU/USD) trading intelligence project built as a serious foundation for systematic gold trading.

This repository is intentionally designed to be:
- gold-only
- multi-timeframe
- regime-aware
- risk-managed
- backtest-friendly
- mobile-friendly for monitoring
- deployable to a low-cost or freemium hosting platform for long-running monitoring and execution

Important note:
- This is not a guarantee of profit.
- No trading system can honestly promise 100% win rate or guaranteed returns.
- The project is structured to prioritize risk control, evidence, and repeatability over hype.

## What this project includes

- Data ingestion for XAU/USD market data
- Feature engineering across multiple timeframes
- Regime classification (trend, range, breakout, high-volatility, low-volatility)
- AI-style ensemble signal scoring
- Risk engine with daily trade cap, max drawdown protection, and positional sizing
- Backtesting harness
- Optional live execution skeleton
- Optional mobile-friendly dashboard via Streamlit

## Recommended quick start

1. Clone the repo.
2. Create a virtual environment.
3. Install dependencies.
4. Run the backtest.
5. Review metrics.
6. If desired, use the dashboard or live runner.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the backtest

```bash
python -m src.gold_bot.backtest
```

## Run the dashboard

```bash
streamlit run src/gold_bot/dashboard.py
```

## Run the live runner (prototype)

```bash
python -m src.gold_bot.live
```

## Project structure

```text
src/
  gold_bot/
    __init__.py
    config.py
    data.py
    features.py
    regime.py
    risk.py
    strategy.py
    backtest.py
    live.py
    dashboard.py
```

## Scope and assumptions

This project is intentionally scoped to gold only, with a conservative architecture:
- asset: XAU/USD
- max daily trade cap: 3 trades/day, configurable
- risk-managed execution by default
- regime-aware decisions only
- adaptive context, not a fixed indicator-only strategy

## Hosting / deployment guidance

A true 24/7 unattended trading bot usually needs a server, not just a phone. A phone can be used for monitoring or as a lightweight control panel, but for reliable long-running execution a cloud VPS or low-cost hosting service is typically needed.

Examples of reasonable hosting choices:
- Render
- Railway
- DigitalOcean droplet
- Hetzner
- Fly.io
- Google Cloud Run (for some patterns)

The live trading module in this repo is intentionally a prototype and should be tested thoroughly in paper mode before any real-money deployment.

## Disclaimer

The software in this repository is for research, education, and controlled testing. It is not a promise of profitability or financial advice.

Use it responsibly, with proper testing, risk controls, and broker/instrument validation.

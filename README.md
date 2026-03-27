# Forex AI Trader

Real-time XAUUSD (Gold) signal dashboard powered by XGBoost machine learning model with MetaTrader 5 integration.

![Dashboard](https://img.shields.io/badge/status-live-brightgreen) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **AI Signal Scoring** — XGBoost model trained on 50+ technical features predicts BUY / SELL / NO_TRADE with confidence scores
- **Live MT5 Data** — Real-time price, account balance, and M30 candle data from MetaTrader 5
- **TradingView Chart** — Embedded interactive XAUUSD chart
- **Signal History** — All signals auto-saved to SQLite database with full indicator snapshots
- **Export** — Download signal history as CSV or JSON
- **Technical Indicators** — RSI, Stochastic, ATR, ADX, Bollinger Bands, EMA Trend, MACD
- **Session Stats** — Daily signal counts and distribution

## Screenshot

Dark Bloomberg-terminal themed interface with live gold price, confidence gauge, signal reasons, and embedded chart.

## Setup

### Prerequisites
- Python 3.10+
- MetaTrader 5 terminal (with XAUUSD.m symbol)
- Windows (MT5 requirement)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/forex-ai-trader.git
cd forex-ai-trader
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Model Files (not included)
Place these in a `models/` directory:
- `gold_signal_model.json` — Trained XGBoost model
- `gold_signal_config.json` — Model configuration and feature definitions

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/signal` | Get current signal with all indicators |
| `GET /api/signals?limit=50&signal=BUY&from=2026-03-27` | Query signal database with filters |
| `GET /api/signals/count` | Signal statistics |
| `GET /api/history` | Recent 20 signals |
| `GET /api/export` | Download CSV |
| `GET /api/export/json` | Download JSON |
| `GET /api/status` | Health check |

## Tech Stack

- **Backend**: Flask + SQLite
- **ML Model**: XGBoost (3-class: BUY/SELL/NO_TRADE)
- **Data Source**: MetaTrader 5 Python API
- **Frontend**: Vanilla JS + CSS (dark terminal theme)
- **Chart**: TradingView widget

## License

MIT

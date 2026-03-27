# Forex AI Trader

Real-time XAUUSD (Gold) signal dashboard powered by XGBoost machine learning model with MetaTrader 5 integration.

![Dashboard](https://img.shields.io/badge/status-live-brightgreen) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **AI Signal Scoring** -- XGBoost model trained on 50+ technical features predicts BUY / SELL / NO_TRADE
- **Live MT5 Data** -- Real-time price, account balance, and M30 candle data from MetaTrader 5
- **WebSocket Updates** -- Real-time push via Flask-SocketIO (polling fallback)
- **Multi-Pair Support** -- XAUUSD, EURUSD, GBPUSD, USDJPY, BTCUSD, ETHUSD
- **Signal Database** -- All signals auto-saved to SQLite with full indicator snapshots
- **Signal History Browser** -- Filter, sort, and paginate past signals at `/history`
- **Backtest Simulator** -- Run the model on historical data at `/simulate`
- **Model Management** -- Retrain, compare, and promote models at `/admin/models`
- **Telegram Alerts** -- High-confidence signals sent to Telegram
- **Authentication** -- Login page + API key for programmatic access
- **TradingView Chart** -- Embedded interactive chart
- **Export** -- Download signal history as CSV or JSON

## Setup

### Prerequisites
- Python 3.10+
- MetaTrader 5 terminal (with XAUUSD.m symbol)
- Windows (MT5 requirement)

### Install

```bash
git clone https://github.com/superweird7/forex-ai-trader.git
cd forex-ai-trader
pip install -r requirements.txt
```

### Configure

1. Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

2. Edit `.env`:
```
DASHBOARD_API_KEY=your-random-api-key
DASHBOARD_USER=admin
DASHBOARD_PASS=your-password
TELEGRAM_BOT_TOKEN=your-bot-token  # optional
TELEGRAM_CHAT_ID=your-chat-id      # optional
FLASK_SECRET_KEY=random-secret-string
```

3. Edit `config.yaml` to set your paths:
```yaml
paths:
  python_dir: "D:/FOREX/python"
  model_path: "D:/FOREX/models/gold_signal_model.json"
  # ...
```

### Model Files
Place these in your models directory:
- `gold_signal_model.json` -- Trained XGBoost model
- `gold_signal_config.json` -- Model configuration

### Run

```bash
python app.py
```

Open http://localhost:5000 in your browser. Login with credentials from `.env`.

### Auto-Start on Boot (Windows)

A VBS launcher is included. Copy to Startup folder:
```bash
cp start_dashboard.vbs "%APPDATA%/Microsoft/Windows/Start Menu/Programs/Startup/"
```

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Live signal with confidence gauge, indicators, chart |
| Signal History | `/history` | Browse, filter, sort all past signals |
| Simulator | `/simulate` | Backtest the model on historical data |
| Model Admin | `/admin/models` | View stats, retrain, promote models |
| Login | `/login` | Authentication page |

## API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/signal?symbol=XAUUSD.m` | Session/Key | Current signal with all indicators |
| `GET /api/signals?limit=50&signal=BUY&from=...&symbol=...` | Session/Key | Query signal database |
| `GET /api/signals/count` | Session/Key | Signal statistics |
| `GET /api/symbols` | Session/Key | List enabled symbols |
| `GET /api/history` | Session/Key | Recent 20 signals |
| `GET /api/export` | Session/Key | Download CSV |
| `GET /api/export/json` | Session/Key | Download JSON |
| `GET /api/status` | Session/Key | Health check |
| `POST /api/simulate` | Session/Key | Run backtest simulation |
| `GET /api/simulations` | Session/Key | List past simulations |
| `GET /api/admin/model-stats` | Session/Key | Current model info |
| `GET /api/admin/models/list` | Session/Key | List saved models |
| `POST /api/admin/retrain` | Session/Key | Retrain model |
| `POST /api/admin/models/promote` | Session/Key | Switch active model |

**API Key auth:** Pass `X-API-Key: your-key` header.

## Telegram Alerts

Set `alerts.enabled: true` in `config.yaml` and fill in your Telegram bot token and chat ID in `.env`. Only signals above the confidence threshold (default 90%) trigger alerts.

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Backend**: Flask + Flask-SocketIO + SQLite
- **ML Model**: XGBoost (3-class: BUY/SELL/NO_TRADE)
- **Data Source**: MetaTrader 5 Python API
- **Frontend**: Tailwind CSS + Chart.js
- **Real-time**: WebSocket via Socket.IO
- **Chart**: TradingView widget

## License

MIT

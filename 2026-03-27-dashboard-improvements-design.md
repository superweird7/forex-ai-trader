# Dashboard Improvements Design

**Date:** 2026-03-27
**Scope:** 10 features for the XAUUSD Signal Dashboard
**Approach:** Incremental upgrade (Approach A) — add features to existing Flask monolith
**Target audience:** Personal use only (single user)

## 1. Config & Hardcoded Paths

Replace 4 hardcoded paths in `app.py` with config files.

**`.env`** — secrets only:
- `DASHBOARD_API_KEY` — for programmatic API access
- `DASHBOARD_USER` / `DASHBOARD_PASS` — login credentials
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — alert delivery

**`config.yaml`** — app settings:
- `paths` — python_dir, model_path, config_path, mt5_terminal
- `scoring` — symbol, timeframe, bars_count, confidence_threshold
- `alerts` — enabled, min_confidence
- `server` — host, port
- `symbols` — list of enabled trading pairs

New deps: `python-dotenv`, `pyyaml`

## 2. Authentication

**Login page** (`/login`):
- Username/password form, dark themed
- Credentials from `.env`
- Flask session cookie, stays logged in until browser close
- All routes except `/login` redirect if no session

**API key** (programmatic):
- `X-API-Key` header checked against `.env`
- For Telegram bot, scripts, external tools
- No session AND no valid key = 401

No user database, no registration. Single-user from `.env`.

## 3. Telegram Alerts

- On new signal save (once per M30 bar), check confidence >= threshold from `config.yaml`
- If passes, POST to `api.telegram.org` with formatted message
- Message format: signal direction, price, confidence, key indicators, timestamp
- Toggle: `alerts.enabled` in `config.yaml`
- No new deps (uses `requests` already available via Flask)

## 4. WebSocket Real-Time Updates

Replace frontend `setInterval` polling with Flask-SocketIO push.

- Server emits `signal_update` when new M30 bar detected
- Background thread checks MT5 every 10 seconds, emits only on data change
- Frontend: `socket.on('signal_update', ...)` replaces `fetch` polling
- Auto-fallback to HTTP polling if WebSocket fails

New dep: `flask-socketio`

## 5. Simulation / Historical Mode

**A. Signal Browser** (`/history` page):
- Table of all past signals from SQLite
- Filter: date range, signal type, min confidence
- Sort by any column, pagination
- Works without MT5

**B. Backtest Scorer** (`/simulate` page):
- Pick symbol + date range
- Load historical M30 from parquet files (`D:/FOREX/data/`)
- Run XGBoost bar-by-bar, generate signals
- Show: total signals, distribution, profitability analysis
- Save results to `simulations` table in SQLite

## 6. Multi-Pair Support

- Symbol list from `config.yaml`
- Dashboard dropdown to switch active symbol
- MT5 data fetched per selected symbol
- New `symbol` column in `signals` table
- History/simulation pages filter by symbol
- Warning badge on non-XAUUSD symbols: "Model not trained for this pair"
- Per-symbol models deferred to future (use same model for now)

## 7. Unit Tests

Test targets:
- DB operations (insert, query, stats, dedup)
- Auth (login redirect, API key, 401 on unauthorized)
- Config loading (.env + config.yaml parsing)
- Telegram alerts (mock requests.post, verify format/threshold)
- Signal endpoint (mock MT5, verify payload structure)

Setup: `tests/` folder, `pytest`, `conftest.py` with temp DB + test client. No MT5 needed.

New dep: `pytest`

## 8. Model Management Panel (`/admin/models`)

**Tab A — Current Model:**
- Stats: feature count, training date, accuracy, winrates
- Feature importance bar chart (top 15)
- Live accuracy: compare signal vs actual price N bars later, `signal_outcomes` table

**Tab B — Retrain:**
- Button to retrain XGBoost on latest parquet data
- Date range + train/val split picker
- Progress bar, background thread
- New model saved as `gold_signal_model_YYYYMMDD.json` (never overwrites)
- Side-by-side comparison: old vs new metrics

**Tab C — A/B Testing:**
- Toggle to run both models per bar
- Store predictions with `model_version` tag
- After N bars (default 100), show accuracy comparison
- One-click "promote" to swap active model

## 9. Add `gold_signal_scorer.py` to Repo

Copy `python/gold_signal_scorer.py` into the dashboard repo so it's self-contained. Update import paths to reference local copy.

## 10. Build Order

Dependencies flow top-down:

| Phase | Feature | Depends On |
|-------|---------|-----------|
| 1 | Config (.env + config.yaml) | nothing |
| 2 | Auth (login + API key) | config |
| 3 | Scorer to repo | nothing |
| 4 | Telegram alerts | config |
| 5 | Multi-pair + symbol DB column | config |
| 6 | WebSocket | config |
| 7 | Signal browser page | DB, auth |
| 8 | Backtest simulator page | scorer, DB |
| 9 | Model management panel | scorer, DB, config |
| 10 | Unit tests | alongside each feature |

## New Dependencies

- `python-dotenv`
- `pyyaml`
- `flask-socketio`
- `pytest` (dev only)

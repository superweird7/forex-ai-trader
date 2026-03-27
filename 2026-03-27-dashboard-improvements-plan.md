# Dashboard Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 10 features to the Gold Signal Dashboard: config files, auth, Telegram alerts, WebSocket, signal browser, backtest simulator, multi-pair, unit tests, model management, and scorer bundling.

**Architecture:** Incremental upgrade to existing Flask monolith (`app.py`). Each feature adds code to `app.py` and new templates/static files. No restructuring.

**Tech Stack:** Flask, SQLite, Flask-SocketIO, XGBoost, python-dotenv, PyYAML, pytest, requests (Telegram API)

---

### Task 1: Install New Dependencies

**Files:**
- Modify: `dashboard/requirements.txt`

**Step 1: Update requirements.txt**

Add these lines to `dashboard/requirements.txt`:
```
flask
pandas
numpy
xgboost
MetaTrader5
python-dotenv
pyyaml
flask-socketio
pytest
requests
```

**Step 2: Install**

Run: `pip install python-dotenv pyyaml flask-socketio pytest requests`

**Step 3: Commit**

```bash
cd D:/FOREX/dashboard
git add requirements.txt
git commit -m "chore: add new dependencies for dashboard improvements"
```

---

### Task 2: Config Files (.env + config.yaml)

**Files:**
- Create: `dashboard/.env`
- Create: `dashboard/config.yaml`
- Modify: `dashboard/app.py` (lines 17, 45, 200-203, 222, 236, 242, 339)
- Modify: `dashboard/.gitignore`

**Step 1: Create `.env`**

Create `dashboard/.env`:
```
DASHBOARD_API_KEY=change-me-to-a-random-key
DASHBOARD_USER=admin
DASHBOARD_PASS=change-me
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
FLASK_SECRET_KEY=change-me-to-random-string
```

**Step 2: Create `config.yaml`**

Create `dashboard/config.yaml`:
```yaml
paths:
  python_dir: "D:/FOREX/python"
  model_path: "D:/FOREX/models/gold_signal_model.json"
  config_path: "D:/FOREX/models/gold_signal_config.json"
  mt5_terminal: "C:/Program Files/MetaTrader 5/terminal64.exe"
  data_dir: "D:/FOREX/data"

scoring:
  default_symbol: "XAUUSD.m"
  timeframe: "M30"
  bars_count: 200
  confidence_threshold: 90

alerts:
  enabled: false
  min_confidence: 90

server:
  host: "0.0.0.0"
  port: 5000

symbols:
  - "XAUUSD.m"
  - "EURUSD.m"
  - "GBPUSD.m"
  - "USDJPY.m"
  - "BTCUSD.m"
  - "ETHUSD.m"
```

**Step 3: Update `.gitignore`**

Add `.env` to `dashboard/.gitignore` (already has it). Add `*.db` if not present. Add `.env.example` pattern note.

**Step 4: Create `.env.example`**

Create `dashboard/.env.example` (committed to repo, shows structure without secrets):
```
DASHBOARD_API_KEY=your-api-key-here
DASHBOARD_USER=admin
DASHBOARD_PASS=your-password-here
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
FLASK_SECRET_KEY=your-secret-key-here
```

**Step 5: Add config loading to `app.py`**

At the top of `app.py`, after imports, add config loading:

```python
import yaml
from dotenv import load_dotenv

# Load config
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Add python path from config
sys.path.insert(0, CONFIG["paths"]["python_dir"])
```

**Step 6: Replace all hardcoded paths**

Replace in `app.py`:
- Line 17: `sys.path.insert(0, "D:/FOREX/python")` → use `CONFIG["paths"]["python_dir"]`
- Line 200-201: model_path/config_path → `CONFIG["paths"]["model_path"]`, `CONFIG["paths"]["config_path"]`
- Line 222, 236: MT5 terminal path → `CONFIG["paths"]["mt5_terminal"]`
- Line 242: `"XAUUSD.m"` → `CONFIG["scoring"]["default_symbol"]`
- Line 339: config.json path → `CONFIG["paths"]["config_path"]`
- `app.run(...)` line: use `CONFIG["server"]["host"]`, `CONFIG["server"]["port"]`

Also add `app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")` after Flask app creation.

**Step 7: Verify app still starts**

Run: `cd D:/FOREX/dashboard && python app.py`
Expected: Dashboard starts normally, loads config from yaml, no hardcoded paths remain.

**Step 8: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py config.yaml .env.example .gitignore
git commit -m "feat: replace hardcoded paths with .env + config.yaml"
```

---

### Task 3: Authentication (Login + API Key)

**Files:**
- Modify: `dashboard/app.py` (add auth decorator, login route)
- Create: `dashboard/templates/login.html`

**Step 1: Add auth imports and decorator to `app.py`**

After config loading, add:

```python
from functools import wraps
from flask import request, redirect, url_for, session

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check API key header first
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key == os.getenv("DASHBOARD_API_KEY"):
            return f(*args, **kwargs)
        # Check session
        if session.get("authenticated"):
            return f(*args, **kwargs)
        # Redirect to login for browser, 401 for API
        if request.path.startswith("/api/"):
            return jsonify({"error": "Unauthorized"}), 401
        return redirect(url_for("login"))
    return decorated
```

**Step 2: Add login/logout routes to `app.py`**

```python
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if (username == os.getenv("DASHBOARD_USER") and
                password == os.getenv("DASHBOARD_PASS")):
            session["authenticated"] = True
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
```

**Step 3: Add `@login_required` to all existing routes**

Add `@login_required` decorator to: `index`, `get_signal`, `get_history`, `export_history`, `export_history_json`, `api_signals`, `api_signals_count`, `get_status`. Do NOT add to `/login`.

**Step 4: Create `dashboard/templates/login.html`**

Dark-themed login page matching dashboard style. Form with username/password fields, submit button, error message display. Uses same Tailwind CDN and color scheme as index.html.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login — XAUUSD Signal Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: { extend: { colors: { bg: '#131313', panel: '#201f1f', gold: '#ffd700' } } }
        }
    </script>
</head>
<body class="bg-bg min-h-screen flex items-center justify-center">
    <div class="bg-panel rounded-2xl p-8 w-96 shadow-2xl border border-gray-800">
        <h1 class="text-gold text-2xl font-bold text-center mb-6">XAUUSD Signal Dashboard</h1>
        {% if error %}
        <div class="bg-red-900/30 border border-red-700 text-red-400 px-4 py-2 rounded mb-4 text-sm">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="mb-4">
                <label class="block text-gray-400 text-sm mb-1">Username</label>
                <input type="text" name="username" class="w-full bg-bg border border-gray-700 rounded px-3 py-2 text-white focus:border-gold focus:outline-none" required>
            </div>
            <div class="mb-6">
                <label class="block text-gray-400 text-sm mb-1">Password</label>
                <input type="password" name="password" class="w-full bg-bg border border-gray-700 rounded px-3 py-2 text-white focus:border-gold focus:outline-none" required>
            </div>
            <button type="submit" class="w-full bg-gold text-black font-bold py-2 rounded hover:bg-yellow-500 transition">Sign In</button>
        </form>
    </div>
</body>
</html>
```

**Step 5: Verify login flow**

Run app, open http://localhost:5000 → should redirect to /login.
Login with admin/password from .env → should redirect to dashboard.
Hit /api/signal without session or API key → should get 401.
Hit /api/signal with `X-API-Key` header → should work.

**Step 6: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py templates/login.html
git commit -m "feat: add login page and API key authentication"
```

---

### Task 4: Copy Scorer to Repo

**Files:**
- Copy: `D:/FOREX/python/gold_signal_scorer.py` → `dashboard/gold_signal_scorer.py`
- Modify: `dashboard/app.py` (update import)

**Step 1: Copy scorer**

```bash
cp D:/FOREX/python/gold_signal_scorer.py D:/FOREX/dashboard/gold_signal_scorer.py
```

**Step 2: Update import in `app.py`**

Change `from gold_signal_scorer import GoldSignalScorer` to import from local copy. Since the file is now in the same directory, the existing import should work. Remove the `sys.path.insert` for python_dir if scorer is the only thing it was needed for — but keep it if other imports depend on it.

**Step 3: Update scorer's default paths**

In the copied `gold_signal_scorer.py`, change the default `MODEL_PATH` and `CONFIG_PATH` to read from config or use relative paths.

**Step 4: Verify scorer loads**

Run: `cd D:/FOREX/dashboard && python -c "from gold_signal_scorer import GoldSignalScorer; print('OK')"`

**Step 5: Commit**

```bash
cd D:/FOREX/dashboard
git add gold_signal_scorer.py app.py
git commit -m "feat: bundle gold_signal_scorer.py into dashboard repo"
```

---

### Task 5: Telegram Alerts

**Files:**
- Modify: `dashboard/app.py` (add send_telegram_alert function, call it after db_insert)

**Step 1: Add Telegram helper function to `app.py`**

After the DB functions section, add:

```python
# ---------------------------------------------------------------------------
# Telegram Alerts
# ---------------------------------------------------------------------------
def send_telegram_alert(signal_data):
    """Send a Telegram message for high-confidence signals."""
    if not CONFIG.get("alerts", {}).get("enabled", False):
        return
    min_conf = CONFIG.get("alerts", {}).get("min_confidence", 90)
    if signal_data.get("confidence", 0) < min_conf:
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    sig = signal_data["signal"]
    icon = "\U0001f7e2" if sig == "BUY" else "\U0001f534" if sig == "SELL" else "\u26aa"
    symbol = signal_data.get("symbol", CONFIG["scoring"]["default_symbol"])
    msg = (
        f"{icon} {sig} Signal \u2014 {symbol}\n"
        f"\U0001f4b0 Price: {signal_data.get('price', 0)}\n"
        f"\U0001f4ca Confidence: {signal_data.get('confidence', 0)}% ({signal_data.get('score', 0)}/10)\n"
        f"\U0001f4c8 RSI: {signal_data.get('rsi', 0)} | ATR: {signal_data.get('atr', 0)}\n"
        f"\u23f0 {signal_data.get('time', '')}"
    )
    try:
        import requests as req
        req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg},
            timeout=5
        )
    except Exception as e:
        print(f"[WARN] Telegram alert failed: {e}")
```

**Step 2: Call alert after DB insert**

In the signal saving block (after `db_insert_signal(signal_record)`), add:

```python
send_telegram_alert(signal_record)
```

**Step 3: Test with alerts disabled**

Run app with `alerts.enabled: false` in config.yaml. Verify no errors, no messages sent.

**Step 4: Test with alerts enabled (optional — needs bot token)**

Set `alerts.enabled: true`, add bot token and chat ID to `.env`. Verify message arrives in Telegram.

**Step 5: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py
git commit -m "feat: add Telegram alerts for high-confidence signals"
```

---

### Task 6: Multi-Pair Support (DB + API)

**Files:**
- Modify: `dashboard/app.py` (DB schema migration, get_mt5_data takes symbol param, API accepts symbol param)
- Modify: `dashboard/templates/index.html` (symbol selector dropdown)

**Step 1: Add DB migration for `symbol` column**

In `init_db()`, after existing CREATE TABLE, add:

```python
# Migration: add symbol column if missing
try:
    conn.execute("SELECT symbol FROM signals LIMIT 1")
except sqlite3.OperationalError:
    conn.execute("ALTER TABLE signals ADD COLUMN symbol TEXT DEFAULT 'XAUUSD.m'")
    print("[OK] Added symbol column to signals table")

# Migration: add model_version column if missing
try:
    conn.execute("SELECT model_version FROM signals LIMIT 1")
except sqlite3.OperationalError:
    conn.execute("ALTER TABLE signals ADD COLUMN model_version TEXT DEFAULT 'v1'")
    print("[OK] Added model_version column to signals table")
```

**Step 2: Update `db_insert_signal` to include symbol**

Add `symbol` to the INSERT statement and values tuple:
```python
signal_data.get("symbol", CONFIG["scoring"]["default_symbol"]),
```

**Step 3: Update `db_get_signals` to filter by symbol**

Add `symbol_filter` parameter:
```python
def db_get_signals(limit=20, offset=0, signal_filter=None, date_from=None, date_to=None, symbol_filter=None):
    ...
    if symbol_filter:
        query += " AND symbol = ?"
        params.append(symbol_filter)
```

**Step 4: Make `get_mt5_data` accept symbol parameter**

```python
def get_mt5_data(symbol=None):
    symbol = symbol or CONFIG["scoring"]["default_symbol"]
    ...
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, CONFIG["scoring"]["bars_count"])
    ...
    tick = mt5.symbol_info_tick(symbol)
```

**Step 5: Update `/api/signal` to accept `?symbol=` query param**

```python
from flask import request
symbol = request.args.get("symbol", CONFIG["scoring"]["default_symbol"])
df, tick_info, account_info = get_mt5_data(symbol)
signal_record["symbol"] = symbol
```

**Step 6: Add symbol dropdown to `index.html`**

Add a `<select>` element next to the dashboard title. Populate from `/api/symbols` endpoint. On change, update the fetch URL to include `?symbol=SELECTED`.

Add new endpoint:
```python
@app.route("/api/symbols")
@login_required
def get_symbols():
    return jsonify(CONFIG.get("symbols", ["XAUUSD.m"]))
```

**Step 7: Add untrained model warning**

In the signal payload, add:
```python
payload["model_warning"] = symbol != "XAUUSD.m"
```

Frontend shows a yellow badge "Model not trained for this pair" when `model_warning` is true.

**Step 8: Verify**

Switch to EURUSD.m in dropdown → should fetch EURUSD data, show warning badge, save with symbol=EURUSD.m in DB.

**Step 9: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py templates/index.html
git commit -m "feat: add multi-pair support with symbol selector"
```

---

### Task 7: WebSocket Real-Time Updates

**Files:**
- Modify: `dashboard/app.py` (replace app.run with SocketIO, add background thread)
- Modify: `dashboard/templates/index.html` (replace fetch polling with socket.io client)

**Step 1: Replace Flask run with SocketIO in `app.py`**

At top, add:
```python
from flask_socketio import SocketIO, emit
```

After app creation:
```python
socketio = SocketIO(app, cors_allowed_origins="*")
```

Replace `app.run(...)` at bottom with:
```python
socketio.run(app, host=CONFIG["server"]["host"], port=CONFIG["server"]["port"], debug=False)
```

**Step 2: Add background thread for signal checking**

```python
import threading

def background_signal_checker():
    """Check for new M30 bars and emit updates via WebSocket."""
    global _last_saved_bar_time
    while True:
        try:
            symbol = CONFIG["scoring"]["default_symbol"]
            df, tick_info, account_info = get_mt5_data(symbol)
            current_bar = str(df.index[-1])
            if current_bar != _last_saved_bar_time:
                # New bar — score and emit
                scorer = get_scorer()
                if scorer:
                    features = scorer.calculate_features(df)
                    result = scorer.score(features)
                    # Build payload (same as /api/signal logic)
                    # ... emit to all connected clients
                    socketio.emit("signal_update", payload)
        except Exception as e:
            print(f"[WARN] Background checker: {e}")
        socketio.sleep(10)  # Check every 10 seconds
```

Start the thread before `socketio.run`:
```python
socketio.start_background_task(background_signal_checker)
```

**Step 3: Update `index.html` — add socket.io client**

Add before closing `</body>`:
```html
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
    const socket = io();
    socket.on('signal_update', function(data) {
        updateDashboard(data);  // Reuse existing update function
    });
    socket.on('connect', () => console.log('WebSocket connected'));
    socket.on('disconnect', () => console.log('WebSocket disconnected, falling back to polling'));
</script>
```

Keep the existing `setInterval` fetch as fallback — if socket disconnects, polling still works.

**Step 4: Verify**

Open dashboard, check browser console for "WebSocket connected". Wait for new M30 bar — dashboard should update without page refresh.

**Step 5: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py templates/index.html
git commit -m "feat: add WebSocket real-time updates via Flask-SocketIO"
```

---

### Task 8: Signal Browser Page (`/history`)

**Files:**
- Modify: `dashboard/app.py` (add /history route)
- Create: `dashboard/templates/history.html`

**Step 1: Add `/history` route to `app.py`**

```python
@app.route("/history")
@login_required
def history_page():
    return render_template("history.html")
```

The page is fully client-side — it fetches from `/api/signals` with query params.

**Step 2: Create `dashboard/templates/history.html`**

Dark-themed table page with:
- Filter bar: date range (from/to inputs), signal type dropdown (ALL/BUY/SELL/NO_TRADE), min confidence slider, symbol dropdown
- Sortable table columns: Time, Symbol, Signal, Confidence, Score, Price, RSI, ATR, EMA Trend
- Pagination: Previous/Next buttons, showing "Page X of Y"
- Export buttons (CSV/JSON) linking to existing export endpoints
- Total count display from `/api/signals/count`

JavaScript fetches `/api/signals?limit=50&offset=0&signal=BUY&from=2026-03-27&symbol=XAUUSD.m` and renders table rows.

**Step 3: Add navigation link**

Add a "History" link in the dashboard header (both `index.html` and `history.html`) for easy navigation between pages.

**Step 4: Verify**

Open /history → should show paginated signal table. Apply filters → table updates. Click column header → sorts.

**Step 5: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py templates/history.html templates/index.html
git commit -m "feat: add signal browser page with filters and pagination"
```

---

### Task 9: Backtest Simulator Page (`/simulate`)

**Files:**
- Modify: `dashboard/app.py` (add /simulate route, /api/simulate endpoint, simulations table)
- Create: `dashboard/templates/simulate.html`

**Step 1: Add simulations table to `init_db()`**

```python
conn.execute("""
    CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        symbol TEXT,
        date_from TEXT,
        date_to TEXT,
        total_signals INTEGER,
        buy_count INTEGER,
        sell_count INTEGER,
        profitable_pct REAL,
        avg_return REAL,
        results_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")
```

**Step 2: Add `/simulate` route and `/api/simulate` endpoint**

```python
@app.route("/simulate")
@login_required
def simulate_page():
    return render_template("simulate.html")

@app.route("/api/simulate", methods=["POST"])
@login_required
def run_simulation():
    data = request.get_json()
    symbol = data.get("symbol", "XAUUSD.m")
    date_from = data.get("from")
    date_to = data.get("to")

    # Load parquet data
    data_dir = CONFIG["paths"]["data_dir"]
    # Find matching parquet file for symbol
    # Read bars in date range
    # Run scorer bar-by-bar
    # Calculate profitability: compare signal price vs price N bars later
    # Save to simulations table
    # Return results
```

**Step 3: Create `dashboard/templates/simulate.html`**

Dark-themed page with:
- Form: symbol dropdown, date from/to pickers, "Run Simulation" button
- Progress bar during simulation
- Results section: summary stats (total signals, BUY/SELL split, profitable %, avg return)
- Signal-by-signal table with outcome column (profitable/not)
- Past simulations list from DB

**Step 4: Verify**

Select XAUUSD.m, date range with available parquet data, run simulation. Should show results.

**Step 5: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py templates/simulate.html
git commit -m "feat: add backtest simulator page with historical scoring"
```

---

### Task 10: Model Management Panel (`/admin/models`)

**Files:**
- Modify: `dashboard/app.py` (add admin routes, signal_outcomes table, retrain logic, A/B test logic)
- Create: `dashboard/templates/admin_models.html`

**Step 1: Add `signal_outcomes` table to `init_db()`**

```python
conn.execute("""
    CREATE TABLE IF NOT EXISTS signal_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        signal_type TEXT,
        entry_price REAL,
        price_after_6bars REAL,
        price_after_12bars REAL,
        outcome TEXT,
        pnl_points REAL,
        model_version TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (signal_id) REFERENCES signals(id)
    )
""")
```

**Step 2: Add model management routes**

```python
@app.route("/admin/models")
@login_required
def admin_models():
    return render_template("admin_models.html")

@app.route("/api/admin/model-stats")
@login_required
def model_stats():
    """Return current model info, feature importance, accuracy."""
    scorer = get_scorer()
    if not scorer:
        return jsonify({"error": "No model loaded"}), 500
    # Return model metadata from config.json
    ...

@app.route("/api/admin/retrain", methods=["POST"])
@login_required
def retrain_model():
    """Retrain XGBoost on latest data in background thread."""
    # Read parquet data from data_dir
    # Train new model
    # Save as gold_signal_model_YYYYMMDD.json
    # Return comparison metrics
    ...

@app.route("/api/admin/models/list")
@login_required
def list_models():
    """List all saved model files."""
    ...

@app.route("/api/admin/models/promote", methods=["POST"])
@login_required
def promote_model():
    """Switch active model to a different version."""
    ...

@app.route("/api/admin/ab-test", methods=["POST"])
@login_required
def toggle_ab_test():
    """Enable/disable A/B testing between two models."""
    ...
```

**Step 3: Create `dashboard/templates/admin_models.html`**

Three-tab page:
- **Current Model tab:** Stats table (features, accuracy, winrates, training date), feature importance bar chart (Chart.js), live accuracy from signal_outcomes
- **Retrain tab:** Date range picker, train/val split slider, "Start Training" button, progress bar, comparison table (old vs new)
- **A/B Test tab:** Model A/B selectors, toggle switch, results comparison table after N bars, "Promote" button

**Step 4: Add outcome tracking**

In the background signal checker, after each new bar: look back at signals from 6 and 12 bars ago, record the actual price move in `signal_outcomes`. This builds the live accuracy data.

**Step 5: Verify**

Open /admin/models → see current model stats. Click retrain with date range → new model trained. Enable A/B test → both models score each bar.

**Step 6: Commit**

```bash
cd D:/FOREX/dashboard
git add app.py templates/admin_models.html
git commit -m "feat: add model management panel with retrain and A/B testing"
```

---

### Task 11: Unit Tests

**Files:**
- Create: `dashboard/tests/__init__.py`
- Create: `dashboard/tests/conftest.py`
- Create: `dashboard/tests/test_db.py`
- Create: `dashboard/tests/test_auth.py`
- Create: `dashboard/tests/test_config.py`
- Create: `dashboard/tests/test_alerts.py`
- Create: `dashboard/tests/test_api.py`

**Step 1: Create test fixtures in `conftest.py`**

```python
import pytest
import os
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

@pytest.fixture
def app():
    """Create test app with temp database."""
    os.environ["FLASK_SECRET_KEY"] = "test-secret"
    os.environ["DASHBOARD_API_KEY"] = "test-api-key"
    os.environ["DASHBOARD_USER"] = "testuser"
    os.environ["DASHBOARD_PASS"] = "testpass"

    import app as dashboard_app
    dashboard_app.app.config["TESTING"] = True
    dashboard_app.DB_PATH = tempfile.mktemp(suffix=".db")
    dashboard_app.init_db()
    yield dashboard_app.app
    os.unlink(dashboard_app.DB_PATH)

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def authed_client(client):
    """Client with valid session."""
    client.post("/login", data={"username": "testuser", "password": "testpass"})
    return client
```

**Step 2: Write `test_db.py`**

```python
def test_insert_and_query(app):
    import app as dashboard_app
    signal = {"time": "2026-03-27 12:00:00", "signal": "BUY", "confidence": 95.0,
              "price": 4430.0, "bid": 4429.7, "score": 10, "buy_prob": 95.0,
              "sell_prob": 2.0, "no_trade_prob": 3.0, "rsi": 62.0, "stoch_k": 88.0,
              "atr": 24.0, "atr_vs_avg": 0.94, "adx": 0, "bb_position": 0.98,
              "ema_trend": "BEARISH", "macd": -5.0, "body_ratio": 0.05,
              "is_bullish": False, "reasons": ["test reason"]}
    dashboard_app.db_insert_signal(signal)
    results = dashboard_app.db_get_signals(limit=10)
    assert len(results) == 1
    assert results[0]["signal"] == "BUY"

def test_stats(app):
    import app as dashboard_app
    stats = dashboard_app.db_get_stats()
    assert stats["total_all_time"] == 0

def test_dedup_not_in_db_layer(app):
    """DB layer always inserts — dedup is in the route."""
    import app as dashboard_app
    signal = {"time": "2026-03-27 12:00:00", "signal": "BUY", "confidence": 95.0,
              "price": 4430.0, "bid": 4429.7, "score": 10, "buy_prob": 95.0,
              "sell_prob": 2.0, "no_trade_prob": 3.0, "rsi": 62.0, "stoch_k": 88.0,
              "atr": 24.0, "atr_vs_avg": 0.94, "adx": 0, "bb_position": 0.98,
              "ema_trend": "BEARISH", "macd": -5.0, "body_ratio": 0.05,
              "is_bullish": False, "reasons": []}
    dashboard_app.db_insert_signal(signal)
    dashboard_app.db_insert_signal(signal)
    stats = dashboard_app.db_get_stats()
    assert stats["total_all_time"] == 2  # No dedup at DB layer
```

**Step 3: Write `test_auth.py`**

```python
def test_unauthenticated_redirect(client):
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]

def test_login_success(client):
    resp = client.post("/login", data={"username": "testuser", "password": "testpass"})
    assert resp.status_code == 302

def test_login_failure(client):
    resp = client.post("/login", data={"username": "wrong", "password": "wrong"})
    assert resp.status_code == 200
    assert b"Invalid" in resp.data

def test_api_key_auth(client):
    resp = client.get("/api/status", headers={"X-API-Key": "test-api-key"})
    assert resp.status_code == 200

def test_api_no_auth_401(client):
    resp = client.get("/api/status")
    assert resp.status_code == 401
```

**Step 4: Write `test_alerts.py`**

```python
from unittest.mock import patch, MagicMock

def test_alert_skipped_when_disabled(app):
    import app as dashboard_app
    dashboard_app.CONFIG["alerts"]["enabled"] = False
    # Should return without sending
    dashboard_app.send_telegram_alert({"signal": "BUY", "confidence": 99})

@patch("requests.post")
def test_alert_sent_when_enabled(mock_post, app):
    import app as dashboard_app
    dashboard_app.CONFIG["alerts"]["enabled"] = True
    dashboard_app.CONFIG["alerts"]["min_confidence"] = 90
    import os
    os.environ["TELEGRAM_BOT_TOKEN"] = "fake-token"
    os.environ["TELEGRAM_CHAT_ID"] = "12345"
    dashboard_app.send_telegram_alert({"signal": "BUY", "confidence": 95, "price": 4430,
                                        "score": 10, "rsi": 62, "atr": 24, "time": "2026-03-27"})
    mock_post.assert_called_once()

def test_alert_skipped_below_threshold(app):
    import app as dashboard_app
    dashboard_app.CONFIG["alerts"]["enabled"] = True
    dashboard_app.CONFIG["alerts"]["min_confidence"] = 90
    # 80% confidence < 90% threshold — should not send
    with patch("requests.post") as mock_post:
        dashboard_app.send_telegram_alert({"signal": "BUY", "confidence": 80})
        mock_post.assert_not_called()
```

**Step 5: Run all tests**

Run: `cd D:/FOREX/dashboard && pytest tests/ -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
cd D:/FOREX/dashboard
git add tests/
git commit -m "test: add unit tests for db, auth, config, and alerts"
```

---

### Task 12: Update README and Final Push

**Files:**
- Modify: `dashboard/README.md`

**Step 1: Update README**

Add sections for:
- New setup steps (.env, config.yaml)
- Authentication (login page, API key)
- Telegram alerts setup
- Multi-pair usage
- Signal browser and simulator pages
- Model management panel
- Running tests
- Updated API endpoints table

**Step 2: Final commit and push**

```bash
cd D:/FOREX/dashboard
git add -A
git commit -m "docs: update README with all new features"
git push
```

---

## Build Order Summary

| Phase | Task | Est. Complexity |
|-------|------|----------------|
| 1 | Install deps | Trivial |
| 2 | Config (.env + yaml) | Low |
| 3 | Auth (login + API key) | Low |
| 4 | Copy scorer to repo | Trivial |
| 5 | Telegram alerts | Low |
| 6 | Multi-pair (DB + API + UI) | Medium |
| 7 | WebSocket | Medium |
| 8 | Signal browser page | Medium |
| 9 | Backtest simulator | Medium-High |
| 10 | Model management panel | High |
| 11 | Unit tests | Medium |
| 12 | README + push | Trivial |

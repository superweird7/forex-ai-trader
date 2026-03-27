"""
Gold AI Signal Scorer — Web Dashboard
Flask backend serving live XAUUSD signals from the XGBoost model via MT5.
"""

import sys
import os
import json
import sqlite3
import traceback
from datetime import datetime, timedelta
from threading import Lock

import yaml
from dotenv import load_dotenv
from functools import wraps
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_socketio import SocketIO, emit

# Load environment and config
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

CONFIG.setdefault("trade_manager", {
    "atr_period": 14,
    "sl_multiplier": 1.5,
    "tp_ratio": 2.5,
    "high_vol_threshold": 1.5,
    "high_vol_duration": 900,
    "normal_vol_duration": 1800,
    "low_vol_duration": 2700,
})

# Add project paths
sys.path.insert(0, CONFIG["paths"]["python_dir"])

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.auto_reload = True
app.jinja_env.bytecode_cache = None


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


signal_history = []
history_lock = Lock()
_last_saved_bar_time = None  # Track last M30 bar to avoid duplicate signals
daily_stats = {
    "total_signals": 0,
    "buy_count": 0,
    "sell_count": 0,
    "no_trade_count": 0,
    "date": datetime.now().strftime("%Y-%m-%d"),
}

# ---------------------------------------------------------------------------
# SQLite Database
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "signals.db")


def get_db():
    """Get a thread-local database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the signals table if it doesn't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT NOT NULL,
            price REAL,
            bid REAL,
            signal TEXT NOT NULL,
            confidence REAL,
            score INTEGER,
            buy_prob REAL,
            sell_prob REAL,
            no_trade_prob REAL,
            rsi REAL,
            stoch_k REAL,
            atr REAL,
            atr_vs_avg REAL,
            adx REAL,
            bb_position REAL,
            ema_trend TEXT,
            macd REAL,
            body_ratio REAL,
            is_bullish INTEGER,
            reasons TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(time)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_signal ON signals(signal)
    """)
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

    conn.execute("""
        CREATE TABLE IF NOT EXISTS active_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            tp_price REAL,
            sl_price REAL,
            max_duration_sec INTEGER,
            atr_value REAL,
            started_at TEXT,
            closed_at TEXT,
            exit_reason TEXT,
            exit_price REAL,
            pnl_points REAL,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"[OK] Database ready: {DB_PATH}")


def db_insert_signal(signal_data):
    """Insert a signal record into the database."""
    conn = get_db()
    reasons = signal_data.get("reasons", [])
    if isinstance(reasons, list):
        reasons = " | ".join(reasons)
    conn.execute("""
        INSERT INTO signals (time, price, bid, signal, confidence, score,
            buy_prob, sell_prob, no_trade_prob, rsi, stoch_k, atr, atr_vs_avg,
            adx, bb_position, ema_trend, macd, body_ratio, is_bullish, reasons,
            symbol)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal_data.get("time"),
        signal_data.get("price"),
        signal_data.get("bid"),
        signal_data.get("signal"),
        signal_data.get("confidence"),
        signal_data.get("score"),
        signal_data.get("buy_prob"),
        signal_data.get("sell_prob"),
        signal_data.get("no_trade_prob"),
        signal_data.get("rsi"),
        signal_data.get("stoch_k"),
        signal_data.get("atr"),
        signal_data.get("atr_vs_avg"),
        signal_data.get("adx"),
        signal_data.get("bb_position"),
        signal_data.get("ema_trend"),
        signal_data.get("macd"),
        signal_data.get("body_ratio"),
        1 if signal_data.get("is_bullish") else 0,
        reasons,
        signal_data.get("symbol", "XAUUSD.m"),
    ))
    conn.commit()
    conn.close()


def db_get_signals(limit=20, offset=0, signal_filter=None, date_from=None, date_to=None, symbol_filter=None):
    """Query signals from the database with optional filters."""
    conn = get_db()
    query = "SELECT * FROM signals WHERE 1=1"
    params = []
    if symbol_filter:
        query += " AND symbol = ?"
        params.append(symbol_filter)
    if signal_filter:
        query += " AND signal = ?"
        params.append(signal_filter)
    if date_from:
        query += " AND time >= ?"
        params.append(date_from)
    if date_to:
        query += " AND time <= ?"
        params.append(date_to)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def db_get_stats():
    """Get signal counts from the database."""
    conn = get_db()
    today = datetime.now().strftime("%Y-%m-%d")
    total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    today_total = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE time LIKE ?", (f"{today}%",)
    ).fetchone()[0]
    today_buys = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE time LIKE ? AND signal='BUY'", (f"{today}%",)
    ).fetchone()[0]
    today_sells = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE time LIKE ? AND signal='SELL'", (f"{today}%",)
    ).fetchone()[0]
    today_notrade = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE time LIKE ? AND signal='NO_TRADE'", (f"{today}%",)
    ).fetchone()[0]
    conn.close()
    return {
        "total_all_time": total,
        "total_signals": today_total,
        "buy_count": today_buys,
        "sell_count": today_sells,
        "no_trade_count": today_notrade,
        "date": today,
    }


# ---------------------------------------------------------------------------
# Telegram Alerts
# ---------------------------------------------------------------------------
def send_telegram_alert(signal_data):
    """Send Telegram message for high-confidence signals."""
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
            timeout=5,
        )
    except Exception as e:
        print(f"[WARN] Telegram alert failed: {e}")


def send_trade_telegram(trade_data, event="opened"):
    """Send Telegram message for trade events."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    if not CONFIG.get("alerts", {}).get("enabled", False):
        return

    d = trade_data
    if event == "opened":
        icon = "\U0001f7e2" if d["direction"] == "BUY" else "\U0001f534"
        dur_min = d.get("max_duration_sec", 1800) // 60
        msg = (
            f"{icon} TRADE {d['direction']} \u2014 {d.get('symbol', 'XAUUSD.m')}\n"
            f"\U0001f4b0 Entry: {d['entry_price']}\n"
            f"\U0001f3af TP: {d['tp_price']} (+{d.get('tp_distance', 0)}pts)\n"
            f"\U0001f6d1 SL: {d['sl_price']} (-{d.get('sl_distance', 0)}pts)\n"
            f"\u23f0 Max: {dur_min} min\n"
            f"\u26a1 ACT NOW"
        )
    else:  # closed
        is_win = d.get("pnl_points", 0) > 0
        icon = "\u2705" if is_win else "\u274c"
        reason_map = {"tp_hit": "TARGET HIT", "sl_hit": "STOPPED OUT", "expired": "EXPIRED", "manual": "MANUAL", "new_signal": "NEW SIGNAL", "reversed": "REVERSED"}
        reason = reason_map.get(d.get("exit_reason", ""), d.get("exit_reason", ""))
        pnl = d.get("pnl_points", 0)
        pnl_icon = "\U0001f4c8" if is_win else "\U0001f4c9"
        msg = (
            f"{icon} TRADE CLOSED \u2014 {reason}\n"
            f"\U0001f4b0 Entry: {d.get('entry_price', 0)} \u2192 Exit: {d.get('exit_price', 0)}\n"
            f"{pnl_icon} PnL: {pnl:+.2f} points\n"
            f"\u23f1 Duration: {d.get('duration_sec', 0) // 60}min {d.get('duration_sec', 0) % 60}s"
        )

    try:
        import requests as req
        req.post(f"https://api.telegram.org/bot{token}/sendMessage",
                 json={"chat_id": chat_id, "text": msg}, timeout=5)
    except Exception as e:
        print(f"[WARN] Trade Telegram failed: {e}")


# ---------------------------------------------------------------------------
# Trade Manager
# ---------------------------------------------------------------------------
def calculate_trade_levels(direction, entry_price, atr_value, atr_vs_avg):
    """Calculate TP, SL, and max duration for a trade."""
    tm = CONFIG.get("trade_manager", {})
    sl_mult = tm.get("sl_multiplier", 1.5)
    tp_ratio = tm.get("tp_ratio", 2.5)

    sl_distance = atr_value * sl_mult
    tp_distance = sl_distance * tp_ratio

    if direction == "BUY":
        sl_price = round(entry_price - sl_distance, 2)
        tp_price = round(entry_price + tp_distance, 2)
    else:  # SELL
        sl_price = round(entry_price + sl_distance, 2)
        tp_price = round(entry_price - tp_distance, 2)

    # ATR-based duration
    high_thresh = tm.get("high_vol_threshold", 1.5)
    if atr_vs_avg > high_thresh:
        max_duration = tm.get("high_vol_duration", 900)
    elif atr_vs_avg < 0.8:
        max_duration = tm.get("low_vol_duration", 2700)
    else:
        max_duration = tm.get("normal_vol_duration", 1800)

    return tp_price, sl_price, sl_distance, tp_distance, max_duration


def open_trade(signal_record):
    """Open a new active trade from a signal."""
    direction = signal_record["signal"]
    if direction not in ("BUY", "SELL"):
        return None

    conf_threshold = CONFIG.get("scoring", {}).get("confidence_threshold", 80)
    if signal_record.get("confidence", 0) < conf_threshold:
        return None

    # Close any existing active trades first
    conn = get_db()
    conn.execute("UPDATE active_trades SET status='closed', exit_reason='new_signal', closed_at=? WHERE status='active'",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    conn.commit()

    entry_price = signal_record.get("price", 0) if direction == "BUY" else signal_record.get("bid", 0)
    atr_value = signal_record.get("atr", 20)
    atr_vs_avg = signal_record.get("atr_vs_avg", 1.0)

    tp_price, sl_price, sl_dist, tp_dist, max_duration = calculate_trade_levels(
        direction, entry_price, atr_value, atr_vs_avg
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO active_trades (symbol, direction, entry_price, tp_price, sl_price,
            max_duration_sec, atr_value, started_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
    """, (
        signal_record.get("symbol", CONFIG["scoring"]["default_symbol"]),
        direction, entry_price, tp_price, sl_price, max_duration, atr_value, now
    ))
    conn.commit()

    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    trade_data = {
        "trade_id": trade_id,
        "direction": direction,
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "sl_distance": round(sl_dist, 2),
        "tp_distance": round(tp_dist, 2),
        "max_duration_sec": max_duration,
        "atr_value": atr_value,
        "started_at": now,
        "status": "active",
        "symbol": signal_record.get("symbol", CONFIG["scoring"]["default_symbol"]),
    }

    print(f"[TRADE] Opened {direction} @ {entry_price} | TP={tp_price} SL={sl_price} | {max_duration}s")
    return trade_data


def check_active_trades(current_ask, current_bid):
    """Check if any active trades hit TP, SL, or expired. Returns list of closed trades."""
    conn = get_db()
    trades = conn.execute("SELECT * FROM active_trades WHERE status='active'").fetchall()
    closed = []

    for trade in trades:
        trade_dict = dict(trade)
        trade_id = trade_dict["id"]
        direction = trade_dict["direction"]
        entry = trade_dict["entry_price"]
        tp = trade_dict["tp_price"]
        sl = trade_dict["sl_price"]
        started = datetime.strptime(trade_dict["started_at"], "%Y-%m-%d %H:%M:%S")
        max_dur = trade_dict["max_duration_sec"] or 1800

        current_price = current_bid if direction == "BUY" else current_ask
        elapsed = (datetime.now() - started).total_seconds()

        exit_reason = None
        exit_price = current_price

        if direction == "BUY":
            if current_bid >= tp:
                exit_reason = "tp_hit"
                exit_price = tp
            elif current_bid <= sl:
                exit_reason = "sl_hit"
                exit_price = sl
        else:  # SELL
            if current_ask <= tp:
                exit_reason = "tp_hit"
                exit_price = tp
            elif current_ask >= sl:
                exit_reason = "sl_hit"
                exit_price = sl

        if not exit_reason and elapsed >= max_dur:
            exit_reason = "expired"

        if exit_reason:
            if direction == "BUY":
                pnl = round(exit_price - entry, 2)
            else:
                pnl = round(entry - exit_price, 2)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("""
                UPDATE active_trades SET status='closed', exit_reason=?, exit_price=?,
                    pnl_points=?, closed_at=? WHERE id=?
            """, (exit_reason, exit_price, pnl, now, trade_id))
            conn.commit()

            closed.append({
                "trade_id": trade_id,
                "direction": direction,
                "entry_price": entry,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_points": pnl,
                "duration_sec": int(elapsed),
            })
            print(f"[TRADE] Closed {direction} | {exit_reason} | PnL={pnl:+.2f}pts")

    conn.close()
    return closed


def get_active_trade():
    """Get the current active trade if any."""
    conn = get_db()
    trade = conn.execute("SELECT * FROM active_trades WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if trade:
        trade_dict = dict(trade)
        started = datetime.strptime(trade_dict["started_at"], "%Y-%m-%d %H:%M:%S")
        elapsed = (datetime.now() - started).total_seconds()
        max_dur = trade_dict["max_duration_sec"] or 1800
        trade_dict["elapsed_sec"] = int(elapsed)
        trade_dict["remaining_sec"] = max(0, int(max_dur - elapsed))
        return trade_dict
    return None


# Initialize database on import
init_db()

# ---------------------------------------------------------------------------
# Lazy-load scorer (so app starts even if model missing)
# ---------------------------------------------------------------------------
_scorer = None
_scorer_error = None


def get_scorer():
    global _scorer, _scorer_error
    if _scorer is not None:
        return _scorer
    if _scorer_error is not None:
        return None
    try:
        from gold_signal_scorer import GoldSignalScorer

        _scorer = GoldSignalScorer(
            model_path=CONFIG["paths"]["model_path"],
            config_path=CONFIG["paths"]["config_path"],
        )
        print(f"[OK] Scorer loaded — {len(_scorer.feature_names)} features")
        return _scorer
    except Exception as e:
        _scorer_error = str(e)
        print(f"[WARN] Scorer failed to load: {e}")
        return None


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------
def init_mt5():
    """Try to initialize MT5 connection. Returns True/False."""
    try:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            # Try with explicit path
            mt5.initialize(CONFIG["paths"]["mt5_terminal"])
        return mt5.terminal_info() is not None
    except Exception:
        return False


def get_mt5_data(symbol=None):
    """
    Fetch live M30 bars and tick from MT5.
    Returns (df, tick_info, account_info) or raises.
    """
    import MetaTrader5 as mt5

    symbol = symbol or CONFIG["scoring"]["default_symbol"]

    if not mt5.initialize():
        mt5.initialize(CONFIG["paths"]["mt5_terminal"])

    info = mt5.terminal_info()
    if info is None:
        raise ConnectionError("MT5 terminal not responding")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 200)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No {symbol} M30 data from MT5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    tick = mt5.symbol_info_tick(symbol)
    account = mt5.account_info()

    tick_info = None
    if tick:
        tick_info = {
            "ask": round(tick.ask, 2),
            "bid": round(tick.bid, 2),
            "spread": round((tick.ask - tick.bid) * 100, 1),
            "time": datetime.fromtimestamp(tick.time).strftime("%H:%M:%S"),
        }

    account_info = None
    if account:
        account_info = {
            "balance": round(account.balance, 2),
            "equity": round(account.equity, 2),
            "profit": round(account.profit, 2),
            "margin_free": round(account.margin_free, 2),
        }

    return df, tick_info, account_info


def reset_daily_stats_if_needed():
    """Reset daily stats at midnight."""
    today = datetime.now().strftime("%Y-%m-%d")
    if daily_stats["date"] != today:
        daily_stats["total_signals"] = 0
        daily_stats["buy_count"] = 0
        daily_stats["sell_count"] = 0
        daily_stats["no_trade_count"] = 0
        daily_stats["date"] = today


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
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


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/history")
@login_required
def history_page():
    return render_template("history.html")


@app.route("/api/signal")
@login_required
def get_signal():
    """Main endpoint: fetch live data, score, return everything."""
    reset_daily_stats_if_needed()

    symbol = request.args.get("symbol", CONFIG["scoring"]["default_symbol"])

    scorer = get_scorer()
    mt5_connected = False
    error_msg = None

    # Defaults for when things fail
    payload = {
        "price": 0,
        "bid": 0,
        "spread": 0,
        "signal": "NO_TRADE",
        "confidence": 0,
        "score": 0,
        "buy_prob": 0,
        "sell_prob": 0,
        "no_trade_prob": 100,
        "reasons": ["System offline"],
        "rsi": 50,
        "stoch_k": 50,
        "atr": 0,
        "adx": 0,
        "bb_position": 0.5,
        "ema_trend": "FLAT",
        "macd": 0,
        "macd_signal": 0,
        "atr_vs_avg": 1.0,
        "body_ratio": 0,
        "is_bullish": False,
        "balance": 0,
        "equity": 0,
        "profit": 0,
        "daily_change_pct": 0,
        "history": [],
        "mt5_connected": False,
        "daily_stats": daily_stats,
        "model_loaded": scorer is not None,
        "error": None,
        "config": {},
    }

    # Load config stats for the sidebar
    try:
        with open(CONFIG["paths"]["config_path"], "r") as f:
            config = json.load(f)
        payload["config"] = {
            "buy_winrate": round(
                config.get("val_metrics", {}).get("buy_highconf_winrate", 0) * 100, 1
            ),
            "sell_winrate": round(
                config.get("val_metrics", {}).get("sell_highconf_winrate", 0) * 100, 1
            ),
            "accuracy": round(
                config.get("val_metrics", {}).get("accuracy", 0) * 100, 1
            ),
            "confidence_threshold": config.get("confidence_threshold", 0.6),
            "n_train": config.get("n_train_samples", 0),
            "n_val": config.get("n_val_samples", 0),
            "feature_importance": config.get("feature_importance", {}),
        }
    except Exception:
        pass

    # Try to get MT5 data
    try:
        df, tick_info, account_info = get_mt5_data(symbol)
        mt5_connected = True

        if tick_info:
            payload["price"] = tick_info["ask"]
            payload["bid"] = tick_info["bid"]
            payload["spread"] = tick_info["spread"]

        if account_info:
            payload["balance"] = account_info["balance"]
            payload["equity"] = account_info["equity"]
            payload["profit"] = account_info["profit"]

        # Daily change: compare current close to first bar of today
        try:
            today = datetime.now().date()
            today_bars = df[df.index.date == today]
            if len(today_bars) > 0:
                open_price = today_bars["open"].iloc[0]
                current_price = df["close"].iloc[-1]
                if open_price > 0:
                    payload["daily_change_pct"] = round(
                        (current_price - open_price) / open_price * 100, 3
                    )
        except Exception:
            pass

        # Score the signal
        if scorer is not None:
            try:
                features = scorer.calculate_features(df)
                result = scorer.score(features)

                payload["signal"] = result["signal"]
                payload["confidence"] = round(result["confidence"] * 100, 1)
                payload["score"] = result["score"]
                payload["buy_prob"] = round(result["buy_prob"] * 100, 1)
                payload["sell_prob"] = round(result["sell_prob"] * 100, 1)
                payload["no_trade_prob"] = round(result["no_trade_prob"] * 100, 1)
                payload["reasons"] = result["top_reasons"]

                # Extract indicator values from features
                payload["rsi"] = round(features.get("rsi14", 50), 1)
                payload["stoch_k"] = round(features.get("stoch_k", 50), 1)
                payload["atr"] = round(features.get("atr14", 0), 2)
                payload["adx"] = round(features.get("adx", 0), 1)
                payload["bb_position"] = round(features.get("bb_position", 0.5), 3)
                payload["ema_trend"] = (
                    "BULLISH" if features.get("ema_trend", 0) == 1 else "BEARISH"
                )
                payload["macd"] = round(features.get("macd", 0), 2)
                payload["macd_signal"] = round(features.get("macd_signal", 0), 2)
                payload["atr_vs_avg"] = round(features.get("atr_vs_avg", 1.0), 2)
                payload["body_ratio"] = round(features.get("body_ratio", 0), 2)
                payload["is_bullish"] = bool(features.get("is_bullish", 0))

                # Build signal record
                signal_record = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": payload.get("price", 0),
                    "bid": payload.get("bid", 0),
                    "signal": result["signal"],
                    "confidence": round(result["confidence"] * 100, 1),
                    "score": result["score"],
                    "buy_prob": round(result.get("buy_prob", 0) * 100, 1),
                    "sell_prob": round(result.get("sell_prob", 0) * 100, 1),
                    "no_trade_prob": round(result.get("no_trade_prob", 0) * 100, 1),
                    "rsi": payload.get("rsi", 0),
                    "stoch_k": payload.get("stoch_k", 0),
                    "atr": payload.get("atr", 0),
                    "atr_vs_avg": payload.get("atr_vs_avg", 0),
                    "adx": payload.get("adx", 0),
                    "bb_position": payload.get("bb_position", 0),
                    "ema_trend": payload.get("ema_trend", ""),
                    "macd": payload.get("macd", 0),
                    "body_ratio": payload.get("body_ratio", 0),
                    "is_bullish": payload.get("is_bullish", False),
                    "reasons": result.get("top_reasons", []),
                }
                signal_record["symbol"] = symbol

                # Only save once per M30 bar (avoid duplicates)
                global _last_saved_bar_time
                current_bar_time = str(df.index[-1])
                if current_bar_time != _last_saved_bar_time:
                    _last_saved_bar_time = current_bar_time

                    # Save to database
                    try:
                        db_insert_signal(signal_record)
                    except Exception as db_err:
                        print(f"[WARN] DB insert failed: {db_err}")

                    send_telegram_alert(signal_record)

                    # Also keep in memory for fast access
                    with history_lock:
                        signal_history.append(signal_record)
                        if len(signal_history) > 500:
                            signal_history.pop(0)

            except Exception as e:
                payload["error"] = f"Scoring error: {e}"
                traceback.print_exc()
        else:
            payload["error"] = f"Model not loaded: {_scorer_error}"

    except Exception as e:
        payload["error"] = f"MT5 error: {e}"
        mt5_connected = False

    payload["mt5_connected"] = mt5_connected
    try:
        payload["daily_stats"] = db_get_stats()
    except Exception:
        payload["daily_stats"] = daily_stats

    with history_lock:
        payload["history"] = list(signal_history[-20:])

    payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["model_warning"] = symbol != "XAUUSD.m"

    return jsonify(payload)


@app.route("/api/history")
@login_required
def get_history():
    with history_lock:
        return jsonify(signal_history[-20:])


@app.route("/api/export")
@login_required
def export_history():
    """Export full signal history as CSV for later analysis."""
    import csv
    import io
    from flask import Response

    with history_lock:
        data = list(signal_history)

    if not data:
        return jsonify({"error": "No signals to export"})

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "time", "price", "bid", "signal", "confidence", "score",
        "buy_prob", "sell_prob", "no_trade_prob",
        "rsi", "stoch_k", "atr", "atr_vs_avg", "adx",
        "bb_position", "ema_trend", "macd", "body_ratio", "is_bullish",
        "reasons"
    ])
    writer.writeheader()
    for row in data:
        export_row = {
            "time": row.get("time", ""),
            "price": row.get("price", ""),
            "bid": row.get("bid", ""),
            "signal": row.get("signal", ""),
            "confidence": row.get("confidence", ""),
            "score": row.get("score", ""),
            "buy_prob": row.get("buy_prob", ""),
            "sell_prob": row.get("sell_prob", ""),
            "no_trade_prob": row.get("no_trade_prob", ""),
            "rsi": row.get("rsi", ""),
            "stoch_k": row.get("stoch_k", ""),
            "atr": row.get("atr", ""),
            "atr_vs_avg": row.get("atr_vs_avg", ""),
            "adx": row.get("adx", ""),
            "bb_position": row.get("bb_position", ""),
            "ema_trend": row.get("ema_trend", ""),
            "macd": row.get("macd", ""),
            "body_ratio": row.get("body_ratio", ""),
            "is_bullish": row.get("is_bullish", ""),
            "reasons": " | ".join(row.get("reasons", [])) if isinstance(row.get("reasons"), list) else row.get("reasons", ""),
        }
        writer.writerow(export_row)

    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=gold_signals_{timestamp}.csv"}
    )


@app.route("/api/export/json")
@login_required
def export_history_json():
    """Export full signal history as JSON for later analysis."""
    with history_lock:
        data = list(signal_history)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from flask import Response
    return Response(
        pd.io.json.dumps(data, indent=2) if hasattr(pd.io, 'json') else __import__('json').dumps(data, indent=2, default=str),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=gold_signals_{timestamp}.json"}
    )


@app.route("/api/signals")
@login_required
def api_signals():
    """Query signals from the database with filters."""
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    signal_filter = request.args.get("signal", None)
    symbol_filter = request.args.get("symbol", None)
    date_from = request.args.get("from", None)
    date_to = request.args.get("to", None)
    signals = db_get_signals(limit=limit, offset=offset,
                             signal_filter=signal_filter,
                             date_from=date_from, date_to=date_to,
                             symbol_filter=symbol_filter)
    stats = db_get_stats()

    # Filtered counts for pagination and stats bar
    conn = get_db()
    base_where = "WHERE 1=1"
    base_params = []
    if symbol_filter:
        base_where += " AND symbol = ?"
        base_params.append(symbol_filter)
    if date_from:
        base_where += " AND time >= ?"
        base_params.append(date_from)
    if date_to:
        base_where += " AND time <= ?"
        base_params.append(date_to)

    # Per-type counts (ignoring signal_filter so we always show all 3)
    filtered_buys = conn.execute(
        f"SELECT COUNT(*) FROM signals {base_where} AND signal='BUY'", base_params
    ).fetchone()[0]
    filtered_sells = conn.execute(
        f"SELECT COUNT(*) FROM signals {base_where} AND signal='SELL'", base_params
    ).fetchone()[0]
    filtered_notrade = conn.execute(
        f"SELECT COUNT(*) FROM signals {base_where} AND signal='NO_TRADE'", base_params
    ).fetchone()[0]

    # Total matching current filters (including signal_filter) for pagination
    pag_where = base_where
    pag_params = list(base_params)
    if signal_filter:
        pag_where += " AND signal = ?"
        pag_params.append(signal_filter)
    filtered_total = conn.execute(
        f"SELECT COUNT(*) FROM signals {pag_where}", pag_params
    ).fetchone()[0]
    conn.close()

    return jsonify({
        "signals": signals,
        "stats": stats,
        "filtered_total": filtered_total,
        "filtered_buys": filtered_buys,
        "filtered_sells": filtered_sells,
        "filtered_notrade": filtered_notrade,
    })


@app.route("/api/signals/count")
@login_required
def api_signals_count():
    """Get total signal counts from the database."""
    return jsonify(db_get_stats())


@app.route("/api/symbols")
@login_required
def get_symbols():
    return jsonify(CONFIG.get("symbols", ["XAUUSD.m"]))


@app.route("/api/status")
@login_required
def get_status():
    """Quick health check."""
    scorer = get_scorer()
    connected = False
    try:
        connected = init_mt5()
    except Exception:
        pass
    try:
        stats = db_get_stats()
    except Exception:
        stats = daily_stats
    return jsonify(
        {
            "mt5_connected": connected,
            "model_loaded": scorer is not None,
            "model_error": _scorer_error,
            "signals_in_memory": len(signal_history),
            "signals_in_db": stats.get("total_all_time", 0),
            "daily_stats": stats,
            "db_path": DB_PATH,
            "uptime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


# ---------------------------------------------------------------------------
# Backtest Simulator
# ---------------------------------------------------------------------------
@app.route("/simulate")
@login_required
def simulate_page():
    return render_template("simulate.html")


@app.route("/api/simulate", methods=["POST"])
@login_required
def run_simulation():
    """Run XGBoost scorer on historical data bar-by-bar."""
    import glob as glob_mod

    data = request.get_json()
    symbol = data.get("symbol", CONFIG["scoring"]["default_symbol"])
    date_from = data.get("from", "2024-04-01")
    date_to = data.get("to", "2026-03-27")

    scorer = get_scorer()
    if not scorer:
        return jsonify({"error": "Model not loaded"}), 500

    # Try to load parquet data
    data_dir = CONFIG["paths"]["data_dir"]
    try:
        parquet_files = glob_mod.glob(os.path.join(data_dir, "*.parquet"))
        if not parquet_files:
            return jsonify({"error": f"No parquet files found in {data_dir}"}), 404

        # Load and filter by date
        df = pd.read_parquet(parquet_files[0])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
        df = df.loc[date_from:date_to]

        if len(df) < 100:
            return jsonify({"error": f"Not enough data: {len(df)} bars (need 100+)"}), 400

        # Score bar-by-bar (sliding window)
        signals = []
        window = int(CONFIG["scoring"].get("bars_count", 200))
        for i in range(window, len(df)):
            window_df = df.iloc[i - window:i]
            try:
                features = scorer.calculate_features(window_df)
                result = scorer.score(features)
                entry_price = df.iloc[i]["close"]

                # Check if profitable: price moved in signal direction within 12 bars
                future_bars = df.iloc[i:i + 12]
                if len(future_bars) > 0:
                    if result["signal"] == "BUY":
                        max_price = future_bars["high"].max()
                        pnl = max_price - entry_price
                    elif result["signal"] == "SELL":
                        min_price = future_bars["low"].min()
                        pnl = entry_price - min_price
                    else:
                        pnl = 0
                else:
                    pnl = 0

                signals.append({
                    "time": str(df.index[i]),
                    "signal": result["signal"],
                    "confidence": round(result["confidence"] * 100, 1),
                    "price": round(entry_price, 2),
                    "pnl": round(pnl, 2),
                    "profitable": pnl > 0,
                })
            except Exception:
                continue

        # Calculate summary
        buy_signals = [s for s in signals if s["signal"] == "BUY"]
        sell_signals = [s for s in signals if s["signal"] == "SELL"]
        profitable = [s for s in signals if s["profitable"] and s["signal"] != "NO_TRADE"]
        total_trades = len(buy_signals) + len(sell_signals)

        summary = {
            "total_signals": len(signals),
            "buy_count": len(buy_signals),
            "sell_count": len(sell_signals),
            "no_trade_count": len(signals) - total_trades,
            "profitable_pct": round(len(profitable) / total_trades * 100, 1) if total_trades > 0 else 0,
            "avg_pnl": round(sum(s["pnl"] for s in signals if s["signal"] != "NO_TRADE") / total_trades, 2) if total_trades > 0 else 0,
        }

        # Save to DB
        conn = get_db()
        conn.execute("""
            INSERT INTO simulations (name, symbol, date_from, date_to, total_signals,
                buy_count, sell_count, profitable_pct, avg_return, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{symbol} {date_from} to {date_to}",
            symbol, date_from, date_to,
            summary["total_signals"], summary["buy_count"], summary["sell_count"],
            summary["profitable_pct"], summary["avg_pnl"],
            json.dumps(signals[-100:])
        ))
        conn.commit()
        conn.close()

        return jsonify({"summary": summary, "signals": signals[-100:], "total_processed": len(df) - window})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulations")
@login_required
def list_simulations():
    """List past simulation runs."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM simulations ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Admin — Model Management
# ---------------------------------------------------------------------------
@app.route("/admin/models")
@login_required
def admin_models():
    return render_template("admin_models.html")


@app.route("/api/admin/model-stats")
@login_required
def model_stats():
    """Return current model info and feature importance."""
    scorer = get_scorer()
    if not scorer:
        return jsonify({"error": "No model loaded"}), 500

    # Load config for metrics
    try:
        with open(CONFIG["paths"]["config_path"], "r") as f:
            model_config = json.load(f)
    except Exception:
        model_config = {}

    val_metrics = model_config.get("val_metrics", {})
    return jsonify({
        "feature_count": len(scorer.feature_names),
        "feature_names": scorer.feature_names,
        "feature_importance": model_config.get("feature_importance", {}),
        "accuracy": round(val_metrics.get("accuracy", 0) * 100, 1),
        "buy_winrate": round(val_metrics.get("buy_highconf_winrate", 0) * 100, 1),
        "sell_winrate": round(val_metrics.get("sell_highconf_winrate", 0) * 100, 1),
        "n_train": model_config.get("n_train_samples", 0),
        "n_val": model_config.get("n_val_samples", 0),
        "confidence_threshold": model_config.get("confidence_threshold", 0.6),
        "model_path": CONFIG["paths"]["model_path"],
        "training_date": model_config.get("training_date", "Unknown"),
    })


@app.route("/api/admin/models/list")
@login_required
def list_models():
    """List all saved model files."""
    import glob
    model_dir = os.path.dirname(CONFIG["paths"]["model_path"])
    models = glob.glob(os.path.join(model_dir, "gold_signal_model*.json"))
    result = []
    for m in sorted(models, reverse=True):
        stat = os.stat(m)
        result.append({
            "path": m,
            "name": os.path.basename(m),
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": os.path.abspath(m) == os.path.abspath(CONFIG["paths"]["model_path"]),
        })
    return jsonify(result)


@app.route("/api/admin/retrain", methods=["POST"])
@login_required
def retrain_model():
    """Retrain XGBoost model on data. Returns new model metrics."""
    data = request.get_json() or {}
    date_from = data.get("from", "2024-04-01")
    date_to = data.get("to", datetime.now().strftime("%Y-%m-%d"))

    try:
        # Load parquet data
        import glob as glob_mod
        data_dir = CONFIG["paths"]["data_dir"]
        parquet_files = glob_mod.glob(os.path.join(data_dir, "*.parquet"))
        if not parquet_files:
            return jsonify({"error": "No parquet data found"}), 404

        df = pd.read_parquet(parquet_files[0])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
        df = df.loc[date_from:date_to]

        if len(df) < 500:
            return jsonify({"error": f"Not enough data: {len(df)} bars (need 500+)"}), 400

        # Import scorer and retrain
        from gold_signal_scorer import GoldSignalScorer
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        # Calculate features for all bars
        features_list = []
        labels = []
        window = int(CONFIG["scoring"].get("bars_count", 200))
        atr_mult = 1.5

        for i in range(window, len(df) - 12):
            window_df = df.iloc[i - window:i]
            try:
                loaded_scorer = get_scorer()
                if loaded_scorer:
                    feats = loaded_scorer.calculate_features(window_df)
                    features_list.append(feats)

                    # Label: what happened in next 12 bars
                    future = df.iloc[i:i + 12]
                    entry = df.iloc[i]["close"]
                    atr = feats.get("atr14", 20)
                    threshold = atr * atr_mult

                    max_up = future["high"].max() - entry
                    max_down = entry - future["low"].min()

                    if max_up >= threshold:
                        labels.append(0)  # BUY
                    elif max_down >= threshold:
                        labels.append(1)  # SELL
                    else:
                        labels.append(2)  # NO_TRADE
            except Exception:
                continue

        if len(features_list) < 200:
            return jsonify({"error": f"Could only compute {len(features_list)} samples"}), 400

        feat_df = pd.DataFrame(features_list)
        X_train, X_val, y_train, y_val = train_test_split(feat_df, labels, test_size=0.2, shuffle=False)

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            use_label_encoder=False
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate
        val_preds = model.predict(X_val)
        accuracy = sum(p == a for p, a in zip(val_preds, y_val)) / len(y_val)

        # Save new model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.dirname(CONFIG["paths"]["model_path"])
        new_model_path = os.path.join(model_dir, f"gold_signal_model_{timestamp}.json")
        model.save_model(new_model_path)

        # Save new config
        new_config = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_names": list(feat_df.columns),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "val_metrics": {"accuracy": accuracy},
            "confidence_threshold": 0.6,
        }
        new_config_path = os.path.join(model_dir, f"gold_signal_config_{timestamp}.json")
        with open(new_config_path, "w") as f:
            json.dump(new_config, f, indent=2)

        return jsonify({
            "success": True,
            "model_path": new_model_path,
            "config_path": new_config_path,
            "accuracy": round(accuracy * 100, 1),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": len(feat_df.columns),
            "timestamp": timestamp,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/models/promote", methods=["POST"])
@login_required
def promote_model():
    """Switch active model to a different version."""
    global _scorer, _scorer_error
    data = request.get_json()
    model_path = data.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "Model file not found"}), 404

    # Update config to point to new model
    CONFIG["paths"]["model_path"] = model_path
    # Also update config.yaml on disk
    try:
        with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
            cfg = yaml.safe_load(f)
        cfg["paths"]["model_path"] = model_path
        with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except Exception as e:
        return jsonify({"error": f"Failed to update config: {e}"}), 500

    # Reload scorer
    _scorer = None
    _scorer_error = None
    scorer = get_scorer()
    if scorer:
        return jsonify({"success": True, "model_path": model_path, "features": len(scorer.feature_names)})
    else:
        return jsonify({"error": f"Failed to load model: {_scorer_error}"}), 500


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@app.route("/settings")
@login_required
def settings_page():
    return render_template("settings.html")


@app.route("/api/settings", methods=["GET"])
@login_required
def get_settings():
    """Return current settings."""
    return jsonify({
        "scoring": {
            "confidence_threshold": CONFIG.get("scoring", {}).get("confidence_threshold", 80),
            "default_symbol": CONFIG.get("scoring", {}).get("default_symbol", "XAUUSD.m"),
            "timeframe": CONFIG.get("scoring", {}).get("timeframe", "M30"),
            "bars_count": CONFIG.get("scoring", {}).get("bars_count", 200),
        },
        "alerts": {
            "enabled": CONFIG.get("alerts", {}).get("enabled", False),
            "min_confidence": CONFIG.get("alerts", {}).get("min_confidence", 90),
        },
        "symbols": CONFIG.get("symbols", ["XAUUSD.m"]),
        "server": {
            "host": CONFIG.get("server", {}).get("host", "0.0.0.0"),
            "port": CONFIG.get("server", {}).get("port", 5000),
        },
        "trade_manager": CONFIG.get("trade_manager", {}),
    })


@app.route("/api/settings", methods=["POST"])
@login_required
def save_settings():
    """Save signal and alert settings to config.yaml and update CONFIG in memory."""
    data = request.get_json()

    # Update in-memory CONFIG
    if "scoring" in data:
        for key, val in data["scoring"].items():
            CONFIG.setdefault("scoring", {})[key] = val
    if "alerts" in data:
        for key, val in data["alerts"].items():
            CONFIG.setdefault("alerts", {})[key] = val
    if "trade_manager" in data:
        for key, val in data["trade_manager"].items():
            CONFIG.setdefault("trade_manager", {})[key] = val

    # Write to config.yaml
    try:
        config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
        return jsonify({"success": True, "message": "Settings saved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/settings/account", methods=["POST"])
@login_required
def save_account_settings():
    """Save account settings to .env and update environment."""
    data = request.get_json()
    env_path = os.path.join(os.path.dirname(__file__), ".env")

    # Read current .env
    env_lines = {}
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    env_lines[key.strip()] = val.strip()
    except FileNotFoundError:
        pass

    # Update values
    if data.get("username"):
        env_lines["DASHBOARD_USER"] = data["username"]
        os.environ["DASHBOARD_USER"] = data["username"]
    if data.get("password"):
        env_lines["DASHBOARD_PASS"] = data["password"]
        os.environ["DASHBOARD_PASS"] = data["password"]
    if data.get("api_key"):
        env_lines["DASHBOARD_API_KEY"] = data["api_key"]
        os.environ["DASHBOARD_API_KEY"] = data["api_key"]
    if "telegram_bot_token" in data:
        env_lines["TELEGRAM_BOT_TOKEN"] = data["telegram_bot_token"]
        os.environ["TELEGRAM_BOT_TOKEN"] = data["telegram_bot_token"]
    if "telegram_chat_id" in data:
        env_lines["TELEGRAM_CHAT_ID"] = data["telegram_chat_id"]
        os.environ["TELEGRAM_CHAT_ID"] = data["telegram_chat_id"]

    # Write back to .env
    try:
        with open(env_path, "w") as f:
            for key, val in env_lines.items():
                f.write(f"{key}={val}\n")
        return jsonify({"success": True, "message": "Account settings saved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Trade Manager Routes
# ---------------------------------------------------------------------------
@app.route("/api/trade/active")
@login_required
def api_active_trade():
    """Get current active trade."""
    trade = get_active_trade()
    return jsonify({"trade": trade})


@app.route("/api/trade/close", methods=["POST"])
@login_required
def api_close_trade():
    """Manually close the active trade."""
    conn = get_db()
    trade = conn.execute("SELECT * FROM active_trades WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()
    if not trade:
        conn.close()
        return jsonify({"error": "No active trade"}), 404
    trade_dict = dict(trade)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE active_trades SET status='closed', exit_reason='manual', closed_at=? WHERE id=?",
                 (now, trade_dict["id"]))
    conn.commit()
    conn.close()
    socketio.emit("trade_closed", {"trade_id": trade_dict["id"], "exit_reason": "manual"})
    return jsonify({"success": True, "trade_id": trade_dict["id"]})


@app.route("/api/trade/history")
@login_required
def api_trade_history():
    """Get closed trades history."""
    conn = get_db()
    limit = request.args.get("limit", 50, type=int)
    trades = conn.execute("SELECT * FROM active_trades WHERE status='closed' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return jsonify({"trades": [dict(t) for t in trades]})


# ---------------------------------------------------------------------------
# Background WebSocket signal checker
# ---------------------------------------------------------------------------
def background_signal_checker():
    """Check for new M30 bars, save to DB, send alerts, and push via WebSocket."""
    global _last_saved_bar_time
    while True:
        try:
            scorer = get_scorer()
            if scorer:
                symbol = CONFIG["scoring"]["default_symbol"]
                df, tick_info, account_info = get_mt5_data(symbol)
                current_bar = str(df.index[-1])
                if current_bar != _last_saved_bar_time:
                    _last_saved_bar_time = current_bar
                    features = scorer.calculate_features(df)
                    result = scorer.score(features)

                    signal_record = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "price": tick_info["ask"] if tick_info else 0,
                        "bid": tick_info["bid"] if tick_info else 0,
                        "signal": result["signal"],
                        "confidence": round(result["confidence"] * 100, 1),
                        "score": result["score"],
                        "buy_prob": round(result.get("buy_prob", 0) * 100, 1),
                        "sell_prob": round(result.get("sell_prob", 0) * 100, 1),
                        "no_trade_prob": round(result.get("no_trade_prob", 0) * 100, 1),
                        "rsi": round(features.get("rsi14", 0), 1),
                        "stoch_k": round(features.get("stoch_k", 0), 1),
                        "atr": round(features.get("atr14", 0), 2),
                        "atr_vs_avg": round(features.get("atr_vs_avg", 0), 2),
                        "adx": round(features.get("adx", 0), 1),
                        "bb_position": round(features.get("bb_position", 0), 3),
                        "ema_trend": "BULLISH" if features.get("ema_trend", 0) == 1 else "BEARISH",
                        "macd": round(features.get("macd", 0), 2),
                        "body_ratio": round(features.get("body_ratio", 0), 2),
                        "is_bullish": bool(features.get("is_bullish", 0)),
                        "reasons": result.get("top_reasons", []),
                        "symbol": symbol,
                    }

                    # Save to DB
                    try:
                        db_insert_signal(signal_record)
                    except Exception as db_err:
                        print(f"[WARN] BG DB insert failed: {db_err}")

                    # Telegram alert
                    send_telegram_alert(signal_record)

                    # Keep in memory
                    with history_lock:
                        signal_history.append(signal_record)
                        if len(signal_history) > 500:
                            signal_history.pop(0)

                    # Push to WebSocket clients
                    socketio.emit("signal_update", signal_record)

                    # Open trade if signal qualifies
                    trade_data = open_trade(signal_record)
                    if trade_data:
                        socketio.emit("trade_opened", trade_data)
                        send_trade_telegram(trade_data, "opened")

                # Check active trades every cycle (not just on new bar)
                if tick_info:
                    closed_trades = check_active_trades(tick_info["ask"], tick_info["bid"])
                    for ct in closed_trades:
                        socketio.emit("trade_closed", ct)
                        send_trade_telegram(ct, "closed")

                    # Send live trade update to clients
                    active = get_active_trade()
                    if active:
                        active["current_ask"] = tick_info["ask"]
                        active["current_bid"] = tick_info["bid"]
                        if active["direction"] == "BUY":
                            active["current_pnl"] = round(tick_info["bid"] - active["entry_price"], 2)
                        else:
                            active["current_pnl"] = round(active["entry_price"] - tick_info["ask"], 2)
                        socketio.emit("trade_update", active)
        except Exception as e:
            print(f"[WARN] Background checker: {e}")
        socketio.sleep(10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  GOLD AI SIGNAL SCORER — Dashboard")
    print("=" * 60)
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  DB:     {DB_PATH}")

    # Pre-load scorer
    scorer = get_scorer()
    if scorer:
        print(f"  Model:  OK ({len(scorer.feature_names)} features)")
    else:
        print(f"  Model:  FAILED — {_scorer_error}")

    # Check MT5
    try:
        connected = init_mt5()
        print(f"  MT5:    {'Connected' if connected else 'Offline'}")
    except Exception as e:
        print(f"  MT5:    Offline ({e})")

    print()
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)

    socketio.start_background_task(background_signal_checker)
    socketio.run(app, host=CONFIG["server"]["host"], port=CONFIG["server"]["port"], debug=False, allow_unsafe_werkzeug=True)

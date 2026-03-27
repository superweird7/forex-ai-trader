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
from flask import Flask, render_template, jsonify

# Load environment and config
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Add project paths
sys.path.insert(0, CONFIG["paths"]["python_dir"])

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.auto_reload = True
app.jinja_env.bytecode_cache = None

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
            adx, bb_position, ema_trend, macd, body_ratio, is_bullish, reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ))
    conn.commit()
    conn.close()


def db_get_signals(limit=20, offset=0, signal_filter=None, date_from=None, date_to=None):
    """Query signals from the database with optional filters."""
    conn = get_db()
    query = "SELECT * FROM signals WHERE 1=1"
    params = []
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


def get_mt5_data():
    """
    Fetch live M30 bars and tick from MT5.
    Returns (df, tick_info, account_info) or raises.
    """
    import MetaTrader5 as mt5

    if not mt5.initialize():
        mt5.initialize(CONFIG["paths"]["mt5_terminal"])

    info = mt5.terminal_info()
    if info is None:
        raise ConnectionError("MT5 terminal not responding")

    symbol = CONFIG["scoring"]["default_symbol"]
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
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/signal")
def get_signal():
    """Main endpoint: fetch live data, score, return everything."""
    reset_daily_stats_if_needed()

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
        df, tick_info, account_info = get_mt5_data()
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

    return jsonify(payload)


@app.route("/api/history")
def get_history():
    with history_lock:
        return jsonify(signal_history[-20:])


@app.route("/api/export")
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
def api_signals():
    """Query signals from the database with filters."""
    from flask import request
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    signal_filter = request.args.get("signal", None)
    date_from = request.args.get("from", None)
    date_to = request.args.get("to", None)
    signals = db_get_signals(limit=limit, offset=offset,
                             signal_filter=signal_filter,
                             date_from=date_from, date_to=date_to)
    stats = db_get_stats()
    return jsonify({"signals": signals, "stats": stats})


@app.route("/api/signals/count")
def api_signals_count():
    """Get total signal counts from the database."""
    return jsonify(db_get_stats())


@app.route("/api/status")
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

    app.run(host=CONFIG["server"]["host"], port=CONFIG["server"]["port"], debug=False)

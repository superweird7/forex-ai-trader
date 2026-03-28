"""Walk-forward backtest V2 — Direction-only exits + wider stops."""
import sys, json
sys.path.insert(0, "D:/FOREX/python")
import numpy as np
import pandas as pd
import xgboost as xgb
from gold_signal_scorer import GoldSignalScorer

# Load data
df = pd.read_parquet("D:/FOREX/data/xauusd_m30_analysis.parquet")
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

# Load V2 model
v2_model = xgb.XGBClassifier()
v2_model.load_model("D:/FOREX/models/gold_signal_model_20260328_021410.json")
with open("D:/FOREX/models/gold_signal_config_20260328_021410.json") as f:
    v2_features = json.load(f)["feature_names"]

# Features
scorer = GoldSignalScorer(
    model_path="D:/FOREX/models/gold_signal_model.json",
    config_path="D:/FOREX/models/gold_signal_config.json",
)
df = scorer._ensure_indicators(df)
feat = scorer._engineer_all_features(df)
feat["rsi_change_3"] = feat["rsi14"].diff(3)
feat["rsi_change_6"] = feat["rsi14"].diff(6)
feat["macd_histogram"] = feat["macd"] - feat["macd_signal"]
feat["macd_hist_change"] = feat["macd_histogram"].diff(3)
feat["momentum_3"] = df["close"].diff(3)
feat["momentum_6"] = df["close"].diff(6)
feat["momentum_12"] = df["close"].diff(12)
feat["momentum_accel"] = feat["momentum_6"].diff(3)
feat["session_position"] = np.where(
    (df["high"].rolling(16).max() - df["low"].rolling(16).min()) > 0,
    (df["close"] - df["low"].rolling(16).min()) /
    (df["high"].rolling(16).max() - df["low"].rolling(16).min()), 0.5)
feat["atr_change"] = feat["atr14"].diff(5)
feat["ema_gap_change"] = feat["ema_gap_pct"].diff(3)
feat = feat.fillna(0).replace([np.inf, -np.inf], 0)

test_idx = feat.index[feat.index >= "2026-01-01"]
SPREAD = 30


def run_backtest(mode, sl_mult=None, tp_mult=None, max_bars=None):
    """Run backtest with given strategy."""
    trades = []
    in_trade = False
    t_dir = t_entry = t_atr = t_bar = None

    for i in range(len(test_idx)):
        idx = test_idx[i]
        bar = df.loc[idx]
        f = feat.loc[idx]
        utc_h = f.get("utc_hour", 12)

        # Check existing trade
        if in_trade:
            atr = t_atr

            # Check SL/TP if using them
            hit_tp = False
            hit_sl = False
            if sl_mult and tp_mult:
                sl_d = atr * sl_mult
                tp_d = atr * tp_mult
                if t_dir == "BUY":
                    hit_tp = bar["high"] >= t_entry + tp_d
                    hit_sl = bar["low"] <= t_entry - sl_d
                else:
                    hit_tp = bar["low"] <= t_entry - tp_d
                    hit_sl = bar["high"] >= t_entry + sl_d

                if hit_tp:
                    trades.append(dict(dir=t_dir, entry=t_entry,
                                       pnl=round(tp_d - SPREAD, 2),
                                       result="WIN", exit="TP", time=str(idx)))
                    in_trade = False
                    continue
                elif hit_sl:
                    trades.append(dict(dir=t_dir, entry=t_entry,
                                       pnl=round(-sl_d - SPREAD, 2),
                                       result="LOSS", exit="SL", time=str(idx)))
                    in_trade = False
                    continue

            # Max bars hold
            if max_bars and (i - t_bar) >= max_bars:
                p = bar["close"] - t_entry if t_dir == "BUY" else t_entry - bar["close"]
                pnl = round(p - SPREAD, 2)
                trades.append(dict(dir=t_dir, entry=t_entry, pnl=pnl,
                                   result="WIN" if pnl > 0 else "LOSS",
                                   exit="EXPIRED", time=str(idx)))
                in_trade = False
                continue

        # Session filter
        if utc_h < 7 or utc_h >= 21:
            continue

        # Score
        row = feat.loc[[idx]][v2_features]
        pred = v2_model.predict(row)[0]
        proba = v2_model.predict_proba(row)[0]
        conf = proba.max() * 100
        sig = ["BUY", "SELL", "NO_TRADE"][pred]

        if conf < 80:
            continue

        # Direction reversal = close trade
        if in_trade and sig in ("BUY", "SELL") and sig != t_dir:
            p = bar["close"] - t_entry if t_dir == "BUY" else t_entry - bar["close"]
            pnl = round(p - SPREAD, 2)
            trades.append(dict(dir=t_dir, entry=t_entry, pnl=pnl,
                               result="WIN" if pnl > 0 else "LOSS",
                               exit="REVERSED", time=str(idx)))
            in_trade = False

        # Open new trade
        if not in_trade and sig in ("BUY", "SELL"):
            in_trade = True
            t_dir = sig
            t_entry = bar["close"] + (SPREAD if sig == "BUY" else -SPREAD)
            t_atr = f["atr14"] if f["atr14"] > 0 else 20
            t_bar = i

    # Close remaining
    if in_trade:
        last = df.iloc[-1]
        p = last["close"] - t_entry if t_dir == "BUY" else t_entry - last["close"]
        pnl = round(p - SPREAD, 2)
        trades.append(dict(dir=t_dir, entry=t_entry, pnl=pnl,
                           result="WIN" if pnl > 0 else "LOSS",
                           exit="OPEN", time=str(df.index[-1])))

    return trades


def print_results(name, trades):
    if not trades:
        print(f"  {name}: NO TRADES")
        return
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pnl = sum(t["pnl"] for t in trades)
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp / gl if gl > 0 else 0

    # Max drawdown
    peak = dd = max_dd = cum = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    wr = len(wins) / len(trades) * 100

    print(f"  {name}")
    print(f"    Trades: {len(trades)} | Wins: {len(wins)} ({wr:.0f}%) | Losses: {len(losses)}")
    print(f"    P&L: {total_pnl:+.0f} pts | Avg: {total_pnl/len(trades):+.1f}/trade")
    print(f"    PF: {pf:.2f} | Max DD: {max_dd:.0f} pts")
    print(f"    Best: {max(t['pnl'] for t in trades):+.0f} | Worst: {min(t['pnl'] for t in trades):+.0f}")

    # By direction
    for d in ["BUY", "SELL"]:
        dt = [t for t in trades if t["dir"] == d]
        if dt:
            dw = sum(1 for t in dt if t["result"] == "WIN")
            dp = sum(t["pnl"] for t in dt)
            print(f"    {d}: {len(dt)} trades, {dw}W/{len(dt)-dw}L ({dw/len(dt)*100:.0f}%), PnL: {dp:+.0f}")

    # By exit
    for ex in ["TP", "SL", "EXPIRED", "REVERSED", "OPEN"]:
        et = [t for t in trades if t["exit"] == ex]
        if et:
            ew = sum(1 for t in et if t["result"] == "WIN")
            print(f"    {ex}: {len(et)} ({ew}W/{len(et)-ew}L)")

    # Equity curve (compact)
    cum = 0
    points = []
    for i, t in enumerate(trades):
        cum += t["pnl"]
        if (i + 1) % max(len(trades) // 5, 1) == 0 or i == len(trades) - 1:
            points.append(f"T{i+1}:{cum:+.0f}")
    print(f"    Equity: {' > '.join(points)}")
    print()


# ===== RUN ALL STRATEGIES =====
print("=" * 70)
print("  V2 MODEL — STRATEGY COMPARISON")
print("  Jan 1 - Mar 25, 2026 | 80%+ conf | London/NY only | 30pt spread")
print("=" * 70)
print()

# Strategy 1: Direction only (no SL/TP, hold until model reverses)
t1 = run_backtest("direction_only")
print_results("STRATEGY 1: Direction Only (hold until reversal)", t1)

# Strategy 2: Direction + 24 bar max hold
t2 = run_backtest("direction_max24", max_bars=24)
print_results("STRATEGY 2: Direction + 24 bar max (12hr)", t2)

# Strategy 3: Direction + 12 bar max hold
t3 = run_backtest("direction_max12", max_bars=12)
print_results("STRATEGY 3: Direction + 12 bar max (6hr)", t3)

# Strategy 4: Wide SL (3x ATR) + Wide TP (5x ATR)
t4 = run_backtest("wide", sl_mult=3.0, tp_mult=5.0, max_bars=24)
print_results("STRATEGY 4: Wide SL=3xATR, TP=5xATR, 24bar max", t4)

# Strategy 5: Very wide SL (4x ATR) + TP (6x ATR)
t5 = run_backtest("verywide", sl_mult=4.0, tp_mult=6.0, max_bars=48)
print_results("STRATEGY 5: Very Wide SL=4xATR, TP=6xATR, 48bar max", t5)

# Strategy 6: Catastrophe SL only (5x ATR) + direction exit
t6 = run_backtest("catastrophe_sl", sl_mult=5.0, max_bars=None)
print_results("STRATEGY 6: Emergency SL=5xATR only + direction exit", t6)

# Strategy 7: No SL, 6 bar hold (3hr scalp)
t7 = run_backtest("scalp", max_bars=6)
print_results("STRATEGY 7: Scalp — 6 bar hold (3hr), no SL", t7)

print("=" * 70)
print("  SUMMARY — BEST STRATEGY RANKING:")
print("=" * 70)
strategies = [
    ("1: Direction Only", t1),
    ("2: Dir + 24bar", t2),
    ("3: Dir + 12bar", t3),
    ("4: Wide 3x/5x 24bar", t4),
    ("5: VWide 4x/6x 48bar", t5),
    ("6: EmergSL + Dir", t6),
    ("7: Scalp 6bar", t7),
]
results = []
for name, trades in strategies:
    if not trades: continue
    pnl = sum(t["pnl"] for t in trades)
    wr = sum(1 for t in trades if t["result"] == "WIN") / len(trades) * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp / gl if gl > 0 else 0
    results.append((name, len(trades), wr, pnl, pf))

results.sort(key=lambda x: x[3], reverse=True)
print(f"  {'Strategy':<25s} {'Trades':>6} {'WR':>6} {'P&L':>10} {'PF':>6}")
print(f"  {'-'*55}")
for name, n, wr, pnl, pf in results:
    print(f"  {name:<25s} {n:>6} {wr:>5.0f}% {pnl:>+10.0f} {pf:>6.2f}")

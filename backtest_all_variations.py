"""Exhaustive strategy search — find what's profitable."""
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

v2_model = xgb.XGBClassifier()
v2_model.load_model("D:/FOREX/models/gold_signal_model_20260328_021410.json")
with open("D:/FOREX/models/gold_signal_config_20260328_021410.json") as f:
    v2_features = json.load(f)["feature_names"]

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
print(f"Test: {test_idx[0]} to {test_idx[-1]} ({len(test_idx)} bars)")

# Pre-compute predictions for speed
all_preds = {}
all_probas = {}
for i in range(len(test_idx)):
    idx = test_idx[i]
    row = feat.loc[[idx]][v2_features]
    all_preds[idx] = v2_model.predict(row)[0]
    all_probas[idx] = v2_model.predict_proba(row)[0]
print("Predictions cached.")


def run(conf_min=80, spread=30, session=(7,21), exit_mode="direction",
        sl_mult=None, tp_mult=None, max_bars=None, buy_only=False,
        sell_only=False, require_trend_align=False, min_atr=None,
        max_atr_vs_avg=None, cooldown_bars=0):
    trades = []
    in_trade = False
    t_dir = t_entry = t_atr = t_bar = None
    last_close_bar = -999

    for i in range(len(test_idx)):
        idx = test_idx[i]
        bar = df.loc[idx]
        f = feat.loc[idx]
        utc_h = f.get("utc_hour", 12)

        # Check existing trade
        if in_trade:
            atr = t_atr
            # SL/TP check
            if sl_mult:
                sl_d = atr * sl_mult
                if t_dir == "BUY" and bar["low"] <= t_entry - sl_d:
                    trades.append(dict(dir=t_dir, pnl=round(-sl_d - spread, 2), result="LOSS", exit="SL", time=str(idx)))
                    in_trade = False; last_close_bar = i; continue
                elif t_dir == "SELL" and bar["high"] >= t_entry + sl_d:
                    trades.append(dict(dir=t_dir, pnl=round(-sl_d - spread, 2), result="LOSS", exit="SL", time=str(idx)))
                    in_trade = False; last_close_bar = i; continue
            if tp_mult:
                tp_d = atr * tp_mult
                if t_dir == "BUY" and bar["high"] >= t_entry + tp_d:
                    trades.append(dict(dir=t_dir, pnl=round(tp_d - spread, 2), result="WIN", exit="TP", time=str(idx)))
                    in_trade = False; last_close_bar = i; continue
                elif t_dir == "SELL" and bar["low"] <= t_entry - tp_d:
                    trades.append(dict(dir=t_dir, pnl=round(tp_d - spread, 2), result="WIN", exit="TP", time=str(idx)))
                    in_trade = False; last_close_bar = i; continue
            # Max bars
            if max_bars and (i - t_bar) >= max_bars:
                p = bar["close"] - t_entry if t_dir == "BUY" else t_entry - bar["close"]
                pnl = round(p - spread, 2)
                trades.append(dict(dir=t_dir, pnl=pnl, result="WIN" if pnl > 0 else "LOSS", exit="EXPIRED", time=str(idx)))
                in_trade = False; last_close_bar = i; continue

        # Session
        if utc_h < session[0] or utc_h >= session[1]:
            continue

        pred = all_preds[idx]
        proba = all_probas[idx]
        conf = proba.max() * 100
        sig = ["BUY", "SELL", "NO_TRADE"][pred]

        if conf < conf_min:
            continue
        if buy_only and sig != "BUY":
            continue
        if sell_only and sig != "SELL":
            continue
        if require_trend_align:
            ema_trend = f.get("ema_trend", 0)
            if sig == "BUY" and ema_trend != 1: continue
            if sig == "SELL" and ema_trend != 0: continue
        if min_atr and f["atr14"] < min_atr:
            continue
        if max_atr_vs_avg and f.get("atr_vs_avg", 1) > max_atr_vs_avg:
            continue

        # Reversal close
        if in_trade and exit_mode == "direction" and sig in ("BUY", "SELL") and sig != t_dir:
            p = bar["close"] - t_entry if t_dir == "BUY" else t_entry - bar["close"]
            pnl = round(p - spread, 2)
            trades.append(dict(dir=t_dir, pnl=pnl, result="WIN" if pnl > 0 else "LOSS", exit="REV", time=str(idx)))
            in_trade = False; last_close_bar = i

        # Cooldown
        if i - last_close_bar < cooldown_bars:
            continue

        # Open
        if not in_trade and sig in ("BUY", "SELL"):
            in_trade = True
            t_dir = sig
            t_entry = bar["close"] + (spread if sig == "BUY" else -spread)
            t_atr = f["atr14"] if f["atr14"] > 0 else 20
            t_bar = i

    if in_trade:
        last = df.iloc[-1]
        p = last["close"] - t_entry if t_dir == "BUY" else t_entry - last["close"]
        trades.append(dict(dir=t_dir, pnl=round(p - spread, 2), result="WIN" if p > spread else "LOSS", exit="OPEN", time=str(df.index[-1])))
    return trades


def score(trades):
    if not trades or len(trades) < 3:
        return None
    pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["result"] == "WIN")
    wr = wins / len(trades) * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp / gl if gl > 0 else 0
    # Max DD
    peak = cum = max_dd = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak: peak = cum
        if peak - cum > max_dd: max_dd = peak - cum
    return dict(n=len(trades), wr=wr, pnl=pnl, pf=pf, dd=max_dd,
                avg=pnl/len(trades), best=max(t["pnl"] for t in trades),
                worst=min(t["pnl"] for t in trades))


# ========== EXHAUSTIVE SEARCH ==========
print("\n" + "=" * 80)
print("  EXHAUSTIVE STRATEGY SEARCH — FINDING WHAT WORKS")
print("=" * 80)

results = []

# Confidence thresholds
for conf in [80, 85, 90, 95, 97]:
    # Exit modes
    for exit_mode in ["direction"]:
        # With/without SL
        for sl in [None, 3.0, 5.0]:
            # With/without TP
            for tp in [None, 3.0, 5.0, 8.0]:
                if tp and not sl: continue  # TP without SL doesn't make sense
                # Max hold
                for mb in [None, 6, 12, 24, 48]:
                    # Direction filters
                    for dir_filter in ["both", "buy_only", "sell_only"]:
                        # Trend alignment
                        for trend in [False, True]:
                            # Session windows
                            for session in [(7,21), (7,16), (13,21), (10,18)]:
                                # Cooldown
                                for cd in [0, 3, 6]:
                                    # ATR filter
                                    for max_atr in [None, 1.5, 2.0]:
                                        t = run(
                                            conf_min=conf, spread=30,
                                            session=session, exit_mode=exit_mode,
                                            sl_mult=sl, tp_mult=tp, max_bars=mb,
                                            buy_only=(dir_filter=="buy_only"),
                                            sell_only=(dir_filter=="sell_only"),
                                            require_trend_align=trend,
                                            max_atr_vs_avg=max_atr,
                                            cooldown_bars=cd,
                                        )
                                        s = score(t)
                                        if s and s["n"] >= 5:
                                            label = (f"C{conf} SL{sl or '-'} TP{tp or '-'} MB{mb or '-'} "
                                                     f"{dir_filter[:4]} trnd={trend} S{session[0]}-{session[1]} "
                                                     f"cd{cd} atr<{max_atr or '-'}")
                                            results.append((label, s))

print(f"\nTested {len(results)} strategy variations.")

# Sort by P&L
results.sort(key=lambda x: x[1]["pnl"], reverse=True)

# Show top 20
print("\n" + "=" * 80)
print("  TOP 20 STRATEGIES BY P&L")
print("=" * 80)
print(f"  {'#':>2} {'Trades':>6} {'WR':>5} {'P&L':>9} {'PF':>5} {'Avg':>7} {'DD':>7} | Strategy")
print(f"  {'-'*75}")

for i, (label, s) in enumerate(results[:20]):
    marker = " <-- PROFITABLE" if s["pnl"] > 0 else ""
    print(f"  {i+1:2d} {s['n']:>6} {s['wr']:>4.0f}% {s['pnl']:>+9.0f} {s['pf']:>5.2f} {s['avg']:>+7.1f} {s['dd']:>7.0f} | {label}{marker}")

# Show top 10 by profit factor
pf_sorted = sorted([r for r in results if r[1]["n"] >= 10], key=lambda x: x[1]["pf"], reverse=True)
print("\n" + "=" * 80)
print("  TOP 10 BY PROFIT FACTOR (min 10 trades)")
print("=" * 80)
print(f"  {'#':>2} {'Trades':>6} {'WR':>5} {'P&L':>9} {'PF':>5} {'Avg':>7} {'DD':>7} | Strategy")
print(f"  {'-'*75}")
for i, (label, s) in enumerate(pf_sorted[:10]):
    marker = " <-- PROFITABLE" if s["pnl"] > 0 else ""
    print(f"  {i+1:2d} {s['n']:>6} {s['wr']:>4.0f}% {s['pnl']:>+9.0f} {s['pf']:>5.2f} {s['avg']:>+7.1f} {s['dd']:>7.0f} | {label}{marker}")

# Show top by win rate
wr_sorted = sorted([r for r in results if r[1]["n"] >= 10], key=lambda x: x[1]["wr"], reverse=True)
print("\n" + "=" * 80)
print("  TOP 10 BY WIN RATE (min 10 trades)")
print("=" * 80)
print(f"  {'#':>2} {'Trades':>6} {'WR':>5} {'P&L':>9} {'PF':>5} {'Avg':>7} {'DD':>7} | Strategy")
print(f"  {'-'*75}")
for i, (label, s) in enumerate(wr_sorted[:10]):
    marker = " <-- PROFITABLE" if s["pnl"] > 0 else ""
    print(f"  {i+1:2d} {s['n']:>6} {s['wr']:>4.0f}% {s['pnl']:>+9.0f} {s['pf']:>5.2f} {s['avg']:>+7.1f} {s['dd']:>7.0f} | {label}{marker}")

# Count profitable
profitable = [r for r in results if r[1]["pnl"] > 0]
print(f"\n  PROFITABLE STRATEGIES: {len(profitable)} out of {len(results)} ({len(profitable)/len(results)*100:.1f}%)")

if profitable:
    print("\n  ALL PROFITABLE STRATEGIES:")
    for label, s in profitable:
        print(f"    {s['n']:>4} trades | WR {s['wr']:.0f}% | PnL {s['pnl']:+.0f} | PF {s['pf']:.2f} | {label}")

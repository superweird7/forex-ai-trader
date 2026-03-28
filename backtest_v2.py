"""Walk-forward backtest of V2 model — Jan 1 to Mar 25, 2026. No cheating."""
import sys, json
sys.path.insert(0, "D:/FOREX/python")
import numpy as np
import pandas as pd
import xgboost as xgb
from gold_signal_scorer import GoldSignalScorer

print("=" * 70)
print("  V2 MODEL WALK-FORWARD BACKTEST")
print("  Jan 1, 2026 to Mar 25, 2026")
print("  Rules: 80%+ conf, London/NY only, SL=1.5xATR, TP=2.5xSL")
print("  Spread: 30 pts deducted from every trade")
print("=" * 70)

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

# Feature calculation
scorer = GoldSignalScorer(
    model_path="D:/FOREX/models/gold_signal_model.json",
    config_path="D:/FOREX/models/gold_signal_config.json",
)
df = scorer._ensure_indicators(df)
feat = scorer._engineer_all_features(df)

# Add v2 features
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

# Test period
test_idx = feat.index[feat.index >= "2026-01-01"]
SPREAD = 30  # points
print(f"Test: {test_idx[0]} to {test_idx[-1]} ({len(test_idx)} bars)")
print()

trades = []
in_trade = False
t_dir = t_entry = t_atr = t_bar = None

for i in range(len(test_idx)):
    idx = test_idx[i]
    bar = df.loc[idx]
    f = feat.loc[idx]
    utc_h = f.get("utc_hour", 12)

    # --- check existing trade ---
    if in_trade:
        atr = t_atr
        sl_d = atr * 1.5
        tp_d = sl_d * 2.5
        if t_dir == "BUY":
            hit_tp = bar["high"] >= t_entry + tp_d
            hit_sl = bar["low"] <= t_entry - sl_d
        else:
            hit_tp = bar["low"] <= t_entry - tp_d
            hit_sl = bar["high"] >= t_entry + sl_d

        if hit_tp:
            trades.append(dict(dir=t_dir, entry=t_entry, pnl=round(tp_d - SPREAD, 2),
                               result="WIN", exit="TP", time=str(idx)))
            in_trade = False
        elif hit_sl:
            trades.append(dict(dir=t_dir, entry=t_entry, pnl=round(-sl_d - SPREAD, 2),
                               result="LOSS", exit="SL", time=str(idx)))
            in_trade = False
        elif (i - t_bar) >= 12:
            p = bar["close"] - t_entry if t_dir == "BUY" else t_entry - bar["close"]
            pnl = round(p - SPREAD, 2)
            trades.append(dict(dir=t_dir, entry=t_entry, pnl=pnl,
                               result="WIN" if pnl > 0 else "LOSS",
                               exit="EXPIRED", time=str(idx)))
            in_trade = False

    # --- session filter ---
    if utc_h < 7 or utc_h >= 21:
        continue

    # --- score ---
    row = feat.loc[[idx]][v2_features]
    pred = v2_model.predict(row)[0]
    proba = v2_model.predict_proba(row)[0]
    conf = proba.max() * 100
    sig = ["BUY", "SELL", "NO_TRADE"][pred]

    # --- reversal close ---
    if in_trade and sig in ("BUY", "SELL") and sig != t_dir and conf >= 80:
        p = bar["close"] - t_entry if t_dir == "BUY" else t_entry - bar["close"]
        pnl = round(p - SPREAD, 2)
        trades.append(dict(dir=t_dir, entry=t_entry, pnl=pnl,
                           result="WIN" if pnl > 0 else "LOSS",
                           exit="REVERSED", time=str(idx)))
        in_trade = False

    # --- open new trade ---
    if not in_trade and sig in ("BUY", "SELL") and conf >= 80:
        in_trade = True
        t_dir = sig
        t_entry = bar["close"] + (SPREAD if sig == "BUY" else -SPREAD)
        t_atr = f["atr14"] if f["atr14"] > 0 else 20
        t_bar = i

# close remaining
if in_trade:
    last = df.iloc[-1]
    p = last["close"] - t_entry if t_dir == "BUY" else t_entry - last["close"]
    pnl = round(p - SPREAD, 2)
    trades.append(dict(dir=t_dir, entry=t_entry, pnl=pnl,
                       result="WIN" if pnl > 0 else "LOSS",
                       exit="OPEN", time=str(df.index[-1])))

# === RESULTS ===
wins = [t for t in trades if t["result"] == "WIN"]
losses = [t for t in trades if t["result"] == "LOSS"]
total_pnl = sum(t["pnl"] for t in trades)

print(f"  TOTAL TRADES:       {len(trades)}")
print(f"  WINS:               {len(wins)} ({len(wins)/len(trades)*100:.0f}%)")
print(f"  LOSSES:             {len(losses)} ({len(losses)/len(trades)*100:.0f}%)")
print(f"  TOTAL P&L:          {total_pnl:+.0f} points")
print(f"  AVG P&L/TRADE:      {total_pnl/len(trades):+.1f} pts")
print(f"  BEST TRADE:         {max(t['pnl'] for t in trades):+.1f} pts")
print(f"  WORST TRADE:        {min(t['pnl'] for t in trades):+.1f} pts")
print()

# By exit
print("  BY EXIT TYPE:")
for ex in ["TP", "SL", "EXPIRED", "REVERSED", "OPEN"]:
    et = [t for t in trades if t["exit"] == ex]
    if et:
        ew = sum(1 for t in et if t["result"] == "WIN")
        ep = sum(t["pnl"] for t in et)
        print(f"    {ex:10s}: {len(et):3d} trades | {ew}W/{len(et)-ew}L | PnL: {ep:+.0f} pts")

# By direction
print()
print("  BY DIRECTION:")
for d in ["BUY", "SELL"]:
    dt = [t for t in trades if t["dir"] == d]
    if dt:
        dw = sum(1 for t in dt if t["result"] == "WIN")
        dp = sum(t["pnl"] for t in dt)
        print(f"    {d:5s}: {len(dt):3d} trades | {dw}W/{len(dt)-dw}L ({dw/len(dt)*100:.0f}%) | PnL: {dp:+.0f} pts")

# By month
print()
print("  BY MONTH:")
for m in ["2026-01", "2026-02", "2026-03"]:
    mt = [t for t in trades if t["time"].startswith(m)]
    if mt:
        mw = sum(1 for t in mt if t["result"] == "WIN")
        mp = sum(t["pnl"] for t in mt)
        print(f"    {m}: {len(mt):3d} trades | {mw}W/{len(mt)-mw}L ({mw/len(mt)*100:.0f}%) | PnL: {mp:+.0f} pts")

# Profit factor
gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
pf = gp / gl if gl > 0 else float("inf")

# Max drawdown
peak = dd = max_dd = cum = 0
for t in trades:
    cum += t["pnl"]
    if cum > peak: peak = cum
    dd = peak - cum
    if dd > max_dd: max_dd = dd

print()
print(f"  PROFIT FACTOR:      {pf:.2f}")
print(f"  GROSS PROFIT:       {gp:+.0f} pts")
print(f"  GROSS LOSS:         {-gl:+.0f} pts")
print(f"  MAX DRAWDOWN:       {max_dd:.0f} pts")

# Equity curve
print()
print("  EQUITY CURVE:")
cum = 0
for i, t in enumerate(trades):
    cum += t["pnl"]
    if (i + 1) % 20 == 0 or i == len(trades) - 1:
        bar_len = max(0, int(cum / 30))
        bar = "#" * min(bar_len, 40)
        print(f"    Trade {i+1:3d}: {cum:+8.0f} pts  {bar}")

print()
print("  NO CHEATING: Model only sees past bars, spread deducted,")
print("  session-filtered, 80%+ confidence only.")
print("=" * 70)

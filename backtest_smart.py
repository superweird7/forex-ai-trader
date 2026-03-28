"""Smart strategy search — focused variations only."""
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

# Pre-compute
preds = {}
probas = {}
for idx in test_idx:
    row = feat.loc[[idx]][v2_features]
    preds[idx] = v2_model.predict(row)[0]
    probas[idx] = v2_model.predict_proba(row)[0]
print("Predictions cached.\n")


def run(conf_min=80, spread=30, session=(7,21), sl_mult=None, tp_mult=None,
        max_bars=None, buy_only=False, sell_only=False, trend_align=False,
        cooldown=0, max_atr_ratio=None, min_momentum=None):
    trades = []
    in_trade = False
    t_dir = t_entry = t_atr = t_bar = None
    last_close = -999

    for i in range(len(test_idx)):
        idx = test_idx[i]
        bar = df.loc[idx]
        f = feat.loc[idx]
        utc_h = f.get("utc_hour", 12)

        if in_trade:
            atr = t_atr
            if sl_mult:
                sd = atr * sl_mult
                if t_dir == "BUY" and bar["low"] <= t_entry - sd:
                    trades.append(dict(dir=t_dir, pnl=round(-sd-spread,2), r="L", ex="SL", t=str(idx)))
                    in_trade=False; last_close=i; continue
                if t_dir == "SELL" and bar["high"] >= t_entry + sd:
                    trades.append(dict(dir=t_dir, pnl=round(-sd-spread,2), r="L", ex="SL", t=str(idx)))
                    in_trade=False; last_close=i; continue
            if tp_mult:
                td = atr * tp_mult
                if t_dir == "BUY" and bar["high"] >= t_entry + td:
                    trades.append(dict(dir=t_dir, pnl=round(td-spread,2), r="W", ex="TP", t=str(idx)))
                    in_trade=False; last_close=i; continue
                if t_dir == "SELL" and bar["low"] <= t_entry - td:
                    trades.append(dict(dir=t_dir, pnl=round(td-spread,2), r="W", ex="TP", t=str(idx)))
                    in_trade=False; last_close=i; continue
            if max_bars and (i - t_bar) >= max_bars:
                p = bar["close"]-t_entry if t_dir=="BUY" else t_entry-bar["close"]
                trades.append(dict(dir=t_dir, pnl=round(p-spread,2), r="W" if p>spread else "L", ex="EXP", t=str(idx)))
                in_trade=False; last_close=i; continue

        if utc_h < session[0] or utc_h >= session[1]: continue

        pred = preds[idx]
        prob = probas[idx]
        conf = prob.max() * 100
        sig = ["BUY","SELL","NO_TRADE"][pred]
        if conf < conf_min: continue
        if buy_only and sig != "BUY": continue
        if sell_only and sig != "SELL": continue
        if trend_align:
            et = f.get("ema_trend", 0)
            if sig == "BUY" and et != 1: continue
            if sig == "SELL" and et != 0: continue
        if max_atr_ratio and f.get("atr_vs_avg", 1) > max_atr_ratio: continue
        if min_momentum:
            mom = f.get("momentum_6", 0)
            if sig == "BUY" and mom < min_momentum: continue
            if sig == "SELL" and mom > -min_momentum: continue

        # Reversal close
        if in_trade and sig in ("BUY","SELL") and sig != t_dir:
            p = bar["close"]-t_entry if t_dir=="BUY" else t_entry-bar["close"]
            trades.append(dict(dir=t_dir, pnl=round(p-spread,2), r="W" if p>spread else "L", ex="REV", t=str(idx)))
            in_trade=False; last_close=i

        if i - last_close < cooldown: continue

        if not in_trade and sig in ("BUY","SELL"):
            in_trade=True; t_dir=sig
            t_entry = bar["close"] + (spread if sig=="BUY" else -spread)
            t_atr = f["atr14"] if f["atr14"]>0 else 20
            t_bar = i

    if in_trade:
        last=df.iloc[-1]
        p=last["close"]-t_entry if t_dir=="BUY" else t_entry-last["close"]
        trades.append(dict(dir=t_dir, pnl=round(p-spread,2), r="W" if p>spread else "L", ex="OPEN", t=str(df.index[-1])))
    return trades


def score(trades):
    if len(trades) < 3: return None
    pnl = sum(t["pnl"] for t in trades)
    w = sum(1 for t in trades if t["r"]=="W")
    gp = sum(t["pnl"] for t in trades if t["pnl"]>0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"]<0))
    pf = gp/gl if gl>0 else 99
    peak=cum=mdd=0
    for t in trades:
        cum+=t["pnl"]
        if cum>peak: peak=cum
        if peak-cum>mdd: mdd=peak-cum
    return dict(n=len(trades), wr=w/len(trades)*100, pnl=pnl, pf=pf, dd=mdd, avg=pnl/len(trades))


# ===== FOCUSED SEARCH =====
configs = []

# Vary confidence
for conf in [80, 85, 90, 95, 97]:
    for session in [(7,21), (7,16), (13,21)]:
        for sl in [None, 5.0, 8.0, 10.0]:
            for tp in [None, 3.0, 5.0, 8.0, 10.0]:
                if tp and not sl: continue
                for mb in [None, 12, 24, 48]:
                    for trend in [False, True]:
                        for cd in [0, 6]:
                            for bo in [False, True]:
                                for so in [False, True]:
                                    if bo and so: continue
                                    for mm in [None, 5, 10]:
                                        configs.append(dict(
                                            conf_min=conf, session=session,
                                            sl_mult=sl, tp_mult=tp, max_bars=mb,
                                            trend_align=trend, cooldown=cd,
                                            buy_only=bo, sell_only=so,
                                            min_momentum=mm,
                                        ))

print(f"Testing {len(configs)} strategy combinations...")

results = []
for i, cfg in enumerate(configs):
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(configs)}...")
    t = run(**cfg)
    s = score(t)
    if s and s["n"] >= 5:
        label = (f"C{cfg['conf_min']} S{cfg['session'][0]}-{cfg['session'][1]} "
                 f"SL{cfg['sl_mult'] or '-'} TP{cfg['tp_mult'] or '-'} "
                 f"MB{cfg['max_bars'] or '-'} "
                 f"{'trnd ' if cfg['trend_align'] else ''}"
                 f"{'BUYonly ' if cfg['buy_only'] else ''}"
                 f"{'SELLonly ' if cfg['sell_only'] else ''}"
                 f"cd{cfg['cooldown']} "
                 f"{'mom>'+str(cfg['min_momentum'])+' ' if cfg['min_momentum'] else ''}")
        results.append((label.strip(), s))

print(f"\nValid results: {len(results)}")

# Sort by P&L
results.sort(key=lambda x: x[1]["pnl"], reverse=True)

profitable = [r for r in results if r[1]["pnl"] > 0]
print(f"PROFITABLE: {len(profitable)} out of {len(results)}")

print("\n" + "=" * 85)
print("  TOP 30 STRATEGIES BY P&L")
print("=" * 85)
print(f"  {'#':>2} {'N':>5} {'WR':>5} {'P&L':>9} {'PF':>5} {'Avg':>7} {'DD':>7} | Strategy")
print(f"  {'-'*80}")
for i, (label, s) in enumerate(results[:30]):
    tag = " *** PROFIT ***" if s["pnl"] > 0 else ""
    print(f"  {i+1:2d} {s['n']:>5} {s['wr']:>4.0f}% {s['pnl']:>+9.0f} {s['pf']:>5.2f} {s['avg']:>+7.1f} {s['dd']:>7.0f} | {label}{tag}")

if profitable:
    print("\n" + "=" * 85)
    print(f"  ALL {len(profitable)} PROFITABLE STRATEGIES")
    print("=" * 85)
    for label, s in profitable:
        print(f"  {s['n']:>5} trades | WR {s['wr']:.0f}% | PnL {s['pnl']:+.0f} | PF {s['pf']:.2f} | DD {s['dd']:.0f} | {label}")
else:
    print("\n  NO PROFITABLE STRATEGIES FOUND.")
    print("  The V2 model cannot profitably trade gold with 30pt spread.")
    print("  Recommendation: The model is good at DIRECTION but not TIMING.")
    print("  Use it as a FILTER for your existing EAs, not as a standalone trader.")

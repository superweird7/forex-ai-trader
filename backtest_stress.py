"""Stress test — comprehensive validation before going live."""
import sys, json
sys.path.insert(0, "D:/FOREX/python")
import numpy as np, pandas as pd, xgboost as xgb
from gold_signal_scorer import GoldSignalScorer
from collections import Counter
from datetime import datetime

print("=" * 80)
print("  COMPREHENSIVE STRESS TEST — IS THIS STRATEGY REAL?")
print("=" * 80)

# =====================================================================
# PART 1: RETRAIN ON EARLY DATA, TEST ON UNSEEN DATA
# =====================================================================
print("\n--- PART 1: OUT-OF-SAMPLE TEST ---")
print("Train: Apr 2024 - Jun 2025 | Test: Jul 2025 - Mar 2026")
print("The model will NEVER have seen the test data.\n")

df = pd.read_parquet("D:/FOREX/data/xauusd_m30_analysis.parquet")
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"]); df.set_index("time", inplace=True)

sc = GoldSignalScorer("D:/FOREX/models/gold_signal_model.json", "D:/FOREX/models/gold_signal_config.json")
df = sc._ensure_indicators(df)
feat = sc._engineer_all_features(df)
feat["rsi_change_3"]=feat["rsi14"].diff(3); feat["rsi_change_6"]=feat["rsi14"].diff(6)
feat["macd_histogram"]=feat["macd"]-feat["macd_signal"]; feat["macd_hist_change"]=feat["macd_histogram"].diff(3)
feat["momentum_3"]=df["close"].diff(3); feat["momentum_6"]=df["close"].diff(6)
feat["momentum_12"]=df["close"].diff(12); feat["momentum_accel"]=feat["momentum_6"].diff(3)
feat["session_position"]=np.where((df["high"].rolling(16).max()-df["low"].rolling(16).min())>0,
    (df["close"]-df["low"].rolling(16).min())/(df["high"].rolling(16).max()-df["low"].rolling(16).min()),0.5)
feat["atr_change"]=feat["atr14"].diff(5); feat["ema_gap_change"]=feat["ema_gap_pct"].diff(3)
feat=feat.fillna(0).replace([np.inf,-np.inf],0)
all_features = list(feat.columns)

# Generate labels for ALL data
window=200; atr_mult=1.5
labels=[]; valid_idx=[]
for i in range(window, len(df)-12):
    future=df.iloc[i:i+12]; entry=df.iloc[i]["close"]; atr=feat.iloc[i]["atr14"]
    if atr<=0 or pd.isna(atr): continue
    th=atr*atr_mult
    mu=future["high"].max()-entry; md=entry-future["low"].min()
    if mu>=th: labels.append(0)
    elif md>=th: labels.append(1)
    else: labels.append(2)
    valid_idx.append(i)

X_all=feat.iloc[valid_idx].copy(); y_all=np.array(labels)

# Session filter
smask=(X_all["utc_hour"]>=7)&(X_all["utc_hour"]<21)
X_all=X_all[smask]; y_all=y_all[smask.values]

# Split: train on pre-July 2025, test on post-July 2025
train_mask_idx = X_all.index < "2025-07-01"
train_bool = np.array(train_mask_idx)
X_train = X_all[train_bool]; y_train = y_all[train_bool]
X_test = X_all[~train_bool]; y_test = y_all[~train_bool]
print(f"Train: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
print(f"Test:  {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")

# Class weights
counts=Counter(y_train); total=len(y_train)
cw={c:total/(3*n) for c,n in counts.items()}
sw=np.array([cw[l] for l in y_train])

# Train fresh model
fresh = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0)
fresh.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_test, y_test)], verbose=False)

# Accuracy
yp=fresh.predict(X_test); acc=sum(yp==y_test)/len(y_test)
print(f"Out-of-sample accuracy: {acc*100:.1f}%")

# Cache predictions for test period
test_dates = feat.index[feat.index >= "2025-07-01"]
oos_preds={}; oos_probas={}
for idx in test_dates:
    row=feat.loc[[idx]][all_features]
    oos_preds[idx]=fresh.predict(row)[0]
    oos_probas[idx]=fresh.predict_proba(row)[0]

SPREAD=30

def run_bt(preds, probas, dates, cm=97, ss=(7,21), sl=10, tp=15, cd=6):
    trades=[]; it=False; td=te=ta=tb=None; lc=-99
    for i in range(len(dates)):
        idx=dates[i]; bar=df.loc[idx]; f=feat.loc[idx]; uh=f.get("utc_hour",12)
        if it:
            a=ta
            if sl:
                sd=a*sl
                if td=="BUY" and bar["low"]<=te-sd:
                    trades.append(dict(d=td,p=round(-sd-SPREAD,2),w=0,t=str(idx))); it=False;lc=i;continue
                if td=="SELL" and bar["high"]>=te+sd:
                    trades.append(dict(d=td,p=round(-sd-SPREAD,2),w=0,t=str(idx))); it=False;lc=i;continue
            if tp:
                tpd=a*tp
                if td=="BUY" and bar["high"]>=te+tpd:
                    trades.append(dict(d=td,p=round(tpd-SPREAD,2),w=1,t=str(idx))); it=False;lc=i;continue
                if td=="SELL" and bar["low"]<=te-tpd:
                    trades.append(dict(d=td,p=round(tpd-SPREAD,2),w=1,t=str(idx))); it=False;lc=i;continue
        if uh<ss[0] or uh>=ss[1]: continue
        if idx not in preds: continue
        pr=preds[idx]; pb=probas[idx]; c=pb.max()*100; s=["BUY","SELL","NO_TRADE"][pr]
        if c<cm: continue
        if it and s in("BUY","SELL") and s!=td:
            pp=bar["close"]-te if td=="BUY" else te-bar["close"]
            trades.append(dict(d=td,p=round(pp-SPREAD,2),w=1 if pp>SPREAD else 0,t=str(idx))); it=False;lc=i
        if i-lc<cd: continue
        if not it and s in("BUY","SELL"):
            it=True;td=s;te=bar["close"]+(SPREAD if s=="BUY" else -SPREAD);ta=f["atr14"] if f["atr14"]>0 else 20;tb=i
    if it:
        l=df.iloc[-1]; pp=l["close"]-te if td=="BUY" else te-l["close"]
        trades.append(dict(d=td,p=round(pp-SPREAD,2),w=1 if pp>SPREAD else 0,t=str(df.index[-1])))
    return trades

def summarize(trades, label=""):
    if len(trades)<1: print(f"  {label}: NO TRADES"); return
    pnl=sum(t["p"] for t in trades); w=sum(t["w"] for t in trades)
    gp=sum(t["p"] for t in trades if t["p"]>0); gl=abs(sum(t["p"] for t in trades if t["p"]<0))
    pf=gp/gl if gl>0 else 99
    pk=cu=md=0; streak=0; max_streak=0
    for t in trades:
        cu+=t["p"]
        if cu>pk: pk=cu
        if pk-cu>md: md=pk-cu
        if t["w"]==0: streak+=1; max_streak=max(max_streak,streak)
        else: streak=0
    wr=w/len(trades)*100
    print(f"  {label}: {len(trades)} trades | WR {wr:.0f}% | PnL {pnl:+.0f} | PF {pf:.2f} | DD {md:.0f} | MaxLoseStreak {max_streak}")
    return dict(n=len(trades),wr=wr,pnl=pnl,pf=pf,dd=md,ms=max_streak)

# Run on out-of-sample data with winning strategy
print("\n--- OUT-OF-SAMPLE RESULTS (model never saw this data) ---")
oos_trades = run_bt(oos_preds, oos_probas, test_dates, cm=97, sl=10, tp=15, cd=6)
oos_r = summarize(oos_trades, "OOS C97 SL10 TP15 cd6")

# Also test nearby params on OOS
print("\n--- PARAMETER SENSITIVITY (out-of-sample) ---")
for cm in [95, 96, 97, 98]:
    for sl in [8, 10, 12]:
        for tp in [12, 15, 18]:
            for cd in [4, 6, 8]:
                t = run_bt(oos_preds, oos_probas, test_dates, cm=cm, sl=sl, tp=tp, cd=cd)
                if len(t) >= 3:
                    pnl=sum(x["p"] for x in t); w=sum(x["w"] for x in t)
                    wr=w/len(t)*100
                    gp=sum(x["p"] for x in t if x["p"]>0); gl=abs(sum(x["p"] for x in t if x["p"]<0))
                    pf=gp/gl if gl>0 else 99
                    tag=" ***" if pnl>0 else ""
                    if pnl > 500 or pnl == max(pnl, 0):  # only show profitable or top
                        print(f"    C{cm} SL{sl} TP{tp} cd{cd}: {len(t)} tr | WR {wr:.0f}% | PnL {pnl:+.0f} | PF {pf:.2f}{tag}")

# =====================================================================
# PART 2: PERIOD-BY-PERIOD (using the already-trained V2 model)
# =====================================================================
print("\n\n--- PART 2: PERIOD-BY-PERIOD (V2 model) ---")

v2 = xgb.XGBClassifier(); v2.load_model("D:/FOREX/models/gold_signal_model_20260328_021410.json")
with open("D:/FOREX/models/gold_signal_config_20260328_021410.json") as f: v2f=json.load(f)["feature_names"]

# Cache V2 predictions
all_dates = feat.index[feat.index >= "2024-07-01"]
v2_preds={}; v2_probas={}
for idx in all_dates:
    row=feat.loc[[idx]][v2f]
    v2_preds[idx]=v2.predict(row)[0]; v2_probas[idx]=v2.predict_proba(row)[0]

periods = [
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2025-Q1", "2025-01-01", "2025-04-01"),
    ("2025-Q2", "2025-04-01", "2025-07-01"),
    ("2025-Q3", "2025-07-01", "2025-10-01"),
    ("2025-Q4", "2025-10-01", "2026-01-01"),
    ("2026-Q1", "2026-01-01", "2026-04-01"),
]

for name, start, end in periods:
    pdates = feat.index[(feat.index >= start) & (feat.index < end)]
    if len(pdates) < 10: continue
    t = run_bt(v2_preds, v2_probas, pdates, cm=97, sl=10, tp=15, cd=6)
    summarize(t, name)

# =====================================================================
# PART 3: PARAMETER SENSITIVITY HEATMAP
# =====================================================================
print("\n\n--- PART 3: PARAMETER SENSITIVITY ---")
print("Testing if small changes break the strategy...\n")

test_2026 = feat.index[feat.index >= "2026-01-01"]
print(f"  {'Conf':>4} {'SL':>4} {'TP':>4} {'CD':>3} | {'N':>4} {'WR':>5} {'PnL':>8} {'PF':>5}")
print(f"  {'-'*50}")

sensitivity_results = []
for cm in [93, 94, 95, 96, 97, 98, 99]:
    for sl in [7, 8, 9, 10, 11, 12]:
        for tp in [10, 12, 14, 15, 16, 18]:
            for cd in [3, 4, 5, 6, 7, 8]:
                t = run_bt(v2_preds, v2_probas, test_2026, cm=cm, sl=sl, tp=tp, cd=cd)
                if len(t) >= 3:
                    pnl=sum(x["p"] for x in t); w=sum(x["w"] for x in t)
                    gp=sum(x["p"] for x in t if x["p"]>0); gl=abs(sum(x["p"] for x in t if x["p"]<0))
                    pf=gp/gl if gl>0 else 99
                    sensitivity_results.append((cm,sl,tp,cd,len(t),w/len(t)*100,pnl,pf))

total_tested = len(sensitivity_results)
profitable = [r for r in sensitivity_results if r[6] > 0]
print(f"  Tested: {total_tested} combinations")
print(f"  Profitable: {len(profitable)} ({len(profitable)/total_tested*100:.0f}%)")

# Show best and worst
sensitivity_results.sort(key=lambda x: x[6], reverse=True)
print(f"\n  TOP 5:")
for cm,sl,tp,cd,n,wr,pnl,pf in sensitivity_results[:5]:
    print(f"  C{cm:>2} SL{sl:>2} TP{tp:>2} cd{cd} | {n:>4} {wr:>4.0f}% {pnl:>+8.0f} {pf:>5.2f}")
print(f"\n  BOTTOM 5:")
for cm,sl,tp,cd,n,wr,pnl,pf in sensitivity_results[-5:]:
    print(f"  C{cm:>2} SL{sl:>2} TP{tp:>2} cd{cd} | {n:>4} {wr:>4.0f}% {pnl:>+8.0f} {pf:>5.2f}")

# Average PnL by each parameter
print("\n  AVG P&L BY CONFIDENCE:")
for cm in [93,94,95,96,97,98,99]:
    subset = [r for r in sensitivity_results if r[0]==cm]
    if subset:
        avg_pnl = np.mean([r[6] for r in subset])
        pct_prof = sum(1 for r in subset if r[6]>0)/len(subset)*100
        print(f"    C{cm}: avg PnL {avg_pnl:+.0f} | {pct_prof:.0f}% profitable")

print("\n  AVG P&L BY SL MULTIPLIER:")
for sl in [7,8,9,10,11,12]:
    subset = [r for r in sensitivity_results if r[1]==sl]
    if subset:
        avg_pnl = np.mean([r[6] for r in subset])
        pct_prof = sum(1 for r in subset if r[6]>0)/len(subset)*100
        print(f"    SL{sl}x: avg PnL {avg_pnl:+.0f} | {pct_prof:.0f}% profitable")

# =====================================================================
# PART 4: WORST CASE ANALYSIS
# =====================================================================
print("\n\n--- PART 4: WORST CASE ANALYSIS ---")
best_trades = run_bt(v2_preds, v2_probas, test_2026, cm=97, sl=10, tp=15, cd=6)
if best_trades:
    # Max consecutive losses
    streak=0; max_streak=0
    for t in best_trades:
        if t["w"]==0: streak+=1; max_streak=max(max_streak,streak)
        else: streak=0
    print(f"  Max consecutive losses: {max_streak}")

    # Worst week
    from collections import defaultdict
    weekly = defaultdict(float)
    for t in best_trades:
        week = t["t"][:10]  # date
        weekly[week] += t["p"]
    worst_day = min(weekly.items(), key=lambda x: x[1])
    best_day = max(weekly.items(), key=lambda x: x[1])
    print(f"  Worst day: {worst_day[0]} ({worst_day[1]:+.0f} pts)")
    print(f"  Best day:  {best_day[0]} ({best_day[1]:+.0f} pts)")

    # Monthly breakdown
    monthly = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for t in best_trades:
        m = t["t"][:7]
        monthly[m]["pnl"] += t["p"]
        monthly[m]["n"] += 1
        monthly[m]["w"] += t["w"]
    print(f"\n  MONTHLY BREAKDOWN:")
    all_profitable_months = True
    for m in sorted(monthly.keys()):
        d = monthly[m]
        wr = d["w"]/d["n"]*100 if d["n"]>0 else 0
        tag = " <-- LOSS" if d["pnl"] < 0 else ""
        if d["pnl"] < 0: all_profitable_months = False
        print(f"    {m}: {d['n']:>3} trades | WR {wr:.0f}% | PnL {d['pnl']:+.0f}{tag}")

    print(f"\n  All months profitable: {'YES' if all_profitable_months else 'NO'}")

# =====================================================================
# FINAL VERDICT
# =====================================================================
print("\n" + "=" * 80)
print("  FINAL VERDICT")
print("=" * 80)
if oos_r and oos_r["pnl"] > 0:
    print("  OUT-OF-SAMPLE: PROFITABLE")
else:
    print("  OUT-OF-SAMPLE: NOT PROFITABLE — strategy may be overfitted")
print(f"  PARAMETER SENSITIVITY: {len(profitable)}/{total_tested} ({len(profitable)/total_tested*100:.0f}%) profitable")
if len(profitable)/total_tested > 0.5:
    print("  ROBUST: >50% of nearby params are also profitable")
else:
    print("  FRAGILE: <50% of nearby params work — possible overfitting")
print("=" * 80)

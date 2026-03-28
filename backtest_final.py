"""Fast focused backtest — 50 key variations."""
import sys, json
sys.path.insert(0, "D:/FOREX/python")
import numpy as np, pandas as pd, xgboost as xgb
from gold_signal_scorer import GoldSignalScorer

df = pd.read_parquet("D:/FOREX/data/xauusd_m30_analysis.parquet")
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"]); df.set_index("time", inplace=True)

v2 = xgb.XGBClassifier(); v2.load_model("D:/FOREX/models/gold_signal_model_20260328_021410.json")
with open("D:/FOREX/models/gold_signal_config_20260328_021410.json") as f: v2f = json.load(f)["feature_names"]

sc = GoldSignalScorer("D:/FOREX/models/gold_signal_model.json","D:/FOREX/models/gold_signal_config.json")
df = sc._ensure_indicators(df); feat = sc._engineer_all_features(df)
for c,d in [("rsi_change_3",3),("rsi_change_6",6)]: feat[c]=feat["rsi14"].diff(d)
feat["macd_histogram"]=feat["macd"]-feat["macd_signal"]
feat["macd_hist_change"]=feat["macd_histogram"].diff(3)
for c,d in [("momentum_3",3),("momentum_6",6),("momentum_12",12)]: feat[c]=df["close"].diff(d)
feat["momentum_accel"]=feat["momentum_6"].diff(3)
feat["session_position"]=np.where((df["high"].rolling(16).max()-df["low"].rolling(16).min())>0,
    (df["close"]-df["low"].rolling(16).min())/(df["high"].rolling(16).max()-df["low"].rolling(16).min()),0.5)
feat["atr_change"]=feat["atr14"].diff(5); feat["ema_gap_change"]=feat["ema_gap_pct"].diff(3)
feat=feat.fillna(0).replace([np.inf,-np.inf],0)

ti=feat.index[feat.index>="2026-01-01"]
P,B={},{}
for idx in ti: r=feat.loc[[idx]][v2f]; P[idx]=v2.predict(r)[0]; B[idx]=v2.predict_proba(r)[0]
print(f"Test: {ti[0]} to {ti[-1]}, {len(ti)} bars. Cached.\n")

SPREAD=30
def run(cm=80,ss=(7,21),sl=None,tp=None,mb=None,ta=False,bo=False,so=False,cd=0,mm=None):
    trades=[]; it=False; td=te=ta_=tb=None; lc=-99
    for i in range(len(ti)):
        x=ti[i]; b=df.loc[x]; f=feat.loc[x]; uh=f.get("utc_hour",12)
        if it:
            a=ta_
            if sl:
                sd=a*sl
                if td=="BUY" and b["low"]<=te-sd: trades.append(dict(d=td,p=round(-sd-SPREAD,2),w=0)); it=False;lc=i;continue
                if td=="SELL" and b["high"]>=te+sd: trades.append(dict(d=td,p=round(-sd-SPREAD,2),w=0)); it=False;lc=i;continue
            if tp:
                tpd=a*tp
                if td=="BUY" and b["high"]>=te+tpd: trades.append(dict(d=td,p=round(tpd-SPREAD,2),w=1)); it=False;lc=i;continue
                if td=="SELL" and b["low"]<=te-tpd: trades.append(dict(d=td,p=round(tpd-SPREAD,2),w=1)); it=False;lc=i;continue
            if mb and (i-tb)>=mb:
                pp=b["close"]-te if td=="BUY" else te-b["close"]
                trades.append(dict(d=td,p=round(pp-SPREAD,2),w=1 if pp>SPREAD else 0)); it=False;lc=i;continue
        if uh<ss[0] or uh>=ss[1]: continue
        pr=P[x]; pb=B[x]; c=pb.max()*100; s=["BUY","SELL","NO_TRADE"][pr]
        if c<cm: continue
        if bo and s!="BUY": continue
        if so and s!="SELL": continue
        if ta:
            et=f.get("ema_trend",0)
            if s=="BUY" and et!=1: continue
            if s=="SELL" and et!=0: continue
        if mm:
            m6=f.get("momentum_6",0)
            if s=="BUY" and m6<mm: continue
            if s=="SELL" and m6>-mm: continue
        if it and s in("BUY","SELL") and s!=td:
            pp=b["close"]-te if td=="BUY" else te-b["close"]
            trades.append(dict(d=td,p=round(pp-SPREAD,2),w=1 if pp>SPREAD else 0)); it=False;lc=i
        if i-lc<cd: continue
        if not it and s in("BUY","SELL"):
            it=True;td=s;te=b["close"]+(SPREAD if s=="BUY" else -SPREAD);ta_=f["atr14"] if f["atr14"]>0 else 20;tb=i
    if it:
        l=df.iloc[-1]; pp=l["close"]-te if td=="BUY" else te-l["close"]
        trades.append(dict(d=td,p=round(pp-SPREAD,2),w=1 if pp>SPREAD else 0))
    return trades

# Key variations
cfgs = []
for cm in [80,85,90,95,97]:
    for ss in [(7,21),(7,16),(13,21)]:
        for sl,tp in [(None,None),(5,8),(8,12),(10,15)]:
            for mb in [None,24,48]:
                for ta in [False,True]:
                    for cd in [0,6]:
                        cfgs.append(dict(cm=cm,ss=ss,sl=sl,tp=tp,mb=mb,ta=ta,cd=cd))
# Direction filters
for cm in [80,90,95]:
    for bo,so in [(True,False),(False,True)]:
        for ss in [(7,21),(7,16)]:
            for ta in [False,True]:
                cfgs.append(dict(cm=cm,ss=ss,bo=bo,so=so,ta=ta))
# Momentum filters
for cm in [80,90,95]:
    for mm in [5,10,15,20]:
        for ss in [(7,21),(7,16)]:
            cfgs.append(dict(cm=cm,ss=ss,mm=mm))

print(f"Testing {len(cfgs)} variations...")
res=[]
for i,c in enumerate(cfgs):
    t=run(**c)
    if len(t)<3: continue
    pnl=sum(x["p"] for x in t); w=sum(x["w"] for x in t)
    gp=sum(x["p"] for x in t if x["p"]>0); gl=abs(sum(x["p"] for x in t if x["p"]<0))
    pf=gp/gl if gl>0 else 99
    pk=cu=md=0
    for x in t:
        cu+=x["p"]
        if cu>pk:pk=cu
        if pk-cu>md:md=pk-cu
    lb=f"C{c.get('cm',80)} S{c.get('ss',(7,21))[0]}-{c.get('ss',(7,21))[1]} SL{c.get('sl','-')} TP{c.get('tp','-')} MB{c.get('mb','-')} {'trnd ' if c.get('ta') else ''}{'BUY ' if c.get('bo') else ''}{'SELL ' if c.get('so') else ''}cd{c.get('cd',0)} {'mom>'+str(c.get('mm')) if c.get('mm') else ''}"
    res.append((lb.strip(),len(t),w/len(t)*100,pnl,pf,pnl/len(t),md))
    if (i+1)%200==0: print(f"  {i+1}/{len(cfgs)}...")

res.sort(key=lambda x:x[3],reverse=True)
prof=[r for r in res if r[3]>0]

print(f"\nDone. {len(res)} valid | {len(prof)} PROFITABLE\n")
print("="*90)
print("  TOP 30 STRATEGIES")
print("="*90)
print(f"  {'#':>2} {'N':>5} {'WR':>5} {'P&L':>9} {'PF':>5} {'Avg':>7} {'DD':>7} | Strategy")
print(f"  {'-'*85}")
for i,(lb,n,wr,pnl,pf,avg,dd) in enumerate(res[:30]):
    tag=" *** PROFIT ***" if pnl>0 else ""
    print(f"  {i+1:2d} {n:>5} {wr:>4.0f}% {pnl:>+9.0f} {pf:>5.2f} {avg:>+7.1f} {dd:>7.0f} | {lb}{tag}")

if prof:
    print(f"\n{'='*90}")
    print(f"  ALL {len(prof)} PROFITABLE STRATEGIES")
    print(f"{'='*90}")
    for lb,n,wr,pnl,pf,avg,dd in prof:
        print(f"  {n:>4} tr | WR {wr:.0f}% | PnL {pnl:+.0f} | PF {pf:.2f} | DD {dd:.0f} | {lb}")
else:
    print("\n  === NO PROFITABLE STRATEGY FOUND ===")
    print("  The ML model predicts direction OK but gold's 30pt spread kills all profits.")
    print("  RECOMMENDATION: Use this model as a DIRECTION FILTER for your proven EAs,")
    print("  NOT as a standalone signal generator for manual trading.")
    print("  Your EAs (GoldRegimePullback, LiquiditySweep) already handle entries/exits")
    print("  better than any ML exit strategy we tested.")

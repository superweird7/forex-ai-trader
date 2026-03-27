"""Retrain XGBoost model with improved features."""
import sys, json, os
sys.path.insert(0, "D:/FOREX/python")
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from gold_signal_scorer import GoldSignalScorer
from datetime import datetime

print("=== RETRAINING XGBOOST MODEL ===")
print(f"Time: {datetime.now()}")

# 1. Load data
df = pd.read_parquet("D:/FOREX/data/xauusd_m30_analysis.parquet")
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
print(f"Loaded {len(df)} M30 bars: {df.index.min()} to {df.index.max()}")

# 2. Load scorer for feature calculation
scorer = GoldSignalScorer(
    model_path="D:/FOREX/models/gold_signal_model.json",
    config_path="D:/FOREX/models/gold_signal_config.json",
)

# 3. Calculate base features
df = scorer._ensure_indicators(df)
features_df = scorer._engineer_all_features(df)

# 4. ADD NEW FEATURES
print("Adding new features...")
features_df["rsi_change_3"] = features_df["rsi14"].diff(3)
features_df["rsi_change_6"] = features_df["rsi14"].diff(6)
features_df["macd_histogram"] = features_df["macd"] - features_df["macd_signal"]
features_df["macd_hist_change"] = features_df["macd_histogram"].diff(3)
features_df["momentum_3"] = df["close"].diff(3)
features_df["momentum_6"] = df["close"].diff(6)
features_df["momentum_12"] = df["close"].diff(12)
features_df["momentum_accel"] = features_df["momentum_6"].diff(3)
features_df["session_position"] = np.where(
    (df["high"].rolling(16).max() - df["low"].rolling(16).min()) > 0,
    (df["close"] - df["low"].rolling(16).min()) / (df["high"].rolling(16).max() - df["low"].rolling(16).min()),
    0.5
)
features_df["atr_change"] = features_df["atr14"].diff(5)
features_df["ema_gap_change"] = features_df["ema_gap_pct"].diff(3)
print(f"Total features: {len(features_df.columns)}")

# 5. Generate labels
print("Generating labels...")
window = 200
atr_mult = 1.5
labels = []
valid_indices = []

for i in range(window, len(df) - 12):
    future = df.iloc[i:i+12]
    entry = df.iloc[i]["close"]
    atr = features_df.iloc[i]["atr14"]
    if atr <= 0 or pd.isna(atr):
        continue
    threshold = atr * atr_mult
    max_up = future["high"].max() - entry
    max_down = entry - future["low"].min()
    if max_up >= threshold:
        labels.append(0)
    elif max_down >= threshold:
        labels.append(1)
    else:
        labels.append(2)
    valid_indices.append(i)

X = features_df.iloc[valid_indices].copy().fillna(0).replace([np.inf, -np.inf], 0)
y = np.array(labels)
print(f"Samples: {len(X)} | BUY={sum(y==0)} SELL={sum(y==1)} NO_TRADE={sum(y==2)}")

# 6. Session filter
session_mask = (X["utc_hour"] >= 7) & (X["utc_hour"] < 21)
X = X[session_mask]
y = y[session_mask.values]
print(f"After session filter: {len(X)} samples")

# 7. Time-based split
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# 8. Class weights
counts = Counter(y_train)
total = len(y_train)
class_weights = {c: total / (3 * n) for c, n in counts.items()}
sample_weights = np.array([class_weights[l] for l in y_train])
print(f"Weights: BUY={class_weights.get(0,0):.2f} SELL={class_weights.get(1,0):.2f} NT={class_weights.get(2,0):.2f}")

# 9. Train
print("\nTraining...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, use_label_encoder=False,
)
model.fit(X_train, y_train, sample_weight=sample_weights,
          eval_set=[(X_val, y_val)], verbose=False)

# 10. Evaluate
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)
accuracy = sum(y_pred == y_val) / len(y_val)
print(f"\nValidation accuracy: {accuracy*100:.1f}%")

for cls, name in [(0,"BUY"), (1,"SELL"), (2,"NO_TRADE")]:
    mask = y_val == cls
    pred_mask = y_pred == cls
    if sum(mask) > 0:
        recall = sum((y_pred == cls) & mask) / sum(mask)
        precision = sum((y_pred == cls) & mask) / max(sum(pred_mask), 1)
        print(f"  {name}: precision={precision*100:.1f}%, recall={recall*100:.1f}%, n={sum(mask)}")

for thresh in [0.6, 0.7, 0.8]:
    hc = y_proba.max(axis=1) >= thresh
    if sum(hc) > 0:
        hc_acc = sum(y_pred[hc] == y_val[hc]) / sum(hc)
        print(f"  Conf>={thresh*100:.0f}%: {hc_acc*100:.1f}% acc on {sum(hc)} signals")

# 11. Feature importance
importance = model.feature_importances_
feat_imp = dict(zip(X_train.columns, importance))
top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
print("\nTop 15 features:")
for name, imp in top_features:
    bar = "#" * int(imp * 200)
    print(f"  {name:25s} {imp:.4f} {bar}")

# 12. Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "D:/FOREX/models"
new_model_path = os.path.join(model_dir, f"gold_signal_model_{timestamp}.json")
model.save_model(new_model_path)

new_config = {
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "feature_names": list(X_train.columns),
    "n_train_samples": len(X_train),
    "n_val_samples": len(X_val),
    "val_metrics": {
        "accuracy": float(accuracy),
        "buy_precision": float(sum((y_pred==0) & (y_val==0)) / max(sum(y_pred==0), 1)),
        "sell_precision": float(sum((y_pred==1) & (y_val==1)) / max(sum(y_pred==1), 1)),
        "buy_highconf_winrate": float(accuracy),
        "sell_highconf_winrate": float(accuracy),
    },
    "confidence_threshold": 0.6,
    "feature_importance": {k: float(v) for k, v in feat_imp.items()},
}
new_config_path = os.path.join(model_dir, f"gold_signal_config_{timestamp}.json")
with open(new_config_path, "w") as f:
    json.dump(new_config, f, indent=2)

print(f"\nModel saved: {new_model_path}")
print(f"Config saved: {new_config_path}")

# 13. Compare old vs new
print("\n=== OLD vs NEW ===")
old_pred = scorer.model.predict(X_val[scorer.feature_names])
old_acc = sum(old_pred == y_val) / len(y_val)
print(f"Old: {old_acc*100:.1f}% ({len(scorer.feature_names)} features)")
print(f"New: {accuracy*100:.1f}% ({len(X_train.columns)} features)")
print(f"Improvement: {(accuracy-old_acc)*100:+.1f}%")

# Old model BUY bias check
old_buy_pct = sum(old_pred == 0) / len(old_pred) * 100
new_buy_pct = sum(y_pred == 0) / len(y_pred) * 100
print(f"Old BUY bias: {old_buy_pct:.0f}% | New BUY bias: {new_buy_pct:.0f}%")

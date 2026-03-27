"""
Gold Signal Scorer — loads trained XGBoost model and scores live M30 bars.

Standalone module that can be imported or used directly.

Usage as module:
    from gold_signal_scorer import GoldSignalScorer
    scorer = GoldSignalScorer()
    result = scorer.score_bars(bars_df)  # DataFrame of last 100+ M30 bars

Usage as script:
    python gold_signal_scorer.py  # Quick test with parquet data
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH = "D:/FOREX/models/gold_signal_model.json"
CONFIG_PATH = "D:/FOREX/models/gold_signal_config.json"

BUY = 0
SELL = 1
NO_TRADE = 2


class GoldSignalScorer:
    """Loads trained XGBoost model and scores live M30 bar data."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        config_path: str = CONFIG_PATH,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run gold_signal_trainer.py first."
            )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config not found at {config_path}. Run gold_signal_trainer.py first."
            )

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        with open(config_path, "r") as fp:
            self.config = json.load(fp)

        self.feature_names = self.config["feature_names"]
        self.feature_importance = self.config.get("feature_importance", {})
        self.confidence_threshold = self.config.get("confidence_threshold", 0.60)

        # Pre-compute sorted feature importance for explanations
        self._sorted_importance = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )

    # ------------------------------------------------------------------
    # Feature calculation (mirrors gold_signal_trainer.engineer_features)
    # ------------------------------------------------------------------
    def calculate_features(
        self, bars_m30: pd.DataFrame, bars_h1: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate all features for the latest bar.

        Parameters
        ----------
        bars_m30 : DataFrame
            At least 100 M30 bars with columns:
            open, high, low, close, tick_volume, spread, real_volume
            May also include pre-calculated indicators.
        bars_h1 : DataFrame, optional
            H1 bars (unused currently, reserved for future multi-TF features).

        Returns
        -------
        dict
            Feature name -> value for the latest bar.
        """
        df = bars_m30.copy()

        # Ensure we have enough data
        if len(df) < 60:
            raise ValueError(f"Need at least 60 M30 bars, got {len(df)}")

        # Calculate indicators if not already present
        df = self._ensure_indicators(df)

        # Build all features on full DataFrame, then extract latest
        features = self._engineer_all_features(df)

        # Get the last row as a dict
        last_row = features.iloc[-1]
        return {col: float(last_row[col]) if not pd.isna(last_row[col]) else 0.0 for col in features.columns}

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate missing indicators that the raw MT5 data won't have."""
        if "range" not in df.columns:
            df["range"] = df["high"] - df["low"]
        if "body" not in df.columns:
            df["body"] = abs(df["close"] - df["open"])
        if "body_ratio" not in df.columns:
            df["body_ratio"] = np.where(df["range"] > 0, df["body"] / df["range"], 0)
        if "is_bullish" not in df.columns:
            df["is_bullish"] = (df["close"] >= df["open"]).astype(int)
        if "upper_wick" not in df.columns:
            df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
        if "lower_wick" not in df.columns:
            df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]

        # ATR14
        if "atr14" not in df.columns:
            tr = pd.concat(
                [
                    df["high"] - df["low"],
                    abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"] - df["close"].shift(1)),
                ],
                axis=1,
            ).max(axis=1)
            df["atr14"] = tr.ewm(span=14, adjust=False).mean()

        # EMAs
        if "ema21" not in df.columns:
            df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        if "ema50" not in df.columns:
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        # RSI14
        if "rsi14" not in df.columns:
            delta = df["close"].diff()
            gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
            rs = np.where(loss > 0, gain / loss, 100)
            df["rsi14"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        if "bb_mid" not in df.columns:
            df["bb_mid"] = df["close"].rolling(20).mean()
            df["bb_std"] = df["close"].rolling(20).std()
            df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
            df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
            df["bb_width"] = np.where(
                df["bb_mid"] > 0,
                (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100,
                0,
            )

        # Returns
        if "return_1bar" not in df.columns:
            df["return_1bar"] = df["close"].diff()
        if "return_3bar" not in df.columns:
            df["return_3bar"] = df["close"].diff(3)
        if "return_6bar" not in df.columns:
            df["return_6bar"] = df["close"].diff(6)
        if "return_12bar" not in df.columns:
            df["return_12bar"] = df["close"].diff(12)

        # Time columns (from index if datetime, else from columns)
        if "utc_hour" not in df.columns:
            if hasattr(df.index, "hour"):
                # Assume server time = UTC+3 for JustMarkets
                df["server_hour"] = df.index.hour
                df["utc_hour"] = (df.index.hour - 3) % 24
                df["day_of_week"] = df.index.dayofweek
            else:
                df["utc_hour"] = 12  # fallback
                df["day_of_week"] = 2

        return df

    def _engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mirrors gold_signal_trainer.engineer_features exactly.
        Returns DataFrame with all feature columns.
        """
        f = pd.DataFrame(index=df.index)

        # PRICE ACTION (10)
        f["return_1"] = df["return_1bar"]
        f["return_3"] = df["return_3bar"]
        f["return_6"] = df["return_6bar"]
        f["return_12"] = df["return_12bar"]
        f["body_ratio"] = df["body_ratio"]
        f["upper_wick_ratio"] = np.where(
            df["range"] > 0, df["upper_wick"] / df["range"], 0
        )
        f["lower_wick_ratio"] = np.where(
            df["range"] > 0, df["lower_wick"] / df["range"], 0
        )
        f["is_bullish"] = df["is_bullish"].astype(float)
        f["bar_range_vs_atr"] = np.where(
            df["atr14"] > 0, df["range"] / df["atr14"], 0
        )
        f["close_vs_open_pct"] = (df["close"] - df["open"]) / df["open"] * 100

        # TREND (8)
        ema21_shifted = df["ema21"].shift(5)
        ema50_shifted = df["ema50"].shift(5)
        f["ema21_slope"] = np.where(
            ema21_shifted > 0,
            (df["ema21"] - ema21_shifted) / ema21_shifted * 100,
            0,
        )
        f["ema50_slope"] = np.where(
            ema50_shifted > 0,
            (df["ema50"] - ema50_shifted) / ema50_shifted * 100,
            0,
        )
        f["ema_trend"] = (df["ema21"] > df["ema50"]).astype(float)
        f["ema_gap_pct"] = np.where(
            df["ema50"] > 0, (df["ema21"] - df["ema50"]) / df["ema50"] * 100, 0
        )
        f["price_vs_ema21"] = np.where(
            df["ema21"] > 0, (df["close"] - df["ema21"]) / df["ema21"] * 100, 0
        )
        f["price_vs_ema50"] = np.where(
            df["ema50"] > 0, (df["close"] - df["ema50"]) / df["ema50"] * 100, 0
        )
        f["price_above_ema21"] = (df["close"] > df["ema21"]).astype(float)
        f["price_above_ema50"] = (df["close"] > df["ema50"]).astype(float)

        # OSCILLATORS (8)
        f["rsi14"] = df["rsi14"]
        f["rsi_zone"] = np.select(
            [df["rsi14"] < 30, df["rsi14"] < 50, df["rsi14"] < 70],
            [0.0, 1.0, 2.0],
            default=3.0,
        )

        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        f["stoch_k"] = np.where(
            (high_14 - low_14) > 0,
            (df["close"] - low_14) / (high_14 - low_14) * 100,
            50,
        )

        bb_range = df["bb_upper"] - df["bb_lower"]
        f["bb_position"] = np.where(
            bb_range > 0, (df["close"] - df["bb_lower"]) / bb_range, 0.5
        )
        f["bb_width"] = df["bb_width"]
        f["bb_width_pctile"] = (
            df["bb_width"].rolling(100, min_periods=20).rank(pct=True)
        )

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        f["macd"] = macd_line
        f["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()

        # VOLATILITY (6)
        f["atr14"] = df["atr14"]
        f["atr_normalized"] = np.where(
            df["close"] > 0, df["atr14"] / df["close"] * 100, 0
        )
        atr_avg50 = df["atr14"].rolling(50, min_periods=10).mean()
        f["atr_vs_avg"] = np.where(atr_avg50 > 0, df["atr14"] / atr_avg50, 1)
        f["range_pctile"] = df["range"].rolling(50, min_periods=10).rank(pct=True)

        range_avg50 = df["range"].rolling(50, min_periods=10).mean()
        f["is_monster_bar"] = (df["range"] > 2 * range_avg50).astype(float)

        # Daily range pct
        if hasattr(df.index, "date"):
            df_date = df.index.date
        else:
            df_date = pd.Series(range(len(df)), index=df.index)

        daily_high = df.groupby(df_date)["high"].cummax()
        daily_low = df.groupby(df_date)["low"].cummin()
        daily_range = daily_high.values - daily_low.values
        adr_50 = df["range"].rolling(50 * 16, min_periods=50).sum().rolling(50).mean()
        f["daily_range_pct"] = np.where(
            adr_50 > 0, daily_range / adr_50.values * 100, 0
        )

        # TIME (8)
        f["utc_hour"] = df["utc_hour"].astype(float)
        f["hour_sin"] = np.sin(2 * np.pi * df["utc_hour"] / 24)
        f["hour_cos"] = np.cos(2 * np.pi * df["utc_hour"] / 24)
        f["day_of_week"] = df["day_of_week"].astype(float)
        f["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        f["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        f["is_london"] = ((df["utc_hour"] >= 7) & (df["utc_hour"] < 16)).astype(float)
        f["is_ny"] = ((df["utc_hour"] >= 13) & (df["utc_hour"] < 21)).astype(float)

        # PATTERN (10)
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_bull = df["is_bullish"].shift(1)

        f["bullish_engulfing"] = (
            (df["close"] > prev_high)
            & (df["is_bullish"] == 1)
            & (prev_bull == 0)
        ).astype(float)

        f["bearish_engulfing"] = (
            (df["close"] < prev_low)
            & (df["is_bullish"] == 0)
            & (prev_bull == 1)
        ).astype(float)

        body = df["body"]
        f["pin_bar_bull"] = (
            (df["lower_wick"] > 2 * body) & (df["upper_wick"] < 0.5 * body)
        ).astype(float)
        f["pin_bar_bear"] = (
            (df["upper_wick"] > 2 * body) & (df["lower_wick"] < 0.5 * body)
        ).astype(float)

        f["inside_bar"] = (
            (df["high"] < prev_high) & (df["low"] > prev_low)
        ).astype(float)

        # Consecutive bullish/bearish
        bull_runs = df["is_bullish"].astype(int).values
        consec_bull = np.zeros(len(df))
        consec_bear = np.zeros(len(df))
        cb = 0
        cr = 0
        for i, b in enumerate(bull_runs):
            if b == 1:
                cb += 1
                cr = 0
            else:
                cb = 0
                cr += 1
            consec_bull[i] = min(cb, 5)
            consec_bear[i] = min(cr, 5)
        f["consecutive_bull"] = consec_bull
        f["consecutive_bear"] = consec_bear

        f["higher_high"] = (df["high"] > prev_high).astype(float)
        f["lower_low"] = (df["low"] < prev_low).astype(float)

        ema_trend = (df["ema21"] > df["ema50"]).astype(float)
        f["pullback_depth"] = np.where(
            (ema_trend == 1) & (df["atr14"] > 0),
            (df["ema21"] - df["low"]) / df["atr14"],
            np.where(
                (ema_trend == 0) & (df["atr14"] > 0),
                (df["high"] - df["ema21"]) / df["atr14"],
                0,
            ),
        )

        return f

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, features: Dict[str, float]) -> Dict:
        """
        Score a single bar's features.

        Parameters
        ----------
        features : dict
            Feature name -> value (from calculate_features).

        Returns
        -------
        dict with keys:
            signal, confidence, buy_prob, sell_prob, no_trade_prob,
            score (1-10), top_reasons (list of strings)
        """
        # Build feature vector in correct order
        x = np.array(
            [[features.get(name, 0.0) for name in self.feature_names]]
        )

        proba = self.model.predict_proba(x)[0]
        buy_prob = float(proba[BUY])
        sell_prob = float(proba[SELL])
        no_trade_prob = float(proba[NO_TRADE])

        # Determine signal
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        signal_map = {BUY: "BUY", SELL: "SELL", NO_TRADE: "NO_TRADE"}
        signal = signal_map[pred_class]

        # Human-readable score 1-10
        # Based on confidence of the directional signal
        directional_conf = max(buy_prob, sell_prob)
        if signal == "NO_TRADE":
            score_10 = max(1, int(no_trade_prob * 3))  # 1-3 for no trade
        else:
            # Map 0.33-1.0 confidence to 1-10 scale
            score_10 = max(1, min(10, int((directional_conf - 0.30) / 0.07) + 1))

        # Generate reasons
        top_reasons = self.get_top_reasons(features, pred_class)

        return {
            "signal": signal,
            "confidence": confidence,
            "buy_prob": buy_prob,
            "sell_prob": sell_prob,
            "no_trade_prob": no_trade_prob,
            "score": score_10,
            "top_reasons": top_reasons,
        }

    def score_bars(self, bars_m30: pd.DataFrame) -> Dict:
        """
        Convenience method: calculate features and score in one call.

        Parameters
        ----------
        bars_m30 : DataFrame
            At least 60 M30 bars with OHLCV data.

        Returns
        -------
        dict — same as score()
        """
        features = self.calculate_features(bars_m30)
        return self.score(features)

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------
    def get_top_reasons(
        self, features: Dict[str, float], prediction: int, max_reasons: int = 5
    ) -> List[str]:
        """
        Generate human-readable explanations for the prediction.

        Uses feature importance + current feature values to produce
        sentences like "RSI at 28.5 (oversold)".
        """
        reasons = []

        # Contextual explanations based on feature values
        rsi = features.get("rsi14", 50)
        if rsi < 30:
            reasons.append(f"RSI at {rsi:.1f} (oversold zone)")
        elif rsi > 70:
            reasons.append(f"RSI at {rsi:.1f} (overbought zone)")
        elif rsi < 40:
            reasons.append(f"RSI at {rsi:.1f} (approaching oversold)")
        elif rsi > 60:
            reasons.append(f"RSI at {rsi:.1f} (approaching overbought)")

        # BB position
        bb_pos = features.get("bb_position", 0.5)
        if bb_pos < 0.1:
            reasons.append(f"Price near BB lower band (position {bb_pos:.2f})")
        elif bb_pos > 0.9:
            reasons.append(f"Price near BB upper band (position {bb_pos:.2f})")
        elif bb_pos < 0.25:
            reasons.append(f"Price in lower BB zone (position {bb_pos:.2f})")
        elif bb_pos > 0.75:
            reasons.append(f"Price in upper BB zone (position {bb_pos:.2f})")

        # Trend
        ema_trend = features.get("ema_trend", 0)
        ema_gap = features.get("ema_gap_pct", 0)
        if ema_trend == 1:
            reasons.append(f"Bullish trend (EMA21 > EMA50, gap {ema_gap:.2f}%)")
        else:
            reasons.append(f"Bearish trend (EMA21 < EMA50, gap {ema_gap:.2f}%)")

        # Patterns
        if features.get("bullish_engulfing", 0) == 1:
            reasons.append("Bullish engulfing candle detected")
        if features.get("bearish_engulfing", 0) == 1:
            reasons.append("Bearish engulfing candle detected")
        if features.get("pin_bar_bull", 0) == 1:
            reasons.append("Bullish pin bar detected")
        if features.get("pin_bar_bear", 0) == 1:
            reasons.append("Bearish pin bar detected")
        if features.get("inside_bar", 0) == 1:
            reasons.append("Inside bar (consolidation)")

        # Volatility regime
        atr_vs_avg = features.get("atr_vs_avg", 1)
        if atr_vs_avg > 1.5:
            reasons.append(f"HIGH volatility regime (ATR {atr_vs_avg:.1f}x average)")
        elif atr_vs_avg < 0.7:
            reasons.append(f"LOW volatility regime (ATR {atr_vs_avg:.1f}x average)")
        else:
            reasons.append(f"Normal volatility (ATR {atr_vs_avg:.1f}x average)")

        # Monster bar
        if features.get("is_monster_bar", 0) == 1:
            reasons.append("Monster bar detected (range > 2x average)")

        # Momentum
        ret1 = features.get("return_1", 0)
        ret6 = features.get("return_6", 0)
        if abs(ret6) > 0:
            direction = "up" if ret6 > 0 else "down"
            reasons.append(f"6-bar momentum: {ret6:.1f}pts ({direction})")

        # Session
        is_london = features.get("is_london", 0)
        is_ny = features.get("is_ny", 0)
        hour = features.get("utc_hour", 12)
        if is_london and is_ny:
            reasons.append(f"London/NY overlap session (UTC {int(hour)}:00)")
        elif is_london:
            reasons.append(f"London session (UTC {int(hour)}:00)")
        elif is_ny:
            reasons.append(f"New York session (UTC {int(hour)}:00)")
        else:
            reasons.append(f"Asian/off-session (UTC {int(hour)}:00)")

        # Stochastic
        stoch = features.get("stoch_k", 50)
        if stoch < 20:
            reasons.append(f"Stochastic oversold at {stoch:.0f}")
        elif stoch > 80:
            reasons.append(f"Stochastic overbought at {stoch:.0f}")

        # Pullback depth
        pullback = features.get("pullback_depth", 0)
        if pullback > 1.0:
            reasons.append(f"Deep pullback ({pullback:.1f}x ATR from EMA21)")

        # MACD
        macd = features.get("macd", 0)
        macd_sig = features.get("macd_signal", 0)
        if macd > macd_sig and macd > 0:
            reasons.append("MACD bullish (above signal, positive)")
        elif macd < macd_sig and macd < 0:
            reasons.append("MACD bearish (below signal, negative)")

        # Consecutive bars
        consec_bull = features.get("consecutive_bull", 0)
        consec_bear = features.get("consecutive_bear", 0)
        if consec_bull >= 3:
            reasons.append(f"{int(consec_bull)} consecutive bullish bars")
        if consec_bear >= 3:
            reasons.append(f"{int(consec_bear)} consecutive bearish bars")

        # Sort by relevance to prediction direction
        if prediction == BUY:
            # Prioritize bullish reasons
            priority_keywords = [
                "oversold", "lower band", "Bullish", "bullish", "up",
                "London", "overlap", "pin bar bull",
            ]
        elif prediction == SELL:
            priority_keywords = [
                "overbought", "upper band", "Bearish", "bearish", "down",
                "London", "overlap", "pin bar bear",
            ]
        else:
            priority_keywords = ["volatility", "consolidation", "off-session", "Asian"]

        def reason_priority(r):
            for i, kw in enumerate(priority_keywords):
                if kw.lower() in r.lower():
                    return i
            return 100

        reasons.sort(key=reason_priority)
        return reasons[:max_reasons]

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------
    def score_history(
        self, bars_m30: pd.DataFrame, last_n: int = 100
    ) -> pd.DataFrame:
        """
        Score the last N bars of historical data.

        Returns DataFrame with columns: signal, confidence, buy_prob,
        sell_prob, no_trade_prob, score.
        """
        df = bars_m30.copy()
        df = self._ensure_indicators(df)
        all_features = self._engineer_all_features(df)

        # Only score last N bars (need warmup for earlier bars)
        start = max(0, len(all_features) - last_n)
        results = []

        for i in range(start, len(all_features)):
            row = all_features.iloc[i]
            feat_dict = {
                col: float(row[col]) if not pd.isna(row[col]) else 0.0
                for col in all_features.columns
            }
            result = self.score(feat_dict)
            result["time"] = all_features.index[i]
            result["close"] = float(df["close"].iloc[i])
            results.append(result)

        return pd.DataFrame(results).set_index("time")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading scorer...")
    scorer = GoldSignalScorer()
    print(f"Model loaded with {len(scorer.feature_names)} features")

    # Test with parquet data
    data_path = "D:/FOREX/data/xauusd_m30_analysis.parquet"
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        print(f"\nScoring last bar of historical data ({df.index[-1]})...")
        result = scorer.score_bars(df.tail(200))

        print(f"\n  Signal:     {result['signal']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Score:      {result['score']}/10")
        print(f"  Buy prob:   {result['buy_prob']*100:.1f}%")
        print(f"  Sell prob:  {result['sell_prob']*100:.1f}%")
        print(f"  No trade:   {result['no_trade_prob']*100:.1f}%")
        print(f"\n  Top Reasons:")
        for i, reason in enumerate(result["top_reasons"], 1):
            print(f"    {i}. {reason}")
    else:
        print(f"No data at {data_path} for testing.")

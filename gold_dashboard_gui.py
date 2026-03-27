"""
Gold AI Signal Scorer — Bloomberg Terminal Style Dashboard (tkinter)

No browser needed. Reads live M30 bars from MT5, scores with XGBoost model,
displays signal/confidence/reasons in a Bloomberg-terminal-themed dashboard.

Auto-refreshes every 30 seconds via background thread.

Launch: pythonw gold_dashboard_gui.py   (no console window)
   or:  python  gold_dashboard_gui.py   (with console for debugging)
"""

import sys
import os
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta

import tkinter as tk
from tkinter import ttk, font as tkfont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
sys.path.insert(0, "D:/FOREX/python")

MODEL_PATH = "D:/FOREX/models/gold_signal_model.json"
CONFIG_PATH = "D:/FOREX/models/gold_signal_config.json"

# ---------------------------------------------------------------------------
# Bloomberg Theme Constants
# ---------------------------------------------------------------------------
BG_BLACK = "#000000"
BG_PANEL = "#0a0a0a"
BG_BORDER = "#333333"
CLR_ORANGE = "#ff8c00"       # Bloomberg orange — primary text
CLR_ORANGE_ACC = "#ff6600"   # Orange accent lines
CLR_GREEN = "#00ff00"        # Terminal green for numbers
CLR_NEG = "#ff3333"          # Negative / sell red
CLR_POS = "#00cc00"          # Positive / buy green
CLR_WHITE = "#ffffff"        # Header text
CLR_MUTED = "#666666"        # Muted / labels
CLR_YELLOW = "#ffff00"       # Flash color for price changes
CLR_DIM_GREEN = "#008800"    # Dimmer green for subtle elements

FONT_MONO = "Consolas"       # Primary monospace font
FONT_MONO_ALT = "Courier New"  # Fallback

REFRESH_MS = 30_000   # 30 seconds
CLOCK_MS = 1_000      # 1 second
BLINK_MS = 2_000      # 2 seconds for signal blink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session_label(utc_hour: float) -> tuple:
    """Return (session_name, color) for a UTC hour."""
    h = int(utc_hour) % 24
    if 13 <= h < 16:
        return "LDN/NY", CLR_ORANGE
    if 7 <= h < 16:
        return "LONDON", CLR_POS
    if 13 <= h < 21:
        return "NEW YORK", "#55aaff"
    return "ASIAN", CLR_NEG


def _adx_label(val: float) -> tuple:
    """Return (label, color) for ADX value."""
    if val >= 40:
        return "VSTRNG", CLR_POS
    if val >= 25:
        return "STRNG", CLR_GREEN
    if val >= 20:
        return "MOD", CLR_ORANGE
    return "WEAK", CLR_NEG


def _rsi_color(val: float) -> str:
    if val < 35:
        return CLR_POS
    if val > 65:
        return CLR_NEG
    return CLR_GREEN


def _arrow(val: float, threshold_low: float = 40, threshold_high: float = 60) -> str:
    """Return directional arrow based on value."""
    if val < threshold_low:
        return "\u25bc"  # down
    if val > threshold_high:
        return "\u25b2"  # up
    return "\u25c6"  # diamond (neutral)


def _score_blocks(score: int) -> str:
    return "\u2588" * score + "\u2591" * (10 - score)


def _score_color(score: int) -> str:
    if score >= 8:
        return CLR_POS
    if score >= 5:
        return CLR_ORANGE
    return CLR_NEG


def _prob_bar(pct: float, max_chars: int = 40) -> str:
    """Return a bar of block characters proportional to percentage."""
    n = max(0, int(pct * max_chars))
    return "\u2588" * n


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class GoldDashboardApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GOLD AI SIGNAL SCORER  |  XAUUSD  |  BLOOMBERG TERMINAL")
        self.root.geometry("1100x900")
        self.root.configure(bg=BG_BLACK)
        self.root.minsize(1000, 800)

        # Try to start maximized
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        # Dark title bar on Windows 10/11
        try:
            from ctypes import windll, byref, c_int, sizeof
            self.root.update()
            hwnd = windll.user32.GetParent(self.root.winfo_id())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                byref(c_int(1)), sizeof(c_int)
            )
        except Exception:
            pass

        # State
        self.scorer = None
        self.mt5_connected = False
        self.signal_history = []
        self.last_error = ""
        self._stop_event = threading.Event()
        self._last_price = None
        self._blink_visible = True
        self._current_signal = "NO_TRADE"
        self._feature_count = 0

        self._build_ui()
        self._load_scorer()
        self._start_background()
        self._tick_clock()
        self._tick_blink()

    # ==================================================================
    # UI CONSTRUCTION
    # ==================================================================
    def _build_ui(self):
        # Scrollable canvas for the whole window
        self.canvas = tk.Canvas(self.root, bg=BG_BLACK, highlightthickness=0,
                                borderwidth=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical",
                                      command=self.canvas.yview,
                                      bg=BG_BLACK, troughcolor=BG_BLACK,
                                      activebackground=BG_BORDER)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.main_frame = tk.Frame(self.canvas, bg=BG_BLACK)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.main_frame, anchor="nw"
        )

        self.main_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Build all panels
        self._build_header_bar()
        self._build_orange_line()
        self._build_signal_and_metrics()
        self._build_separator()
        self._build_analysis_panel()
        self._build_separator()
        self._build_probability_panel()
        self._build_separator()
        self._build_history_panel()
        self._build_orange_line()
        self._build_status_bar()

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # HEADER BAR
    # ------------------------------------------------------------------
    def _build_header_bar(self):
        frame = tk.Frame(self.main_frame, bg=BG_PANEL, padx=12, pady=6)
        frame.pack(fill="x")

        # Left: title
        tk.Label(
            frame, text="GOLD AI SIGNAL SCORER",
            font=(FONT_MONO, 14, "bold"),
            fg=CLR_ORANGE, bg=BG_PANEL
        ).pack(side="left")

        # Spacer with symbol
        tk.Label(
            frame, text="    XAUUSD",
            font=(FONT_MONO, 14, "bold"),
            fg=CLR_WHITE, bg=BG_PANEL
        ).pack(side="left")

        # Right side
        right = tk.Frame(frame, bg=BG_PANEL)
        right.pack(side="right")

        # Clock
        self.lbl_clock = tk.Label(
            right, text="----.--.--  --:--:--",
            font=(FONT_MONO, 12),
            fg=CLR_ORANGE, bg=BG_PANEL
        )
        self.lbl_clock.pack(side="right")

    # ------------------------------------------------------------------
    # SEPARATOR HELPERS
    # ------------------------------------------------------------------
    def _build_separator(self):
        tk.Frame(self.main_frame, height=1, bg=BG_BORDER).pack(fill="x", padx=0)

    def _build_orange_line(self):
        tk.Frame(self.main_frame, height=2, bg=CLR_ORANGE_ACC).pack(fill="x", padx=0)

    # ------------------------------------------------------------------
    # SIGNAL + METRICS (two-column panel)
    # ------------------------------------------------------------------
    def _build_signal_and_metrics(self):
        container = tk.Frame(self.main_frame, bg=BG_BLACK)
        container.pack(fill="x")
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=0)  # thin separator
        container.columnconfigure(2, weight=2)

        # LEFT: Signal area
        left = tk.Frame(container, bg=BG_PANEL, padx=20, pady=16)
        left.grid(row=0, column=0, sticky="nsew")

        # Signal direction - large
        self.lbl_signal = tk.Label(
            left, text="WAITING",
            font=(FONT_MONO, 60, "bold"),
            fg=CLR_MUTED, bg=BG_PANEL
        )
        self.lbl_signal.pack(anchor="w")

        # Confidence
        self.lbl_confidence = tk.Label(
            left, text="Confidence: --.-%",
            font=(FONT_MONO, 14),
            fg=CLR_MUTED, bg=BG_PANEL
        )
        self.lbl_confidence.pack(anchor="w", pady=(2, 0))

        # Score bar row
        score_row = tk.Frame(left, bg=BG_PANEL)
        score_row.pack(anchor="w", pady=(4, 0))

        tk.Label(
            score_row, text="Score:",
            font=(FONT_MONO, 13),
            fg=CLR_MUTED, bg=BG_PANEL
        ).pack(side="left")

        self.lbl_score_blocks = tk.Label(
            score_row, text="\u2591" * 10,
            font=(FONT_MONO, 16),
            fg=CLR_MUTED, bg=BG_PANEL
        )
        self.lbl_score_blocks.pack(side="left", padx=(6, 0))

        self.lbl_score_num = tk.Label(
            score_row, text="--/10",
            font=(FONT_MONO, 14, "bold"),
            fg=CLR_MUTED, bg=BG_PANEL
        )
        self.lbl_score_num.pack(side="left", padx=(8, 0))

        # Price row
        price_row = tk.Frame(left, bg=BG_PANEL)
        price_row.pack(anchor="w", pady=(12, 0))

        self.lbl_price = tk.Label(
            price_row, text="-----.--",
            font=(FONT_MONO, 32, "bold"),
            fg=CLR_GREEN, bg=BG_PANEL
        )
        self.lbl_price.pack(side="left")

        self.lbl_price_change = tk.Label(
            price_row, text="",
            font=(FONT_MONO, 16),
            fg=CLR_MUTED, bg=BG_PANEL
        )
        self.lbl_price_change.pack(side="left", padx=(16, 0))

        # Vertical separator
        tk.Frame(container, width=1, bg=BG_BORDER).grid(row=0, column=1, sticky="ns")

        # RIGHT: Signal Metrics
        right = tk.Frame(container, bg=BG_PANEL, padx=16, pady=12)
        right.grid(row=0, column=2, sticky="nsew")

        # Section header
        tk.Label(
            right, text="SIGNAL METRICS",
            font=(FONT_MONO, 10, "bold"),
            fg=CLR_WHITE, bg=BG_PANEL
        ).pack(anchor="w")

        # Thin line under header
        tk.Frame(right, height=1, bg=BG_BORDER).pack(fill="x", pady=(3, 6))

        # Metrics rows - each is label + value right-aligned
        self.metric_cells = {}
        metrics = [
            ("RSI(14)", "rsi"),
            ("STOCH.K", "stoch_k"),
            ("ATR(14)", "atr"),
            ("ADX", "adx"),
            ("BB.POS", "bb_pos"),
            ("EMA TREND", "ema_trend"),
            ("MACD", "macd"),
            ("SESSION", "session"),
        ]

        for label_text, key in metrics:
            row = tk.Frame(right, bg=BG_PANEL)
            row.pack(fill="x", pady=1)

            tk.Label(
                row, text=label_text,
                font=(FONT_MONO, 10),
                fg=CLR_MUTED, bg=BG_PANEL, anchor="w", width=12
            ).pack(side="left")

            val_lbl = tk.Label(
                row, text="--",
                font=(FONT_MONO, 13, "bold"),
                fg=CLR_GREEN, bg=BG_PANEL, anchor="e"
            )
            val_lbl.pack(side="right")

            extra_lbl = tk.Label(
                row, text="",
                font=(FONT_MONO, 10),
                fg=CLR_MUTED, bg=BG_PANEL, anchor="e"
            )
            extra_lbl.pack(side="right", padx=(0, 6))

            self.metric_cells[key] = (val_lbl, extra_lbl)

    # ------------------------------------------------------------------
    # ANALYSIS PANEL
    # ------------------------------------------------------------------
    def _build_analysis_panel(self):
        frame = tk.Frame(self.main_frame, bg=BG_PANEL, padx=16, pady=10)
        frame.pack(fill="x")

        # Section header
        tk.Label(
            frame, text="ANALYSIS",
            font=(FONT_MONO, 10, "bold"),
            fg=CLR_WHITE, bg=BG_PANEL
        ).pack(anchor="w")

        tk.Frame(frame, height=1, bg=BG_BORDER).pack(fill="x", pady=(3, 6))

        self.reason_labels = []
        for i in range(5):
            lbl = tk.Label(
                frame, text="",
                font=(FONT_MONO, 11),
                fg=CLR_MUTED, bg=BG_PANEL, anchor="w", justify="left"
            )
            lbl.pack(fill="x", pady=0)
            self.reason_labels.append(lbl)

    # ------------------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # ------------------------------------------------------------------
    def _build_probability_panel(self):
        frame = tk.Frame(self.main_frame, bg=BG_PANEL, padx=16, pady=10)
        frame.pack(fill="x")

        # Section header
        tk.Label(
            frame, text="PROBABILITY DISTRIBUTION",
            font=(FONT_MONO, 10, "bold"),
            fg=CLR_WHITE, bg=BG_PANEL
        ).pack(anchor="w")

        tk.Frame(frame, height=1, bg=BG_BORDER).pack(fill="x", pady=(3, 6))

        # BUY row
        buy_row = tk.Frame(frame, bg=BG_PANEL)
        buy_row.pack(fill="x", pady=1)
        tk.Label(
            buy_row, text="BUY  ",
            font=(FONT_MONO, 11, "bold"),
            fg=CLR_POS, bg=BG_PANEL, width=6, anchor="w"
        ).pack(side="left")
        self.lbl_prob_buy_bar = tk.Label(
            buy_row, text="",
            font=(FONT_MONO, 11),
            fg=CLR_POS, bg=BG_PANEL, anchor="w"
        )
        self.lbl_prob_buy_bar.pack(side="left", fill="x", expand=True)
        self.lbl_prob_buy_pct = tk.Label(
            buy_row, text="--.-%",
            font=(FONT_MONO, 11, "bold"),
            fg=CLR_POS, bg=BG_PANEL, anchor="e", width=8
        )
        self.lbl_prob_buy_pct.pack(side="right")

        # SELL row
        sell_row = tk.Frame(frame, bg=BG_PANEL)
        sell_row.pack(fill="x", pady=1)
        tk.Label(
            sell_row, text="SELL ",
            font=(FONT_MONO, 11, "bold"),
            fg=CLR_NEG, bg=BG_PANEL, width=6, anchor="w"
        ).pack(side="left")
        self.lbl_prob_sell_bar = tk.Label(
            sell_row, text="",
            font=(FONT_MONO, 11),
            fg=CLR_NEG, bg=BG_PANEL, anchor="w"
        )
        self.lbl_prob_sell_bar.pack(side="left", fill="x", expand=True)
        self.lbl_prob_sell_pct = tk.Label(
            sell_row, text="--.-%",
            font=(FONT_MONO, 11, "bold"),
            fg=CLR_NEG, bg=BG_PANEL, anchor="e", width=8
        )
        self.lbl_prob_sell_pct.pack(side="right")

        # SKIP row
        skip_row = tk.Frame(frame, bg=BG_PANEL)
        skip_row.pack(fill="x", pady=1)
        tk.Label(
            skip_row, text="SKIP ",
            font=(FONT_MONO, 11, "bold"),
            fg=CLR_MUTED, bg=BG_PANEL, width=6, anchor="w"
        ).pack(side="left")
        self.lbl_prob_skip_bar = tk.Label(
            skip_row, text="",
            font=(FONT_MONO, 11),
            fg=CLR_MUTED, bg=BG_PANEL, anchor="w"
        )
        self.lbl_prob_skip_bar.pack(side="left", fill="x", expand=True)
        self.lbl_prob_skip_pct = tk.Label(
            skip_row, text="--.-%",
            font=(FONT_MONO, 11, "bold"),
            fg=CLR_MUTED, bg=BG_PANEL, anchor="e", width=8
        )
        self.lbl_prob_skip_pct.pack(side="right")

    # ------------------------------------------------------------------
    # SIGNAL HISTORY
    # ------------------------------------------------------------------
    def _build_history_panel(self):
        frame = tk.Frame(self.main_frame, bg=BG_PANEL, padx=16, pady=10)
        frame.pack(fill="x")

        # Section header
        tk.Label(
            frame, text="SIGNAL HISTORY",
            font=(FONT_MONO, 10, "bold"),
            fg=CLR_WHITE, bg=BG_PANEL
        ).pack(anchor="w")

        tk.Frame(frame, height=1, bg=BG_BORDER).pack(fill="x", pady=(3, 4))

        # Column headers
        hdr = tk.Frame(frame, bg=BG_PANEL)
        hdr.pack(fill="x")

        cols = [("TIME", 20), ("PRICE", 12), ("SIGNAL", 10),
                ("CONF", 8), ("SCORE", 6)]
        for text, w in cols:
            tk.Label(
                hdr, text=text,
                font=(FONT_MONO, 10, "bold"),
                fg=CLR_MUTED, bg=BG_PANEL, width=w, anchor="w"
            ).pack(side="left", padx=1)

        # Separator under header
        tk.Frame(frame, height=1, bg=BG_BORDER).pack(fill="x", pady=(2, 2))

        # Scrollable rows container
        self.history_frame = tk.Frame(frame, bg=BG_PANEL)
        self.history_frame.pack(fill="x")

        self.history_row_widgets = []

    # ------------------------------------------------------------------
    # STATUS BAR
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        frame = tk.Frame(self.main_frame, bg=BG_BLACK, padx=12, pady=4)
        frame.pack(fill="x")

        # MT5 status
        self.lbl_mt5 = tk.Label(
            frame, text="MT5: OFFLINE",
            font=(FONT_MONO, 10, "bold"),
            fg=CLR_NEG, bg=BG_BLACK
        )
        self.lbl_mt5.pack(side="left")

        # Separator
        tk.Label(
            frame, text=" \u2502 ",
            font=(FONT_MONO, 10),
            fg=BG_BORDER, bg=BG_BLACK
        ).pack(side="left")

        # Model info
        self.lbl_model_info = tk.Label(
            frame, text="MODEL: --",
            font=(FONT_MONO, 10),
            fg=CLR_MUTED, bg=BG_BLACK
        )
        self.lbl_model_info.pack(side="left")

        # Separator
        tk.Label(
            frame, text=" \u2502 ",
            font=(FONT_MONO, 10),
            fg=BG_BORDER, bg=BG_BLACK
        ).pack(side="left")

        # Countdown
        self.lbl_countdown = tk.Label(
            frame, text="REFRESH: --s",
            font=(FONT_MONO, 10),
            fg=CLR_MUTED, bg=BG_BLACK
        )
        self.lbl_countdown.pack(side="left")

        # Separator
        tk.Label(
            frame, text=" \u2502 ",
            font=(FONT_MONO, 10),
            fg=BG_BORDER, bg=BG_BLACK
        ).pack(side="left")

        # Refresh count
        self.lbl_refresh_count = tk.Label(
            frame, text="#0",
            font=(FONT_MONO, 10),
            fg=CLR_MUTED, bg=BG_BLACK
        )
        self.lbl_refresh_count.pack(side="left")

        # Right side: status message
        self.lbl_status = tk.Label(
            frame, text="Starting up...",
            font=(FONT_MONO, 9),
            fg=CLR_MUTED, bg=BG_BLACK, anchor="e"
        )
        self.lbl_status.pack(side="right")

    # ==================================================================
    # SCORER
    # ==================================================================
    def _load_scorer(self):
        try:
            from gold_signal_scorer import GoldSignalScorer
            self.scorer = GoldSignalScorer(
                model_path=MODEL_PATH,
                config_path=CONFIG_PATH,
            )
            self._feature_count = len(self.scorer.feature_names)
            self.lbl_model_info.config(
                text=f"MODEL: {self._feature_count} FEATURES",
                fg=CLR_GREEN
            )
            self._set_status(f"Model loaded ({self._feature_count} features)")
        except Exception as e:
            self.scorer = None
            self._set_status(f"Model load FAILED: {e}")

    # ==================================================================
    # MT5 DATA
    # ==================================================================
    def _fetch_mt5_data(self):
        """
        Connect to MT5, pull 200 M30 bars for XAUUSD.m, return DataFrame.
        Returns (df, price, change_pct) or raises on failure.
        """
        import MetaTrader5 as mt5

        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")

        self.mt5_connected = True

        symbol = "XAUUSD.m"
        if not mt5.symbol_select(symbol, True):
            symbol = "XAUUSD"
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Symbol XAUUSD not found in MT5")

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 200)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No M30 data from MT5 for {symbol}")

        import pandas as pd
        import numpy as np

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        price = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        change_pct = (price - prev_close) / prev_close * 100

        return df, price, change_pct

    # ==================================================================
    # BACKGROUND REFRESH
    # ==================================================================
    def _start_background(self):
        self._refresh_count = 0
        self._last_refresh_time = None
        t = threading.Thread(target=self._background_loop, daemon=True)
        t.start()

    def _background_loop(self):
        """Runs in background thread. Fetches data + scores, then schedules UI update."""
        while not self._stop_event.is_set():
            self._do_refresh()
            self._stop_event.wait(REFRESH_MS / 1000.0)

    def _do_refresh(self):
        """Single refresh cycle (called from background thread)."""
        try:
            import pandas as pd

            df, price, change_pct = self._fetch_mt5_data()

            if self.scorer is None:
                self.root.after(0, self._update_mt5_status, True)
                self.root.after(0, self._update_price, price, change_pct)
                self.root.after(0, self._set_status, "Model not loaded")
                return

            # Score
            result = self.scorer.score_bars(df.tail(200))

            # Extract feature values for market grid
            features = self.scorer.calculate_features(df.tail(200))

            # Build update payload
            payload = {
                "signal": result["signal"],
                "confidence": result["confidence"],
                "score": result["score"],
                "buy_prob": result["buy_prob"],
                "sell_prob": result["sell_prob"],
                "no_trade_prob": result["no_trade_prob"],
                "reasons": result["top_reasons"],
                "price": price,
                "change_pct": change_pct,
                "features": features,
                "time": datetime.now(),
            }

            # Schedule UI update on main thread
            self.root.after(0, self._apply_update, payload)

        except ConnectionError as e:
            self.mt5_connected = False
            self.root.after(0, self._update_mt5_status, False)
            self.root.after(0, self._set_status, f"MT5 offline: {e}")
        except Exception as e:
            tb = traceback.format_exc()
            self.root.after(0, self._set_status, f"Error: {e}")
            print(f"[Refresh Error] {tb}", file=sys.stderr)

    # ==================================================================
    # UI UPDATES (called on main thread via root.after)
    # ==================================================================
    def _apply_update(self, p: dict):
        """Apply a full data update to the UI."""
        self._last_refresh_time = datetime.now()
        self._refresh_count += 1

        sig = p["signal"]
        conf = p["confidence"]
        score = p["score"]
        self._current_signal = sig

        # -- Signal colors --
        sig_colors = {
            "BUY": CLR_POS,
            "SELL": CLR_NEG,
            "NO_TRADE": CLR_MUTED
        }
        sig_color = sig_colors.get(sig, CLR_WHITE)

        # -- Signal text --
        display_signal = sig.replace("_", " ")
        self.lbl_signal.config(text=display_signal, fg=sig_color)

        # Flash background briefly on signal change
        self.lbl_signal.config(bg="#1a1a00" if sig == "BUY" else
                               "#1a0000" if sig == "SELL" else BG_PANEL)
        self.root.after(500, lambda: self.lbl_signal.config(bg=BG_PANEL))

        # -- Confidence --
        self.lbl_confidence.config(
            text=f"Confidence: {conf * 100:.1f}%",
            fg=sig_color
        )

        # -- Score blocks --
        blocks = _score_blocks(score)
        sc = _score_color(score)
        self.lbl_score_blocks.config(text=blocks, fg=sc)
        self.lbl_score_num.config(text=f"{score}/10", fg=sc)

        # -- Price --
        self._update_price(p["price"], p["change_pct"])

        # -- MT5 status --
        self._update_mt5_status(True)

        # -- Analysis / Reasons --
        for i, lbl in enumerate(self.reason_labels):
            if i < len(p["reasons"]):
                lbl.config(
                    text=f"  {i+1}. {p['reasons'][i]}",
                    fg=CLR_ORANGE
                )
            else:
                lbl.config(text="", fg=CLR_MUTED)

        # -- Probability bars (terminal style) --
        self._draw_prob_bars(p["buy_prob"], p["sell_prob"], p["no_trade_prob"])

        # -- Signal Metrics --
        self._update_market_grid(p["features"])

        # -- History --
        self._add_history_entry(p)

        # -- Status bar --
        self.lbl_refresh_count.config(text=f"#{self._refresh_count}")
        self._set_status(
            f"{sig} {conf*100:.0f}%  Score {score}/10  "
            f"Last: {self._last_refresh_time.strftime('%H:%M:%S')}"
        )

    def _update_price(self, price: float, change_pct: float):
        # Flash yellow briefly when price changes
        if self._last_price is not None and abs(price - self._last_price) > 0.01:
            self.lbl_price.config(fg=CLR_YELLOW)
            self.root.after(600, lambda: self.lbl_price.config(fg=CLR_GREEN))

        self._last_price = price
        self.lbl_price.config(text=f"{price:,.2f}")

        if change_pct >= 0:
            self.lbl_price_change.config(
                text=f"+{change_pct:.2f}%", fg=CLR_POS
            )
        else:
            self.lbl_price_change.config(
                text=f"{change_pct:.2f}%", fg=CLR_NEG
            )

    def _update_mt5_status(self, connected: bool):
        self.mt5_connected = connected
        if connected:
            self.lbl_mt5.config(text="MT5: CONNECTED", fg=CLR_POS)
        else:
            self.lbl_mt5.config(text="MT5: OFFLINE", fg=CLR_NEG)

    def _draw_prob_bars(self, buy: float, sell: float, notrade: float):
        """Draw probability bars using block characters."""
        max_chars = 40

        # BUY
        buy_bar = _prob_bar(buy, max_chars)
        self.lbl_prob_buy_bar.config(text=buy_bar)
        self.lbl_prob_buy_pct.config(text=f"{buy*100:.1f}%")

        # SELL
        sell_bar = _prob_bar(sell, max_chars)
        self.lbl_prob_sell_bar.config(text=sell_bar)
        self.lbl_prob_sell_pct.config(text=f"{sell*100:.1f}%")

        # SKIP (no_trade)
        skip_bar = _prob_bar(notrade, max_chars)
        self.lbl_prob_skip_bar.config(text=skip_bar)
        self.lbl_prob_skip_pct.config(text=f"{notrade*100:.1f}%")

    def _update_market_grid(self, feat: dict):
        # RSI
        rsi = feat.get("rsi14", 0)
        rsi_clr = _rsi_color(rsi)
        arrow = "\u25bc" if rsi < 40 else ("\u25b2" if rsi > 60 else "")
        val_lbl, extra_lbl = self.metric_cells["rsi"]
        val_lbl.config(text=f"{rsi:.1f}", fg=rsi_clr)
        extra_lbl.config(text=arrow, fg=rsi_clr)

        # Stoch K
        stoch = feat.get("stoch_k", 0)
        stoch_clr = CLR_POS if stoch < 20 else (CLR_NEG if stoch > 80 else CLR_GREEN)
        arrow = "\u25bc" if stoch < 30 else ("\u25b2" if stoch > 70 else "")
        val_lbl, extra_lbl = self.metric_cells["stoch_k"]
        val_lbl.config(text=f"{stoch:.1f}", fg=stoch_clr)
        extra_lbl.config(text=arrow, fg=stoch_clr)

        # ATR
        atr = feat.get("atr14", 0)
        atr_vs = feat.get("atr_vs_avg", 1)
        val_lbl, extra_lbl = self.metric_cells["atr"]
        val_lbl.config(text=f"{atr:.1f}", fg=CLR_GREEN)
        extra_lbl.config(
            text=f"{atr_vs:.1f}x",
            fg=CLR_NEG if atr_vs > 1.5 else (CLR_ORANGE if atr_vs > 1.2 else CLR_MUTED)
        )

        # ADX (approximated from atr_normalized)
        atr_norm = feat.get("atr_normalized", 0)
        val_lbl, extra_lbl = self.metric_cells["adx"]
        if atr_norm > 0:
            adx_approx = min(100, atr_norm * 30)
            lbl, clr = _adx_label(adx_approx)
            val_lbl.config(text=f"{adx_approx:.0f}", fg=clr)
            extra_lbl.config(text=lbl, fg=clr)
        else:
            val_lbl.config(text="N/A", fg=CLR_MUTED)
            extra_lbl.config(text="", fg=CLR_MUTED)

        # BB Position
        bb_pos = feat.get("bb_position", 0.5)
        bb_fg = CLR_NEG if bb_pos > 0.8 else (CLR_POS if bb_pos < 0.2 else CLR_GREEN)
        bb_label = "HIGH" if bb_pos > 0.7 else ("LOW" if bb_pos < 0.3 else "MID")
        val_lbl, extra_lbl = self.metric_cells["bb_pos"]
        val_lbl.config(text=f"{bb_pos:.2f}", fg=bb_fg)
        extra_lbl.config(text=bb_label, fg=bb_fg)

        # EMA Trend
        ema_trend = feat.get("ema_trend", 0)
        val_lbl, extra_lbl = self.metric_cells["ema_trend"]
        if ema_trend >= 1:
            val_lbl.config(text="BULL", fg=CLR_POS)
            extra_lbl.config(text="\u25b2", fg=CLR_POS)
        else:
            val_lbl.config(text="BEAR", fg=CLR_NEG)
            extra_lbl.config(text="\u25bc", fg=CLR_NEG)

        # MACD (use body_ratio as proxy if MACD not available)
        br = feat.get("body_ratio", 0)
        momentum = feat.get("momentum_6bar", 0)
        val_lbl, extra_lbl = self.metric_cells["macd"]
        if momentum != 0:
            m_clr = CLR_POS if momentum > 0 else CLR_NEG
            val_lbl.config(text=f"{momentum:.1f}", fg=m_clr)
            extra_lbl.config(text=f"BR:{br:.2f}", fg=CLR_MUTED)
        else:
            val_lbl.config(text=f"{br:.2f}", fg=CLR_GREEN)
            extra_lbl.config(text="BR", fg=CLR_MUTED)

        # Session
        utc_hour = feat.get("utc_hour", 12)
        sess_name, sess_clr = _session_label(utc_hour)
        val_lbl, extra_lbl = self.metric_cells["session"]
        val_lbl.config(text=sess_name, fg=sess_clr)
        extra_lbl.config(text=f"UTC{int(utc_hour):02d}", fg=CLR_MUTED)

    def _add_history_entry(self, p: dict):
        entry = {
            "time": p["time"].strftime("%H:%M:%S"),
            "price": p["price"],
            "signal": p["signal"],
            "confidence": p["confidence"],
            "score": p["score"],
        }
        self.signal_history.insert(0, entry)
        self.signal_history = self.signal_history[:20]

        # Rebuild history rows
        for w in self.history_row_widgets:
            w.destroy()
        self.history_row_widgets.clear()

        for i, e in enumerate(self.signal_history):
            bg = BG_PANEL if i % 2 == 0 else BG_BLACK
            sig_color = {
                "BUY": CLR_POS, "SELL": CLR_NEG, "NO_TRADE": CLR_MUTED
            }.get(e["signal"], CLR_WHITE)

            row = tk.Frame(self.history_frame, bg=bg)
            row.pack(fill="x")
            self.history_row_widgets.append(row)

            # Time
            tk.Label(
                row, text=e["time"],
                font=(FONT_MONO, 10),
                fg=CLR_MUTED, bg=bg, width=20, anchor="w"
            ).pack(side="left", padx=1)

            # Price
            tk.Label(
                row, text=f"{e['price']:,.2f}",
                font=(FONT_MONO, 10),
                fg=CLR_GREEN, bg=bg, width=12, anchor="w"
            ).pack(side="left", padx=1)

            # Signal
            sig_display = e["signal"].replace("_", " ")
            tk.Label(
                row, text=sig_display,
                font=(FONT_MONO, 10, "bold"),
                fg=sig_color, bg=bg, width=10, anchor="w"
            ).pack(side="left", padx=1)

            # Confidence
            tk.Label(
                row, text=f"{e['confidence']*100:.1f}%",
                font=(FONT_MONO, 10),
                fg=CLR_ORANGE, bg=bg, width=8, anchor="w"
            ).pack(side="left", padx=1)

            # Score
            sc = _score_color(e["score"])
            tk.Label(
                row, text=f"{e['score']}/10",
                font=(FONT_MONO, 10, "bold"),
                fg=sc, bg=bg, width=6, anchor="w"
            ).pack(side="left", padx=1)

    # ==================================================================
    # CLOCK, BLINK & STATUS
    # ==================================================================
    def _tick_clock(self):
        now = datetime.now()
        self.lbl_clock.config(text=now.strftime("%Y-%m-%d  %H:%M:%S"))

        # Countdown to next refresh
        if self._last_refresh_time:
            elapsed = (now - self._last_refresh_time).total_seconds()
            remaining = max(0, int(REFRESH_MS / 1000.0 - elapsed))
            self.lbl_countdown.config(text=f"REFRESH: {remaining}s")
        else:
            self.lbl_countdown.config(text="REFRESH: --s")

        self.root.after(CLOCK_MS, self._tick_clock)

    def _tick_blink(self):
        """Subtle blink effect on the signal label every 2 seconds."""
        if self._current_signal in ("BUY", "SELL"):
            self._blink_visible = not self._blink_visible
            if self._blink_visible:
                sig_colors = {"BUY": CLR_POS, "SELL": CLR_NEG}
                self.lbl_signal.config(fg=sig_colors.get(self._current_signal, CLR_WHITE))
            else:
                # Slightly dimmer version for blink
                sig_colors_dim = {"BUY": CLR_DIM_GREEN, "SELL": "#881111"}
                self.lbl_signal.config(
                    fg=sig_colors_dim.get(self._current_signal, CLR_MUTED)
                )
        self.root.after(BLINK_MS, self._tick_blink)

    def _set_status(self, text: str):
        self.lbl_status.config(text=text)

    # ==================================================================
    # RUN
    # ==================================================================
    def run(self):
        try:
            self.root.mainloop()
        finally:
            self._stop_event.set()
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = GoldDashboardApp()
    app.run()

# ⚡ 243A Algo Trading Console — Complete A-to-Z Guide

An institutional-grade algorithmic trading console, execution sandbox, and real-time analytics suite built for the **NIFTY 50 Index** and its 50 constituent stocks. The platform integrates direct tick-by-tick streaming via **Angel One SmartAPI SmartWebSocketV2**, an ensemble of **3 Machine Learning models (LightGBM, TCN, HMM)**, an **18-year 15-Minute Historical Pattern Match Engine**, a thermodynamic **Thermal Dissipation Lot Sizer**, and high-performance interactive charting powered by **TradingView Lightweight Charts**.

---

## 📑 Table of Contents

1. [Local Server Quickstart (How to Run)](#1-local-server-quickstart-how-to-run)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [The 3 ML Models Prediction Engine](#3-the-3-ml-models-prediction-engine)
4. [15-Minute Historical Pattern Match Engine](#4-15-minute-historical-pattern-match-engine)
5. [Dynamic Thermal Dissipation Lot Sizer](#5-dynamic-thermal-dissipation-lot-sizer)
6. [Active Trade Management & Concurrent Slots](#6-active-trade-management--concurrent-slots)
7. [Institutional Risk & Performance Metrics Dashboard](#7-institutional-risk--performance-metrics-dashboard)
8. [Real-Time Market Breadth & Sentiment Meter](#8-real-time-market-breadth--sentiment-meter)
9. [Interactive Candlestick & Volume Charting](#9-interactive-candlestick--volume-charting)
10. [Right-Column Deck Structure & Layout](#10-right-column-deck-structure--layout)
11. [Project Directory Structure & Key Files](#11-project-directory-structure--key-files)
12. [Safety, Paper Trading & Confidentiality Guidelines](#12-safety-paper-trading--confidentiality-guidelines)

---

## 1. Local Server Quickstart (How to Run)

### Prerequisites
* **Python**: 3.10 or higher (recommended: Python 3.10 or 3.11).
* **OS**: Windows, Linux, or macOS.
* **C++ Build Tools** (Windows only): Required by TA-Lib and PyTorch if building from source.

---

### Step-by-Step Instructions

#### Step 1: Open Terminal & Navigate to Project Directory
```powershell
cd d:\algo-trading-console\44AAA_risk_ok
```

#### Step 2: Create & Activate a Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

#### Step 4: Launch the Local Server
Run the single startup command:
```powershell
python app.py
```

#### What `python app.py` Does Automatically Behind the Scenes:
1. **Verifies TA-Lib**: Checks if the C TA-Lib library and Python wrapper are present; compiles or binds them dynamically without manual wheel downloading.
2. **Loads Historical Data**: Caches 108,000+ 5-minute candles from `backend_engine/old data.csv` into memory for sub-millisecond lookups.
3. **Builds Pattern KD-Tree**: Preloads 99,205 historical 15-minute pattern vectors into RAM in ~0.11 seconds.
4. **Boots 3 ML Models**: Instantiates LightGBM, TCN, and HMM inference pipelines.
5. **Connects to WebSocket**: Authenticates with Angel One SmartAPI and begins live streaming ticks for NIFTY 50 and all 50 constituent stocks.
6. **Starts FastAPI Server**: Runs Uvicorn on port `7860` (`http://0.0.0.0:7860`).

---

### Accessing the Web Console in Your Browser

Once the terminal outputs `Uvicorn running on http://0.0.0.0:7860`, open your browser:

| Interface | URL | Description |
| :--- | :--- | :--- |
| **Main Trading Console** | **[http://127.0.0.1:7860/](http://127.0.0.1:7860/)** or **[http://localhost:7860/](http://localhost:7860/)** | The primary institutional dark-theme console with full real-time streaming, chart lines, and deck. |
| **Console Direct Path** | [http://127.0.0.1:7860/console](http://127.0.0.1:7860/console) | Direct alias to the main trading console. |
| **Legacy Dashboard** | [http://127.0.0.1:7860/legacy](http://127.0.0.1:7860/legacy) | Legacy glassmorphism view (`index.html`). |
| **GPU Node Status** | [http://127.0.0.1:7860/gradio](http://127.0.0.1:7860/gradio) | ZeroGPU / Gradio status monitor. |

---

## 2. High-Level System Architecture

```
                                 ┌──────────────────────────────────────────────┐
                                 │     Angel One SmartWebSocketV2 (Mode 2)      │
                                 └──────────────────────┬───────────────────────┘
                                                        │ Live Ticks (NIFTY + 50 Stocks)
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ backend_engine/live_feed_service.py          │
                                 │ • 5-Min Candle Aggregator                    │
                                 │ • True Constituent Volume Accumulator        │
                                 │ • Active Trade Slot Monitor (SL 60 / TP / EOD)│
                                 └───────┬──────────────┬──────────────┬────────┘
                                         │              │              │
                   ┌─────────────────────┘              │              └─────────────────────┐
                   ▼                                    ▼                                    ▼
┌────────────────────────────────────┐ ┌────────────────────────────────────┐ ┌────────────────────────────────────┐
│ 3 ML Models Prediction Engine      │ │ 15M Historical Pattern Match Engine│ │ Thermodynamic Thermal Lot Sizer  │
│ • LightGBM Classifier (BUY/SELL)   │ │ • 18-Year Vector Match (KD-Tree)   │ │ • Newton's Law of Cooling Math   │
│ • Temporal Convolutional Net (TCN) │ │ • Line 1: Target (-20% Reduction)  │ │ • Dynamic Temperature T (0.5-4.4)│
│ • Hidden Markov Model (HMM Regime) │ │ • Line 2: 0.05° Angular Slope      │ │ • Dynamic Sizing: 1x to 4x Lots  │
└──────────────────┬─────────────────┘ └──────────────────┬─────────────────┘ └──────────────────┬─────────────────┘
                   │                                      │                                      │
                   └──────────────────────────────┬───────┴──────────────────────────────────────┘
                                                  ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ FastAPI REST & WebSocket Endpoints           │
                                 │ • /api/live/state  • /api/live/candles       │
                                 │ • /api/live/ticks  • /api/health             │
                                 └──────────────────────┬───────────────────────┘
                                                        │ JSON Payload (every 600ms - 1s)
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ Frontend Console (ui_ux/console.html)        │
                                 │ • TradingView Lightweight Charts Candlesticks│
                                 │ • Bottom Volume Histogram Bars (Green/Red)   │
                                 │ • Custom HTML5 Canvas Shaded Wedge Overlay   │
                                 │ • 6-Deck Cards (Trades, ML, Lots, Metrics)   │
                                 └──────────────────────────────────────────────┘
```

---

## 3. The 3 ML Models Prediction Engine

Every 5-minute candle is processed through a feature extraction pipeline and evaluated concurrently by three independent machine learning architectures:

### 1. LightGBM (Light Gradient Boosted Machine)
* **Role**: Fast non-linear decision boundary classification on tabulated technical indicators.
* **Output**: Categorical signal (`BUY`, `SELL`, or `HOLD`) accompanied by calibrated probabilities (`gbm_prob_buy` and `gbm_prob_sell`).
* **Feature Vector Inputs**:
  * RSI (14-period, 7-period)
  * Normalized MACD Histogram
  * ATR (Average True Range) & ATR Expansion Ratio
  * Bollinger Bands %B and Bandwidth
  * VWAP Distance & Price Displacement
  * Stochastic Oscillator (%K, %D)
  * Realized Volatility (20-period rolling)
  * ROC (Rate of Change, 10-period)

### 2. TCN (Temporal Convolutional Network)
* **Role**: Deep sequential neural network utilizing causal dilated convolutions and residual blocks.
* **Advantage**: Captures multi-candle temporal patterns, trends, and momentum memory without the vanishing gradient problems of traditional RNNs or LSTMs.
* **Output**: Prediction probabilities across classes: `tcn_prob_buy`, `tcn_prob_sell`, and `tcn_predicted`.

### 3. HMM (Hidden Markov Model Regime Classifier)
* **Role**: Unsupervised market regime detection using Gaussian emissions.
* **Classifies into 7 Distinct Regimes**:
  1. `markup`: Strong upward directional trend.
  2. `expansionup`: High-volatility upward expansion.
  3. `compression`: Low-volatility squeeze / consolidation before a breakout.
  4. `distributiondown`: Top rollover into initial downward momentum.
  5. `distributionup`: Bottom rollover into initial upward momentum.
  6. `expansiondown`: High-volatility downward expansion.
  7. `markdown`: Strong downward directional trend.

### 4. Ensemble Consensus Logic
A trade signal triggers **only when models reach multi-agent consensus**:
* For a **BUY Signal**: LightGBM and TCN must agree on bullish direction, confirmed by a supportive HMM regime (e.g. `markup` or `expansionup`), passing minimum probability thresholds.
* For a **SELL Signal**: Models must align in the bearish direction within `markdown` or `expansiondown` regimes.
* The signal is timestamped at the close of **Candle T**, with order entry executed at the opening tick of **Candle T + 1**.

---

## 4. 15-Minute Historical Pattern Match Engine

The system features an 18-year pattern recognition engine (`backend_engine/pattern_match_engine.py`) that matches today's price action against 99,205 historical 15-minute trading sessions from 2008 through 2026.

### How It Works:
1. **Fingerprint Vectorization**: At the 09:45 AM evaluation window, the engine extracts a 21-dimensional normalized fingerprint vector of the opening session.
2. **KD-Tree Nearest Neighbor Search**: Queries the 18-year database using cosine similarity and Euclidean distance to locate the top 5 historical analogues.
3. **Consensus Direction**: If 4 out of 5 or 5 out of 5 historical analogues trended in the same direction, a projection is generated.

### Visual Overlays Drawn on the Live Chart:
* **Line 1 (Primary Spike Projection — Solid Line)**:
  * Projects from the breakout entry point forward to the session target (15:00).
  * **20% Conservative Target Reduction**: To prevent overshooting, the drawn target is pulled back by 20%:
    $$\text{Drawn Target} = \text{Base Price} + (\text{Raw Target} - \text{Base Price}) \times 0.80$$
  * Rendered in **Solid Green (`#10b981`)** for BUY or **Solid Red (`#ef4444`)** for SELL.
  * Right-axis price tag is hidden (`lastValueVisible: false`) to keep the price scale clean.

* **Line 2 (Angular Projection — Dashed Line)**:
  * Evaluated at a fixed **0.05° angular slope** ($0.05\%$) off the horizontal breakout flatline:
    $$\text{Target}_2 = \text{Base Price} \times (1 + 0.0005) \quad \text{[BUY]}$$
    $$\text{Target}_2 = \text{Base Price} \times (1 - 0.0005) \quad \text{[SELL]}$$
  * Rendered in **Dashed Green (`#34d399`)** or **Dashed Red (`#f87171`)**.

* **Breakout Flatline**:
  * A horizontal reference line drawn at the breakout entry price in dotted slate gray (`#64748b`).

* **Semi-Transparent Shaded Wedge (`#patternFillCanvas`)**:
  * An HTML5 Canvas layer positioned directly over the chart.
  * Automatically fills the polygon wedge bounded by Line 1, Line 2, and the entry apex with a subtle gradient (`rgba(16, 185, 129, 0.20)` for BUY, `rgba(239, 68, 68, 0.20)` for SELL).
  * Synchronized seamlessly with pan and zoom events (`subscribeVisibleLogicalRangeChange`).

* **80-Point Stop Loss Invalidation**:
  * If the live spot price moves $\pm 80$ points against the entry price at any time, the pattern is invalidated, and all three lines plus the shaded wedge are **instantly cleared from the chart**.

* **Previous Day Persistence**:
  * The previous day's pattern lines remain visible on the chart during the morning session until a fresh signal arrives today, at which point the old lines are automatically cleared.

---

## 5. Dynamic Thermal Dissipation Lot Sizer

Rather than fixed position sizing, the platform implements a thermodynamic capital allocation engine inspired by **Newton's Law of Cooling**:

$$\frac{dT}{dt} = \alpha \cdot \text{PnL} - \beta \cdot (T - T_{\text{ambient}})$$

### Variables & Parameters:
* $T$: System Temperature (bounded strictly between $T_{\min} = 0.50$ and $T_{\max} = 4.40$).
* $T_{\text{ambient}} = 1.00$: Baseline equilibrium temperature.
* $\alpha = 0.015$: Heating coefficient (gains from winning trades increase temperature).
* $\beta = 0.40$: Thermal dissipation coefficient (drawdowns and losses dissipate heat back toward equilibrium).

### Dynamic Lot Allocation:
| Temperature ($T$) Range | Sizing | Total Quantity | Behavioral Regime |
| :--- | :---: | :---: | :--- |
| $T < 1.50$ | **1 Lot** | 65 QTY | Base / Conservative (Cooling / Post-Drawdown) |
| $1.50 \le T < 2.50$ | **2 Lots** | 130 QTY | Moderate Confidence (Heating) |
| $2.50 \le T < 3.50$ | **3 Lots** | 195 QTY | High Conviction (Hot Momentum) |
| $T \ge 3.50$ | **4 Lots** | 260 QTY | Maximum Compounding (Peak Run) |

---

## 6. Active Trade Management & Concurrent Slots

The execution engine (`backend_engine/live_feed_service.py`) manages real-time paper positions with institutional execution rules:

* **Max Concurrent Positions**: Supports up to **4 concurrent trade slots**.
* **Candle Timing**: Signal generated at **Candle T close**; entry filled at **Candle T + 1 open**.
* **Fixed Stop Loss**: 60 index points.
* **Target Profit**: Scaled dynamically from ML model confidence.
* **End-of-Day (EOD) Exit**: Forceful square-off at **15:10 IST** to eliminate overnight gap risk.
* **Smart Time Filter**: Trade entry is restricted to peak-liquidity market hours:
  * Morning Window: **09:30 to 11:00 IST**
  * Afternoon Window: **13:00 to 14:30 IST**
  * Signals generated outside these windows are marked as `TIME FILTERED` and not executed, protecting capital from midday chop.

---

## 7. Institutional Risk & Performance Metrics Dashboard

The metrics engine calculates performance strictly on **entered/executed trades** (excluding time-filtered trades that were skipped):

| Metric | Current Value | Definition / Calculation |
| :--- | :--- | :--- |
| **Win Rate** | **50.0%** | $\frac{\text{Wins}}{\text{Total Executed}} \times 100$ (51 Wins / 51 Losses across 102 trades) |
| **Profit Factor** | **1.73** | $\frac{\text{Gross Profit (7,217.60 pts)}}{\text{Gross Loss (4,166.65 pts)}}$ |
| **Net Points PnL** | **+3,050.95 pts** | Cumulative points across all entered trades |
| **Net INR PnL** | **+₹1,98,311.75** | Cumulative rupee profit ($65 \times \text{Lots} \times \text{Points}$) |
| **Max Drawdown (pts)** | **-2,243.55 pts** | Maximum peak-to-trough decline in cumulative points |
| **Max Drawdown (INR)**| **-₹1,45,830.75**| Maximum peak-to-trough drawdown in rupees |
| **Recovery Factor** | **1.36x** | $\frac{\text{Net PnL}}{\text{Max Drawdown}}$ |
| **Average Lot Size** | **2.17x Lots** | Average contract size allocated by the Thermal Sizer |
| **Executed Trades** | **102 Trades** | Number of trades actively entered in the market |
| **Filtered Trades** | **225 Trades** | Low-probability signals filtered out by time rules |

---

## 8. Real-Time Market Breadth & Sentiment Meter

Located in the top header bar, the Market Breadth system computes live statistics across all **50 NIFTY constituent stocks**:

* **Advances / Declines / Unchanged Counters**: Track stock movements in real time.
* **Visual Ratio Bar**: Dual-color linear gauge (Green advances vs Red declines).
* **Top Movers Ticker Chips**: Real-time display of the top 3 gainers and top 3 losers with percentage changes.
* **Sentiment Indicator**:
  * $\ge 32$ Advances $\to$ **STRONGLY BULLISH** (Green)
  * $27 - 31$ Advances $\to$ **MODERATELY BULLISH** (Emerald)
  * Balanced $\to$ **NEUTRAL / BALANCED** (Slate Gray)
  * $27 - 31$ Declines $\to$ **MODERATELY BEARISH** (Rose)
  * $\ge 32$ Declines $\to$ **STRONGLY BEARISH** (Red)

---

## 9. Interactive Candlestick & Volume Charting

Built using TradingView's official **Lightweight Charts** library (downloaded and cached locally to prevent CDN outages):

### Candlestick Series:
* 5-minute interval candles with green/red bodies and wicks.
* Time scale automatically handles Indian Market Hours (09:15 to 15:30 IST).
* Right price scale margins set to `top: 0.08, bottom: 0.22` (reserving upper ~78% for candlesticks).

### Volume Histogram Bars (Anchored at Bottom):
* **Placement**: Volume series scale margins set to `top: 0.80, bottom: 0.0` (anchored strictly to the bottom 20% of the pane).
* **Zero Overlap**: Candlesticks and volume bars never overlap.
* **Directional Coloring**:
  * Bullish candle (`close >= open`): Semi-transparent green bar (`rgba(16, 185, 129, 0.50)`).
  * Bearish candle (`close < open`): Semi-transparent red bar (`rgba(244, 63, 94, 0.50)`).
* **Real-Time Forming Candle**: Dynamically updates volume bar height and color with every incoming tick.
* **Crosshair Inspection**: Hovering the cursor over any candle instantly updates the `#legV` readout in the top chart legend with formatted constituent volume (e.g. `1.24M`).

### Event Marker Chips:
* `BUY Signal` (T) — Green arrow up
* `SELL Signal` (T) — Red arrow down
* `ENTRY POINT` (T+1) — Blue circle
* `TP EXIT` — Green target
* `SL EXIT` — Red cross
* `EOD EXIT` — Amber clock

---

## 10. Right-Column Deck Structure & Layout

The right-hand panel provides real-time data cards arranged in exact prioritized sequence:

| Order | Card Title | ID | Category | Key Contents |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Active Trade Tracker** | `#cardActivePos` | Trades | Up to 4 active trade cards, entry price, current LTP, live points PnL, live ₹INR PnL, SL 60 level, TP level. |
| **2** | **3 ML Models Prediction Engine** | `#cardModels` | Models | Multi-model consensus badge, LightGBM Buy/Sell probabilities & bar, TCN Buy/Sell probabilities & bar, HMM regime badge. |
| **3** | **Dynamic Lot Sizer** | `#cardThermalSizer`| Trades | Temperature $T$, thermal status, interactive visual gauge, active lot chips (1x, 2x, 3x, 4x). |
| **4** | **Technical Features** | `#cardFeatures` | Models | RSI, ATR, MACD, Bollinger Bands, Volume, VWAP, plus live candle timestamp badge (`#featTimeBadge`). |
| **5** | **Performance & Risk Metrics**| `#cardMetrics` | Trades | Win rate %, win/loss count, profit factor, max DD (pts & INR), recovery factor, average lot size, executed vs filtered trades. |
| **6** | **Live Tick Board & Trades History**| `#cardTickBoard`<br>`#cardTradeHistory` | Data | Dual-tab view: streaming tick-by-tick constituent data table + complete closed trades history log. |

### Quick Filter Toolbar:
* `⚡ All Views`: Displays all 6 cards.
* `🎯 Trades & Risk`: Focuses on Active Trades, Thermal Sizer, and Performance Metrics.
* `🤖 ML Models`: Focuses on the 3 ML Models Engine and Technical Features.
* `📊 Tables`: Focuses on Live Ticks Board and Closed Trades History.
* **Expand All / Collapse All**: One-click collapsible section control.

---

## 11. Project Directory Structure & Key Files

```text
d:\algo-trading-console\44AAA_risk_ok\
│
├── app.py                             # Main server entrypoint (TA-Lib check + Uvicorn runner on port 7860)
├── requirements.txt                   # Complete Python dependencies list
├── README.md                          # Comprehensive A-to-Z platform documentation (this file)
│
├── 243A/                              # Strategy & Pre-Trained ML Models Directory
│   ├── live_prediction_engine.py      # Feature extraction and multi-model inference pipeline
│   ├── strategy_243a.py               # 243A consensus decision rulebook and thresholds
│   ├── AAAback.py                     # Historical backtest runner
│   └── models/                        # Pre-trained ML model binaries
│       ├── lgbm_model.pkl             # Serialized LightGBM classifier
│       ├── tcn_model.pth              # PyTorch weights for Temporal Convolutional Network
│       └── hmm_model.pkl              # Fitted Hidden Markov Model regime classifier
│
├── backend_engine/                    # Core Backend Trading & Streaming Services
│   ├── live_feed_service.py           # Angel One SmartWebSocketV2 streaming, 4-slot trade tracker, candle store
│   ├── pattern_match_engine.py        # 15M 18-year pattern matcher, dual lines, 80-pt SL, canvas wedge
│   ├── thermal_dissipation_sizer.py   # Thermodynamic Newton's cooling lot sizing engine
│   ├── web_app.py                     # FastAPI REST API endpoints, routing, and template serving
│   ├── live_dryrun.py                 # Historical candle sync and database gap filler
│   ├── paper_trade_engine.py          # Paper execution sandbox simulator
│   ├── auth.py                        # JWT authentication and user session management
│   ├── users_db.py                    # SQLite database for user credentials and settings
│   ├── old data.csv                   # Historical NIFTY 50 5-minute database (2008–2026)
│   ├── model signal.csv               # Historical model signals and trades log
│   └── pattern_15m_cache.npz          # Pre-computed 99,205 pattern vectors for sub-second KD-Tree search
│
├── ui_ux/                             # User Interface & Frontend Templates
│   ├── templates/
│   │   ├── console.html               # Main institutional dark-theme trading console (Active)
│   │   └── index.html                 # Legacy dashboard interface
│   └── static/
│       └── lightweight-charts.js      # Locally cached TradingView Lightweight Charts v4.1.1 library
│
└── model_2024_25/                     # Validation datasets and benchmark logs
```

---

## 12. Safety, Paper Trading & Confidentiality Guidelines

1. **Strict Paper Trading Sandbox**:
   * The platform operates in simulated paper trading mode (`paper_trading = True`).
   * **Zero Real Capital Risk**: Orders are recorded and managed in simulated sandbox memory without routing real orders to exchanges.
2. **Credential Confidentiality**:
   * Angel One client keys and credentials must remain strictly confidential. Never commit raw passwords, TOTP secrets, or passkeys to public version control.
3. **Database & Settings Exclusions**:
   * `users.db`, `backend_engine/settings.json`, and temporary data buffers are excluded via `.gitignore` to protect user privacy.

---

## 🚀 Quick Verification Command Summary

```powershell
# 1. Navigate
cd d:\algo-trading-console\44AAA_risk_ok

# 2. Run
python app.py

# 3. Open Browser
Start-Process "http://127.0.0.1:7860/"
```

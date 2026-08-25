# ⚡ Algo Trading Console (Nifty Pro)

An enterprise-grade algorithmic trading console, machine learning decision engine, and real-time execution platform built for Nifty 50 spot & options trading. Powered by **Angel One SmartAPI**, **FastAPI**, **Lightweight Charts**, and a multi-model ML ensemble (**HMM**, **LightGBM**, **TCN**, **Random Forest**).

---

## 🌟 Architecture & Core Features

### 1. 📡 24/7 Centralized Angel One Feed (0% Rate Limit Risk)
* **Single Master WebSocket Feed**: Only **1 master central connection** (`AAAE696417`) connects to Angel One SmartConnect WebSocket stream.
* **Sub-1ms Data Relay**: Incoming tick-by-tick Nifty 50 prices and volume are broadcasted to all logged-in user sessions in real-time without duplicate API calls or rate-limit bans (HTTP 429).
* **Multi-User Isolation**: Every registered account maintains its own isolated paper trading / live trading portfolio (`PaperTradeEngine`), position tracking (`data/active_positions.json`), and strategy decision logs while drawing market feeds from the central engine.

### 2. 🧠 Multi-Model ML Strategy Consensus (243A & LONGPING)
* **243A Consensus Model**: Multi-layer ensemble combining **Hidden Markov Models (HMM)**, **LightGBM**, **TCN**, and **Random Forest** with unified risk rules (Stop Loss, Take Profit, trailing stops).
* **LONGPING Strategy**: Positional trend-following model ported directly from TradingView Pine Script for overnight holding.

### 3. 🧬 HMM Volatility Regime Channels (15-Candle Projections)
* **Dynamic Drift & Volatility Envelopes**: Renders 15-candle forward projection upper/lower channel envelopes on Lightweight Charts based on real-time Hidden Markov Model volatility state (`markup`, `markdown`, `compression`, `expansionup`, `expansiondown`).
* **Persistence & History**: Regime transition anchor points are stored permanently in `backend_engine/regime_history.json`.

### 4. ⚡ Pattern Matcher Acceleration (< 10ms Response)
* **Historical Day Fingerprints**: Self-growing pattern library (`pattern_library.json`) storing 20-period Normalized Price, Volatility Ratio, RSI, and Volume Expansion vectors.
* **Instant Top-3 Similarity Matches**: Computes cosine/Euclidean vector similarity to instantly output the **Top 3 Historical Matching Days** and predicted EOD direction.

### 5. 📱 100% Mobile Responsive Dashboard & Touch Bottom Nav
* **Touch Bottom Navigation Bar**: 5 dedicated touch tabs (**Chart**, **Strategy**, **Positions**, **Pattern**, **Logs**) for seamless navigation on smartphones.
* **Horizontal Touch-Scrollable Controls Toolbar**: Utility buttons (`Go to Date`, `Download Data`, `Sync 72 Candles`, `Load More History`) sit in a horizontal touch toolbar so the chart canvas is 100% unobstructed.

---

## 📂 Project Architecture

```text
├── backend_engine/             # Core Backend Services
│   ├── web_app.py              # FastAPI REST API & status web server
│   ├── live_dryrun.py          # Session Manager, Central Feed & gap filler
│   ├── angel_ws_handler.py     # Centralized Angel One WebSocket client
│   ├── paper_trade_engine.py   # Dryrun sandbox execution engine
│   ├── pattern_library.json    # Instant JSON pre-computed fingerprints
│   ├── regime_history.json     # Persistent HMM regime change history
│   ├── users_db.py             # SQLite authentication register
│   └── old data.csv            # Consolidated Nifty 5-Min OHLCV database
│
├── 243A/                       # ML Consensus Models Strategy Folder
│   ├── models/                 # Pretrained Pickles & Scalers (TCN, LGBM, HMM)
│   ├── AAAback.py              # 243A 6-Month Backtest Runner
│   └── strategy_243a.py        # 243A Entry/Exit Decision Rulebook
│
├── longpine/                   # LONGPING Trend-Following Strategy Folder
│   ├── longping_original.pinescript # Original Pinescript source
│   └── backtest_runner.py      # LONGPING Backtest Runner
│
├── ui_ux/                      # Frontend templates & chart files
│   └── templates/index.html    # Glassmorphism Mobile-Responsive HTML5 Dashboard
│
├── app.py                      # Root Application Launcher
└── requirements.txt            # Python Dependencies
```

---

## 🛠️ Quick Start & Installation

### Local Running:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rushit16102004/algo-trading-console.git
   cd algo-trading-console
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Local Application**:
   ```bash
   python app.py
   ```

4. **Open in Browser**:
   Navigate to [http://localhost:7860](http://localhost:7860).

---

## ☁️ Oracle VPS 24/7 Cloud Deployment

The application runs continuously on Oracle Cloud VPS using systemd daemon (`algo-console.service`) and automated GitHub Actions CI/CD deployment (`.github/workflows/deploy.yml`).

- 🌐 **Live Cloud Server**: [http://161.118.186.178:7860](http://161.118.186.178:7860)

---

## 🛡️ Security & Privacy
- **Private Repository Ready**: Supported out of the box with zero setup.
- **Credential Protection**: Angel One keys are managed via environment variables (`.env`) and GitHub Encrypted Secrets with safe fallback handlers.

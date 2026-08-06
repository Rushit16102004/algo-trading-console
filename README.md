# AI-Based Algorithmic Trading System

An advanced algorithmic trading console and execution engine supporting multi-user registration, historical backtesting, real-time paper trading (dryrun), and live tick-by-tick charting using the Angel One SmartAPI.

---

## 🚀 Key Features

* **Dual-Strategy Engine**:
  * **243A Consensus Strategy**: Multi-model machine learning architecture combining **Temporal Convolutional Networks (TCN)**, **LightGBM**, and **Hidden Markov Models (HMM)** with unified risk management rules (SL, TP, and trailing exits).
  * **ZFTF Strategy (Longpine)**: High-frequency trend-following model ported directly from TradingView Pine Script.
* **Auto-Recovery Historical Candle Sync**: On server startup, the system automatically detects database gaps since the last entry in `old data.csv` and queries the missing 5-minute candles using authenticated connection keys (or fallback developer credentials) to keep the records contiguous.
* **5-Minute Free Trial**: A built-in user trial allows new users to view live feeds instantly by auto-authenticating with developer credentials. After 5 minutes, an interface blocker requests sign-in or signup to release developer bandwidth.
* **Ultra-Fast RAM Caching**: Precalculates and loads 1,128 markers for 243A and 192 markers for ZFTF directly into memory on boot for instant chart navigation and switching (under 2ms).
* **Live WebSocket Integration**: Aggregates tick-by-tick market data and volume from Nifty Future tokens to compile real-time 5-minute candles.

---

## 📂 Project Architecture

```text
├── 243A/                       # ML Consensus Models Strategy Folder
│   ├── models/                 # Pretrained Pickles & Scalers (TCN, LGBM, HMM)
│   ├── AAAback.py              # 243A 6-Month Backtest Runner
│   ├── strategy_243a.py        # 243A Entry/Exit Decision Rulebook
│   └── backtest_results.csv    # 6-Month simulation trades log
│
├── longpine/                   # ZFTF Trend-Following Strategy Folder
│   ├── zftf_original.pinescript# Original Pinescript source
│   ├── backtest_runner.py      # ZFTF 6-Month Backtest Runner
│   └── backtest_results.csv    # 6-Month simulation trades log
│
├── backend_engine/             # Core Backend Services
│   ├── web_app.py              # FastAPI REST & status web servers
│   ├── live_dryrun.py          # Session Manager, WebSockets, & gap filler
│   ├── paper_trade_engine.py   # Dryrun sandbox executor
│   ├── users_db.py             # SQLite authentication register
│   └── old data.csv            # Consolidated Nifty 5-Min OHLCV database
│
├── ui_ux/                      # Frontend templates, animations, & chart files
│   └── templates/index.html    # Glassmorphism HTML5/JS Dashboard
│
└── requirements.txt            # Python Dependencies
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rushit16102004/AI-Based-Algorithmic-Trading-System.git
   cd AI-Based-Algorithmic-Trading-System
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Trading Console**:
   ```bash
   uvicorn backend_engine.web_app:app --host 127.0.0.1 --port 8050
   ```

4. **Open in Browser**:
   Navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050).

---

## 🛡️ Security & Gitignore Guidelines
To protect user credentials and sandbox data, the following local components are strictly excluded from git tracking:
* `users.db` (Local SQLite database storing user configurations)
* `backend_engine/settings.json` (Active API login credentials)
* `data/` and `logs/` (Historical ticks logs and paper trades)
* `.env` and `__pycache__/`

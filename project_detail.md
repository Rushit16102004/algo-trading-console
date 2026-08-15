# Project Details: Algo Trading Console

This document provides a comprehensive overview of the Algorithmic Trading Console, detailing the real-world problem statements, implementation details, core machine learning models, and empirical backtesting performance metrics.

---

## 0. About the Project & Core Features
The **Algo Trading Console** is an advanced algorithmic trading execution desk and analytics dashboard. It is designed to run locally or on private cloud instances to monitor the Nifty 50 Index Spot, calculate machine learning model predictions, and execute trades in paper-trading (dryrun) or live configurations using the **Angel One SmartAPI**.

### Core Features:
* **Multi-User Live Desk**: Supports multiple concurrent trading sessions with independent state management.
* **Consensus-Driven Signal Execution (243A)**: Combines deep learning, gradient boosting, and regime filtering to make highly protected intraday trading decisions.
* **Auto-Recovery Historical Sync**: Automatically checks your local CSV database on startup, identifies data gaps, and queries missing 5-minute Nifty Spot candles via the Angel One API to ensure data contiguity.
* **Interactive Charting Dashboard**: Mounts a FastAPI backend with a lightweight HTML5 canvas chart to render real-time tick feeds, historical candles, entry/exit markers, and open positions.
* **Thermal Dissipation Position Sizer**: Features a risk-based position sizer that adjusts lot sizes dynamically based on recent trading temperature (win/loss patterns).

---

## 1. Problem Statement
In retail algorithmic trading, traders face two primary points of failure:

### Real-Life Problem:
Retail traders lose money because they trade emotionally, lack risk management, and struggle to identify true market momentum. While basic indicators (like RSI or moving averages) work occasionally, they fail during market regime shifts (e.g., transitioning from high-volatility expansion to low-volatility consolidation), leading to account blowouts.

### The Reality We Solve:
1. **False Breakout Traps**: Filtering out false entry signals during trend-less compression and distribution phases.
2. **Delayed Execution & Signal Lag**: Traditional indicators lag. We solve this by using deep learning to predict forward price movements based on multi-scale structural patterns.
3. **Improper Position Sizing**: Traders typically keep constant lot sizes. We solve this by dynamically reducing capital exposure (lot size scaling) when the strategy is experiencing a draw-down streak.

---

## 2. Why It Needed Solving (How it Helps Others)
1. **Eliminating Human Bias**: By delegating entry, exit, Stop Loss (SL), and Take Profit (TP) parameters to a machine learning consensus pipeline, emotional trading is eliminated.
2. **Protecting Capital during Bad Regimes**: The integration of a Hidden Markov Model (HMM) actively blocks trading in dangerous regimes (e.g., distribution down for buys, markup for sells), preserving capital when market conditions are adverse.
3. **Low-Bandwidth Deployment**: Provides retail traders with institutional-grade risk models and execution speeds without requiring high-cost servers.

---

## 3. How It Was Implemented
The project is built around a **tri-model consensus architecture (Strategy 243A)** that runs on 5-minute Nifty index candles.

```
                  +-----------------------------------+
                  |      5-Min Nifty Candle Feed      |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      Feature Generation Block     |
                  |  (SMA, BB, Stoch, CCI, ATR, Vol)  |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
+-----------------------+ +-----------------------+ +-----------------------+
|  LightGBM Classifier  | |   TCN Deep Learning   | |  HMM Regime Decoder   |
| (Directional Trend)   | | (Temporal Patterns)   | | (Market Phase Filter) |
+-----------------------+ +-----------------------+ +-----------------------+
            |                       |                       |
            +-----------+-----------+                       |
                        | (Consensus Match?)                |
                        v                                   v
             +---------------------+             +---------------------+
             |  Consensus Signal   |------------>| Regime Filter Check |
             |   (BUY or SELL)     |             +---------------------+
             +---------------------+                        |
                                                            v
                                                 +---------------------+
                                                 | Final Trade Trigger |
                                                 +---------------------+
```

### Core Models:
1. **LightGBM (Gradient Boosting)**:
   * Processes standard technical indicator features (CCI, Stochastic, BB Width, SMA deviations) to identify directional price trends.
   * Optimizes binary classification (upward vs. downward candles) with fast execution.
2. **TCN (Temporal Convolutional Network)**:
   * A PyTorch deep learning network that captures long-range temporal dependencies using causal dilated convolutions.
   * Evaluates sequential patterns across 500-candle lookback windows to predict directional signals.
3. **HMM (Hidden Markov Model)**:
   * Unsupervised regime classifier that decodes current market phases into distinct hidden states (e.g., markup, markdown, expansion, compression, distribution).
   * **The Filter**: Actively blocks BUY entries during compression, markdown, and distribution-down phases. Actively blocks SHORT entries during compression, markup, and distribution-up phases.

### Risk & Execution Logic:
* **EOD Force Exit**: Automatically closes all active intraday positions at 15:10 to avoid overnight gap risk.
* **Stop Loss (SL) & Take Profit (TP)**: Set at a strict 60-point index SL and 150-point index TP.
* **Risk Sizer**: Uses a thermodynamic temperature sizer that scales trade size (lots) down during losses and increases it up to a maximum limit during win streaks.

---

## 4. Languages Used
* **Python**: Core backend engine, machine learning models, database initialization, and data pipelines.
* **JavaScript**: Frontend interactive tick charting using Lightweight Charts library.
* **HTML & CSS**: Dashboard structure and responsive design layout.

---

## 5. ML / DL Frameworks Used
* **PyTorch**: Used for building and running the Temporal Convolutional Network (TCN) models (`model_TCN.pt`).
* **LightGBM**: Used for gradient boosted tree classification models (`model_LGBM.pkl`).
* **hmmlearn**: Used for regime detection and hidden state decoding via Hidden Markov Models (`model_HMM.pkl`).
* **scikit-learn**: Used for data pre-processing, feature scaling, and performance metric calculations.

---

## 6. Backtesting Performance Outcome (2024 - Present)
These metrics represent the performance of the **243A Consensus Model** on Nifty Option data from **January 1, 2024 to June 30, 2026** (using the Option backtest database, **excluding all 0 lot trades**):

* **Dataset**: Option Backtest (`merged_01-01-2024_to_06-30-2026_option.csv`)
* **Total Signals Generated**: 3,665 signals
* **Active Trades Executed (Lot Size > 0)**: 1,995 trades
* **Win Rate (Active)**: **`55.44%`** (Option Win Rate)
* **Total PnL (Option Net PnL)**: **`+801,170.50 INR`** (on a 100,000 INR starting capital)
* **Max Drawdown (DD)**: **`80,047.50 INR`**
* **Recovery Factor**: **`10.01`**
* **Total Nifty Spot Points PnL**: **`+23,590.25 points`** (Spot Win Rate: `50.23%`)
  *(Note: This is the raw Nifty Spot Index point return, while the Option Net PnL represents the options premium outcome)*

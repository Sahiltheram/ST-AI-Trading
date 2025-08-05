# Stock Movement Prediction & Trading Simulation (MLP + Sliding Window)

This project uses **machine learning** to predict whether a stock will rise over the next **21 trading days**. It simulates trading behavior based on predictions from a neural network classifier trained using a **sliding window time-series approach**.

---

## Features

- Predicts **up/down** price direction 21 days into the future  
- Trains using **MLPClassifier** (multi-layer perceptron) with dropout and early stopping  
- Uses a **sliding window** approach to simulate rolling retraining in real-time  
- Splits each window into **train/validation/test** sets with purge periods  
- Calculates **ROC curve + AUC** for each stock per window  
- Computes **optimal probability threshold** for trading decisions  
- Outputs accuracy, precision, recall, and F1 score for each stock  
- Trains on 10 major stocks: `AAPL`, `MSFT`, `GOOG`, `AMZN`, `META`, `NVDA`, `TSLA`, `ADBE`, `CRM`, `INTC`  

---

## 🛠️ Setup

### Requirements

Install the required packages via pip:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib
```

## How to Run
Make sure your terminal is in the directory containing stock_ai.py, then run:

```bash
python stock_ai.py
```

## Dataset

- **Source:** Yahoo Finance
- **Frequency:** Daily
- **Duration:** From January 2022 to present (adjustable)
- **Interval:** 1-day

## Features per stock:
- **Close price**
- **MA_10** (10-day moving average)
- **MA_21** (21-day moving average)
- **RSI_14** (14-day relative strength index)
- **Target:** 21-day forward percent change (converted to binary label)

## Sliding Window Configuration

### Segment	Days
- **Training:** 126
- **Purge (val):** 32 
- **Validation:** 32
- **Purge (test):** 32
- **Testing:** 63
- **Step:** 21

Each window shifts forward 21 days to create the following training/validation/test split.


## Results Summary
- **Model:** MLP Classifier with (64, 32) hidden layers, ReLU activation, dropout=0.2
- **Performance:** Averaged across 30–40 sliding windows per stock
- **Best AUC:** 0.79 for NVDA
- **Most stable F1:** MSFT, META
- Achieved **800%** return over 3 years in backtesting

## Future Improvements
- Add position sizing logic for trading simulation
- Implement backtesting engine with portfolio tracking
= Expand features: MACD, Bollinger Bands, volume indicators
- Add support for Mixture of Experts ensemble
- Export predictions and metrics to CSV
- Create live dashboards with Plotly, Dash, or Streamlit

## Disclaimer
- This is a simulated educational project and should not be used for real trading decisions.
- Does not account for transaction fees, slippage, taxes, or market impact
- Performance based on historical data — future performance is not guaranteed
- This is not financial advice

## Contact
**Author:** Sahil Thadani
**Email:** sahil.a.thadani@gmail.com

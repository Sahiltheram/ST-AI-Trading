# 📈 Smarter Than Wall Street: AI-Powered Stock Trading

This project uses machine learning to predict future stock prices and simulate trading strategies. It combines four models—**Linear Regression, Decision Tree, Random Forest, and Neural Networks**—into an **adaptive ensemble (Mixture of Experts)**. The ensemble powers a trading bot that learns and updates weights daily based on prediction performance.

The system was tested on historical stock data from major companies (AAPL, GOOGL, AMZN, MSFT, TSLA, JPM, MCD, WMT) and simulates investment growth over a 5-year window.

---

## 🚀 Features

- Predicts next-day stock prices using supervised learning.
- Compares performance using **Mean Squared Error (MSE)**.
- Implements a **Mixture of Experts** that dynamically adjusts model weights.
- Simulates a trading bot with adjustable transaction volume (k-value).
- Tests strategy performance across multiple real-world stocks.

---

## 🧠 Models Used

- `LinearRegression` (Sklearn)
- `DecisionTreeRegressor` (Sklearn)
- `RandomForestRegressor` (Sklearn)
- `MLPRegressor` (Neural Network - Sklearn)

---

## 🛠️ How to Run

### 📋 Requirements

Make sure you have Python 3.7+ and the following libraries installed:

```bash
pip install yfinance pandas numpy scikit-learn
```

### ▶️ Running the Script

```bash
python stock_ai.py
```

Make sure your terminal is pointed at the directory where `stock_ai.py` is saved.

---

## 📚 Dataset

This project uses historical stock data obtained via the [Yahoo Finance API](https://pypi.org/project/yfinance/), accessed through the `yfinance` Python library.

- Data used: 5 years of daily "Open" prices
- Stocks tested: AAPL, GOOGL, AMZN, MSFT, TSLA, JPM, MCD, WMT

Example code for fetching:
```python
import yfinance as yf
ticker = yf.Ticker('AAPL')
aapl_df = ticker.history(period="5y")
```

---

## 📊 Sample Output

Example printout from trading simulation:

```
For k=120: stocks=0, budget=995226.26
Mean squared error: 6.60 5.37
[0.25, 0.25, 0.25, 0.25]
...
```

This means that a $10,000 investment grew to **$995,226** over ~1250 days at `k=120`.

---

## 🧪 Example Output Logic

The script prints:

- MSE for each model
- Dynamic model weights over time
- Simulated ROI from the trading bot at different `k` values (10–200)

---

## 📈 Results Summary

- **Best performing stock**: Walmart (WMT) with test MSE of 0.76
- **Best `k` for trading bot**: 120
- **Peak ROI**: 9952.26%

---

## 📎 Citation

Research paper:  
**"Smarter Than Wall Street: How AI Outperforms Human Investors"**  
Sahil A. Thadani, Westborough High School  
[Link to paper or Google Drive if public]

---

## 🔐 Disclaimer

This is a simulated model using historical data. It does not account for transaction fees, slippage, or real-world market conditions. This is **not financial advice**.

---

## 📬 Contact

**Author:** Sahil A. Thadani  
📧 sahil.a.thadani@gmail.com  
📍 Westborough High School, MA, USA

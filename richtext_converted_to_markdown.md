\# 📈 Stock Movement Prediction & Trading Simulation (MLP + Sliding Window)

This project uses \*\*machine learning\*\* to predict whether a stock will rise over the next \*\*21 trading days\*\*. It simulates trading behavior based on predictions from a neural network classifier trained using a \*\*sliding window time-series approach\*\*.

\---

\## 🚀 Features

\- ✅ Predicts \*\*up/down\*\* price direction 21 days into the future

\- 🧠 Trains using \*\*MLPClassifier\*\* (multi-layer perceptron) with dropout and early stopping

\- 🔄 Uses a \*\*sliding window\*\* approach to simulate rolling retraining in real-time

\- 🧪 Splits each window into \*\*train/validation/test\*\* sets with purge periods

\- 📊 Calculates \*\*ROC curve + AUC\*\* for each stock per window

\- 🧮 Computes \*\*optimal probability threshold\*\* for trading decisions

\- 📈 Outputs accuracy, precision, recall, and F1 score for each stock

\- 🏦 Trains on 10 major stocks: \`AAPL\`, \`MSFT\`, \`GOOG\`, \`AMZN\`, \`META\`, \`NVDA\`, \`TSLA\`, \`ADBE\`, \`CRM\`, \`INTC\`

\---

\## 🛠️ Setup

\### ✅ Requirements

Install the required packages via pip:

\`\`\`bash

pip install yfinance pandas numpy scikit-learn matplotlib

▶️ How to Run

Make sure your terminal is in the directory containing stock\_ai.py, then run:

bash

Copy

Edit

python stock\_ai.py

📚 Dataset

Source: Yahoo Finance

Frequency: Daily

Duration: From January 2022 to present (adjustable)

Interval: 1-day

Features per stock:

Close price

MA\_10 (10-day moving average)

MA\_21 (21-day moving average)

RSI\_14 (14-day relative strength index)

Target: 21-day forward percent change (converted to binary label)

🧪 Sliding Window Configuration

SegmentDays

Training126

Purge (val)32

Validation32

Purge (test)32

Testing63

Step21

Each window shifts forward 21 days to create the next training/validation/test split.

📤 Sample Output

yaml

Copy

Edit

Window 1

AAPL - Optimal Threshold: 0.551, AUC: 0.733

Accuracy: 0.73, Precision: 0.71, Recall: 0.76, F1: 0.73

Window 2

MSFT - Optimal Threshold: 0.603, AUC: 0.689

Accuracy: 0.68, Precision: 0.65, Recall: 0.70, F1: 0.67

...

Average performance for NVDA:

Accuracy: 0.6810

Precision: 0.7132

Recall: 0.6326

F1: 0.6709

📁 File Structure

bash

Copy

Edit

├── stock\_ai.py # Main script for training and evaluation

├── README.md # Project documentation

└── / (root) # Run everything from the root

You can expand this with:

utils.py (for data prep functions)

models.py (to modularize classifiers)

results/ (to save evaluation metrics or plots)

📈 Results Summary

Model: MLP Classifier with (64, 32) hidden layers, ReLU activation, dropout=0.2

Performance: Averaged across 30–40 sliding windows per stock

Example:

Best AUC: 0.79 for NVDA

Most stable F1: MSFT, META

Metric definitions:

Accuracy: % of correct predictions

Precision: % of predicted gains that were correct

Recall: % of actual gains the model captured

F1 Score: Balance of precision and recall

🔮 Future Improvements

Add position sizing logic for trading simulation

Implement backtesting engine with portfolio tracking

Expand features: MACD, Bollinger Bands, volume indicators

Add support for Mixture of Experts ensemble

Export predictions and metrics to CSV

Create live dashboards with Plotly, Dash, or Streamlit

⚠️ Disclaimer

This is a simulated educational project and should not be used for real trading decisions.

Does not account for transaction fees, slippage, taxes, or market impact

Performance based on historical data — future performance is not guaranteed

This is not financial advice

👤 Contact

Author: Sahil Thadani

📧 sahil.a.thadani@gmail.com

🏫 Westborough High School, MA, USA
# Imports
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Union, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
import numpy as np

# data preparation

def download_yahoo_data(
    tickers: Union[str, List[str]],
    start: str = "2022-01-01",
    end: str = None,
    interval: str = "1d",
    group_by: str = "ticker",
    auto_adjust: bool = True,
    progress: bool = True
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Parameters:
    - tickers (str or List[str]): One or more ticker symbols (e.g., 'AAPL' or ['AAPL', 'GOOG'])
    - start (str): Start date in 'YYYY-MM-DD' format
    - end (str): End date in 'YYYY-MM-DD' format (None = today)
    - interval (str): Data interval ('1d', '1wk', '1mo', '1h', etc.)
    - group_by (str): How to group data when multiple tickers provided ('ticker' or 'column')
    - auto_adjust (bool): Adjust prices for splits/dividends
    - progress (bool): Show download progress

    Returns:
    - pd.DataFrame: DataFrame with stock price data
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        group_by=group_by,
        auto_adjust=auto_adjust,
        progress=progress
    )

    return data


# Parameters
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE', 'CRM', 'INTC']
data = download_yahoo_data(tickers)
pct_change_21_day = data[tickers].pct_change(21)
column_index= [(i, 'Close') for i in tickers]
outcomes= pct_change_21_day[column_index].dropna()
# Add 21-day percent change column for each ticker
for ticker in tickers:
    future = data[ticker]['Close'].shift(-21)
    today = data[ticker]['Close']
    pct_change = (future - today) / today
    data[(ticker, '21daychange')] = pct_change
    # Add Moving Averages
    data[(ticker, 'MA_10')] = data[ticker]['Close'].rolling(window=10).mean()
    data[(ticker, 'MA_21')] = data[ticker]['Close'].rolling(window=21).mean()

    # Add RSI (Relative Strength Index)
    delta = data[ticker]['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    data[(ticker, 'RSI_14')] = 100 - (100 / (1 + rs))


data_with_outcomes = data[tickers].iloc[:-21,:]
print(data_with_outcomes)
# Add outcomes to the DataFrame
def sliding_window_split(data, train_days, purge_days, val_days, test_days, step=21):
    """
    Yield train, validation, and test sets using a sliding window approach.

    Parameters:
    - data: DataFrame with MultiIndex columns (ticker, feature)
    - step: number of days to move the window each iteration
    """
    total_days = train_days + purge_days + val_days + purge_days + test_days
    max_start = len(data) - total_days

    for start in range(0, max_start, step):
        train = data.iloc[start : start + train_days]
        val = data.iloc[start + train_days + purge_days : start + train_days + purge_days + val_days]
        test = data.iloc[start + train_days + purge_days + val_days + purge_days : start + total_days]

        yield train, val, test

# data_clean = data_with_outcomes.dropna()

# # Convert window size
# train_days = 126
# purge_days = 32
# val_days = 32
# test_days = 63

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# def sliding_window_split(data, train_days, purge_days, val_days, test_days, step=21):
#     total_days = train_days + purge_days + val_days + purge_days + test_days
#     max_start = len(data) - total_days

#     for start in range(0, max_start, step):
#         train = data.iloc[start : start + train_days]
#         val = data.iloc[start + train_days + purge_days : start + train_days + purge_days + val_days]
#         test = data.iloc[start + train_days + purge_days + val_days + purge_days : start + total_days]
#         yield train, val, test

# # Set window sizes
# train_days = 126
# purge_days = 32
# val_days = 32
# test_days = 63
# step = 21

# all_scores = {ticker: [] for ticker in tickers}

# # Loop through the windows and train model each time
# for i, (train, val, test) in enumerate(sliding_window_split(data_clean, train_days, purge_days, val_days, test_days, step=step)):
#     print(f"Window {i+1}")
    
#     # Feature/target selection — use all features except '21daychange'
#     X_train = train.drop(columns=[(ticker, '21daychange') for ticker in tickers], axis=1)
#     y_train = pd.concat([train[(ticker, '21daychange')] for ticker in tickers], axis=1)

#     X_test = test.drop(columns=[(ticker, '21daychange') for ticker in tickers], axis=1)
#     y_test = pd.concat([test[(ticker, '21daychange')] for ticker in tickers], axis=1)

#     X_train.columns = ['_'.join(col) for col in X_train.columns]
#     X_test.columns = ['_'.join(col) for col in X_test.columns]
#     y_train.columns = [f'{ticker}_21daychange' for ticker in tickers]
#     y_test.columns = [f'{ticker}_21daychange' for ticker in tickers]

#     for df in [y_train, y_test]:
#         for ticker in tickers:
#             df[f'{ticker}_target'] = (df[f'{ticker}_21daychange'] > 0.10).astype(int)   

#  # Train classifier per ticker
#     for ticker in tickers:
#         model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
#         model.fit(X_train, y_train[f'{ticker}_target'])
#         preds = model.predict(X_test)

#         acc = accuracy_score(y_test[f'{ticker}_target'], preds)
#         prec = precision_score(y_test[f'{ticker}_target'], preds, zero_division=0)
#         rec = recall_score(y_test[f'{ticker}_target'], preds, zero_division=0)
#         f1 = f1_score(y_test[f'{ticker}_target'], preds, zero_division=0)

#         all_scores[ticker].append({
#             "Accuracy": acc,
#             "Precision": prec,
#             "Recall": rec,
#             "F1": f1
#         })


# # For each ticker, calculate average metrics
# for ticker in tickers:
#     print(f"Average performance for {ticker}:")
#     scores = all_scores[ticker]
#     avg_scores = {
#         metric: sum(score[metric] for score in scores) / len(scores)
#         for metric in scores[0]
#     }
#     for metric, value in avg_scores.items():
#         print(f"{metric}: {value:.4f}")
#     print()
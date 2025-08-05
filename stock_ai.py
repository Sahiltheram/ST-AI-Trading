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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    future = data[ticker]['Close'].shift(-1)
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

# Add outcomes to the DataFrame

data_clean = data_with_outcomes.dropna()

# Convert window size
train_days = 126
purge_days = 32
val_days = 32
test_days = 63

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def sliding_window_split(data, train_days, purge_days, val_days, test_days, step=21):
    total_days = train_days + purge_days + val_days + purge_days + test_days
    max_start = len(data) - total_days

    for start in range(0, max_start, step):
        train = data.iloc[start : start + train_days]
        val = data.iloc[start + train_days + purge_days : start + train_days + purge_days + val_days]
        test = data.iloc[start + train_days + purge_days + val_days + purge_days : start + total_days]
        yield train, val, test

# Set window sizes
train_days = 126
purge_days = 32
val_days = 32
test_days = 63
step = 21

all_scores = {ticker: [] for ticker in tickers}

# --- Simulation Variables ---
initial_capital = 100000  # Starting money for the bot
bot_portfolio_value = [initial_capital]
cash_on_hand = initial_capital
buy_prices = {ticker: None for ticker in tickers}
# To store the number of shares held for each stock
shares_held = {ticker: 0 for ticker in tickers}
last_sell_date = {ticker: None for ticker in tickers}
last_buy_date = {ticker: None for ticker in tickers}

# Keep track of the daily close prices for calculation
daily_close_prices_sim = pd.DataFrame() # To store close prices of tickers during test periods
simulation_dates = []

# Download S&P 500 data for comparison
spy_data = download_yahoo_data('SPY', start=data.index.min().strftime('%Y-%m-%d'), end=data.index.max().strftime('%Y-%m-%d'))
spy_returns = spy_data['SPY']['Close'].pct_change().dropna()
spy_cumulative_returns = (1 + spy_returns).cumprod()

# Loop through the windows and train model each time
for i, (train, val, test) in enumerate(sliding_window_split(data_clean, train_days, purge_days, val_days, test_days, step=step)):
    print(f"Window {i+1}")
    
    # Feature/target selection — use all features except '21daychange'
    X_train = train.drop(columns=[(ticker, '21daychange') for ticker in tickers], axis=1)
    y_train = pd.concat([train[(ticker, '21daychange')] for ticker in tickers], axis=1)

    X_test = test.drop(columns=[(ticker, '21daychange') for ticker in tickers], axis=1)
    y_test = pd.concat([test[(ticker, '21daychange')] for ticker in tickers], axis=1)

    X_train.columns = ['_'.join(col) for col in X_train.columns]
    X_test.columns = ['_'.join(col) for col in X_test.columns]
    y_train.columns = [f'{ticker}_21daychange' for ticker in tickers]
    y_test.columns = [f'{ticker}_21daychange' for ticker in tickers]

    for df_target in [y_train, y_test]: # Renamed df to df_target to avoid conflict
        for ticker in tickers:
            df_target[f'{ticker}_target'] = (df_target[f'{ticker}_21daychange'] > 0.00).astype(int)   

    # Store the actual close prices for the test period for simulation
    current_test_close_prices = test[[(ticker, 'Close') for ticker in tickers]]
    current_test_close_prices.columns = [ticker for ticker, _ in current_test_close_prices.columns]
    daily_close_prices_sim = pd.concat([daily_close_prices_sim, current_test_close_prices])
    
    # Train classifier per ticker
    models = {} # To store trained models for each ticker in the current window
    optimal_thresholds = {} # To store optimal thresholds for each ticker

    for ticker in tickers:
        if len(np.unique(y_train[f'{ticker}_target'])) < 2:
            print(f"Skipping {ticker} in window {i+1} due to only one class in training data.")
            continue

        model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        model.fit(X_train, y_train[f'{ticker}_target'])
        models[ticker] = model # Store the trained model

        # Predict probabilities
        y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (positive class)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test[f'{ticker}_target'], y_probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds[ticker] = optimal_threshold # Store the optimal threshold

        # Optional: print or plot
        print(f"{ticker} - Optimal Threshold: {optimal_threshold:.3f}, AUC: {auc(fpr, tpr):.3f}")
        
        # Apply optimal threshold to convert proba → binary prediction for evaluation
        preds = (y_probs >= optimal_threshold).astype(int)

        # Scores
        acc = accuracy_score(y_test[f'{ticker}_target'], preds)
        prec = precision_score(y_test[f'{ticker}_target'], preds, zero_division=0)
        rec = recall_score(y_test[f'{ticker}_target'], preds, zero_division=0)
        f1 = f1_score(y_test[f'{ticker}_target'], preds, zero_division=0)

        all_scores[ticker].append({
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })
    
    # --- Simulation Logic for the current test window ---
    test_close_prices = data_clean.loc[X_test.index, [(t, 'Close') for t in tickers]]
    test_close_prices.columns = [t for t, _ in test_close_prices.columns] # Flatten column names for easier access
    for date_idx in range(len(X_test)):
        current_date = X_test.index[date_idx]
        simulation_dates.append(current_date)
        
        X_current_day = X_test.loc[[current_date]] # Need to use loc with list to preserve DataFrame structure

            # Drop columns that are not features (i.e. '21daychange' which is the target)
        X_current_day_features = X_current_day.reindex(columns=X_train.columns, fill_value=0) # Fill missing with 0 or NaN
        print(X_current_day_features.tail(5))

        daily_up_probabilities = {} 

        # Calculate current portfolio value (before potential trades)
        current_portfolio_value = cash_on_hand
        for ticker in tickers:
            if shares_held[ticker] > 0:
                # Use .get() with a default value of 0 to handle missing dates gracefully
                current_price = test_close_prices.loc[current_date, ticker] if current_date in daily_close_prices_sim.index else 0
                current_portfolio_value += shares_held[ticker] * current_price

        
        # Store portfolio value for the day (before trades are executed based on previous day's signal)
        # Note: For accurate backtesting, trade execution typically happens at the OPEN of the NEXT day
        # following the signal. For simplicity here, we're assuming end-of-day close for current day's value
        # and then making decisions for the next day, using the next day's open/close for execution.
        # Here we will simplify and assume we trade based on today's prediction and the stock price used for value calculation is also today's close.
        # This is a simplification but good for illustrative purposes.

        # --- Decision Making for the next day's trade (based on current day's data) ---
        # The model's prediction is for the NEXT 21 days, so a signal for today implies a trade for tomorrow's price.
        # For a simplified daily simulation, we'll assume the model predicts for the *next* day's direction.
        # However, your current target is 21-day change. Let's adapt the simulation to that.

        # We're making a prediction for the 21-day change starting *from* the current day.
        # So, if we buy today based on a prediction of "up" for the next 21 days, we hold for 21 days.
        # This makes the simulation more complex for daily tracking of portfolio.

        # Let's simplify the simulation slightly for clarity based on your existing 21-day prediction:
        # If the model predicts UP for the *next* 21 days, we "buy and hold" for that 21-day period.
        # If the model predicts DOWN for the *next* 21 days, we "sell" if we hold, and do not buy.

        # This requires tracking when we bought a stock and when the 21-day period ends.
        # Let's refine the simulation to be more aligned with a daily trading bot, but using your
        # 21-day prediction as a "long-term sentiment".

        # Let's change the target to next day's movement for simplicity in daily trading simulation.
        # Or, we stick to 21-day target and interpret a "buy" signal as holding for 21 days,
        # and a "sell" signal as exiting any position.

        # Given your target is `21daychange`, let's interpret the strategy:
        # "If the bot predicts the stock will go up the next day, and sell if it will go down."
        # This implies a daily rebalancing. However, your target variable `21daychange` is for a 21-day horizon.
        # To align with "next day" trading, we would need to redefine `y_train` and `y_test` to be `shift(-1)` for next day's change.

        # Let's adjust the target creation slightly for a more direct "next day" prediction for the simulation.
        # We will keep your existing 21-day change as a feature, but redefine the *actual* outcome for the simulation.

        # Rerun data preparation slightly to generate a 'next_day_change' for the simulation
        # It's better to process this at the data preparation stage.
        # For the purpose of simulating a *daily* trading bot that acts on next day's price,
        # the target should be the next day's price change.

        # We need to re-run the data prep to get a 1-day change target for simulation.
        # Let's assume for the simulation, you actually want to predict `(future - today) / today` for `shift(-1)`
        # This is a crucial point for a daily trading bot.

        # Since your initial code already has `data[(ticker, '21daychange')] = pct_change`,
        # where `pct_change` is indeed `(future - today) / today` with `future = data[ticker]['Close'].shift(-1)`,
        # this means your `21daychange` column is already the *next day's* percentage change, not 21-day.
        # The variable `pct_change_21_day` in the beginning is separate and not used in your feature/target creation `data[(ticker, '21daychange')]`.
        # This is good! It means your target is indeed for the next day. The naming is a bit confusing.

        # So, `y_train[f'{ticker}_21daychange']` and `y_test[f'{ticker}_21daychange']` actually contain the *next day's* percentage change.
        # And `y_test[f'{ticker}_target'] = (df[f'{ticker}_21daychange'] > 0.00).astype(int)` is correctly predicting next day's UP/DOWN.

        # Let's use this.fcash

        # Make predictions for the current day's close price, for the *next* day's movement.
        # We use X_test[date_idx:date_idx+1] to get features for the current day.

        for ticker in tickers:
            model = models.get(ticker)
            optimal_threshold = optimal_thresholds.get(ticker)

            if model is None or optimal_threshold is None:
                daily_up_probabilities[ticker] = 0.5 # Neutral if no model/threshold
                continue
                # Get the probability of the positive class (class 1, i.e., "up")
            prob_up = model.predict_proba(X_current_day_features)[:, 1][0]
            daily_up_probabilities[ticker] = prob_up
            
 # --- Selling Logic ---
        # For selling, you might stick to a threshold-based decision, or also use confidence.
        # A simple approach: if probability of "up" is below 0.5 (or below the optimal_threshold) AND you hold shares, sell.
        for ticker in tickers:
            current_price_for_ticker = test_close_prices.loc[current_date, ticker]
            if current_price_for_ticker <= 0:
                continue

            # Ensure model and threshold exist for current ticker
            model = models.get(ticker)
            optimal_threshold = optimal_thresholds.get(ticker)

            if shares_held[ticker] > 0 and model is not None and optimal_threshold is not None and last_buy_date[ticker] != current_date:
                # Get the probability of "up" for selling decision
                prob_up_for_sell = model.predict_proba(X_current_day_features)[:, 1][0]

                # Sell if probability of going up is below optimal threshold (or a lower fixed threshold like 0.5)
                # You might set a specific "sell_threshold" that's different from the buy threshold
                if prob_up_for_sell < optimal_threshold: # Or, a more aggressive `prob_up_for_sell < 0.5`
                    revenue = shares_held[ticker] * current_price_for_ticker
                    cash_on_hand += revenue
                    print(f"{current_date.strftime('%Y-%m-%d')}: Sold {shares_held[ticker]} of {ticker} at {current_price_for_ticker:.2f} (Confidence: {prob_up_for_sell:.2f}). Cash: {cash_on_hand:.2f}")
                    shares_held[ticker] = 0
                    last_sell_date[ticker] = current_date

        # Execute trades based on predictions
        buy_candidates = {}
        for ticker, prob_up in daily_up_probabilities.items():
            current_rsi = data.loc[current_date, (ticker, 'RSI_14')]

            if prob_up > optimal_thresholds.get(ticker, 0.5): # Use the optimal threshold for initial screening
                 # And check if not currently holding (simple re-entry) and cooldown passed
                days_since_last_sell = (current_date - last_sell_date[ticker]).days if last_sell_date[ticker] else 9999
                if shares_held[ticker] == 0 and current_rsi < 60 and days_since_last_sell >= 5: # Assuming current_rsi is available (you'd need to fetch it)
                    buy_candidates[ticker] = prob_up

        total_confidence_score = sum(prob - optimal_thresholds.get(t, 0.5) for t, prob in buy_candidates.items()) # Sum of excess confidence above threshold

        # Ensure we have a positive total confidence and cash to invest
        if total_confidence_score > 0 and cash_on_hand > 0:
            for ticker, prob_up in buy_candidates.items():
                current_price_for_ticker = test_close_prices.loc[current_date, ticker]

                
                if current_price_for_ticker <= 0:
                    continue

                # Calculate investment amount based on confidence relative to total confidence
                # Scale by the excess probability over the threshold
                confidence_weight = (prob_up - optimal_thresholds[ticker]) / total_confidence_score
                # print (f"Ticker: {ticker}, Confidence Weight: {confidence_weight:.4f}, Cash on Hand: {cash_on_hand:.2f}")
                investment_amount = cash_on_hand * confidence_weight 
                
                num_shares_to_buy = int(investment_amount / current_price_for_ticker)

                cost = num_shares_to_buy * current_price_for_ticker

                if num_shares_to_buy > 0 and cash_on_hand >= cost:
                    cash_on_hand -= cost
                    shares_held[ticker] += num_shares_to_buy
                    last_buy_date[ticker] = current_date
                    print(f"{current_date.strftime('%Y-%m-%d')}: Bought {num_shares_to_buy} of {ticker} at {current_price_for_ticker:.2f} (Confidence: {prob_up:.2f}). Cash: {cash_on_hand:.2f}")

        # Update overall portfolio value at the end of the day after trades
        current_day_value_after_trades = cash_on_hand
        for ticker in tickers:
            if shares_held[ticker] > 0:
                current_price = test_close_prices.loc[current_date, ticker]
                current_day_value_after_trades += shares_held[ticker] * current_price
        
        bot_portfolio_value.append(current_day_value_after_trades)
        print(f"--- End of {current_date.strftime('%Y-%m-%d')} --- Total Portfolio Value: ${current_day_value_after_trades:,.2f}")




# For each ticker, calculate average metrics
for ticker in tickers:
    print(f"Average performance for {ticker}:")
    scores = all_scores[ticker]
    if scores: # Check if there are any scores to avoid division by zero
        avg_scores = {
            metric: sum(score[metric] for score in scores) / len(scores)
            for metric in scores[0]
        }
        for metric, value in avg_scores.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No scores available.")
    print()

# --- Post-simulation Analysis and Graphing ---

# Ensure simulation_dates align with bot_portfolio_value (first value is initial_capital before any dates)
# Adjust simulation_dates to match the length of bot_portfolio_value
# The first element in `bot_portfolio_value` is `initial_capital` which represents the start of the simulation *before* the first trading day.
# `simulation_dates` starts from the first day where predictions are made.
# So, `simulation_dates` should correspond to `bot_portfolio_value[1:]`.

# If simulation_dates is empty, it means no test windows were processed or test sets were too small.
if not simulation_dates:
    print("No simulation dates. Please check your data and window sizes.")
else:
    simulation_dates_idx = pd.to_datetime(simulation_dates)
    # Remove duplicate dates and sort, as extending a list might add duplicates if windows overlap
    simulation_dates_unique_sorted = pd.Series(simulation_dates_idx).drop_duplicates().sort_values()

    # Align S&P 500 data with simulation dates
    # Make sure spy_cumulative_returns has an index that matches the type of simulation_dates_unique_sorted
    spy_portfolio_value = (initial_capital * spy_cumulative_returns.loc[simulation_dates_unique_sorted]).dropna()

    # Align bot_portfolio_value with simulation dates
    # bot_portfolio_value[1:] because the first element is the initial capital, not tied to a specific date
    bot_portfolio_series = pd.Series(bot_portfolio_value[1:], index=simulation_dates_idx)
    
    # Drop duplicates from bot_portfolio_series if any exist (e.g. from overlapping test windows)
    # This keeps the last value for a given date in case of duplicates
    bot_portfolio_series_clean = bot_portfolio_series.loc[~bot_portfolio_series.index.duplicated(keep='last')]
    
    # Reindex both series to a common set of dates (intersection)
    common_dates = bot_portfolio_series_clean.index.intersection(spy_portfolio_value.index)
    
    if common_dates.empty:
        print("No common dates between simulated portfolio and S&P 500 for comparison. Check data ranges or window overlaps.")
    else:
        bot_portfolio_aligned = bot_portfolio_series_clean.loc[common_dates]
        spy_portfolio_aligned = spy_portfolio_value.loc[common_dates]

        # Normalize starting values for comparison
        # Only normalize if the first value is not zero to avoid division by zero
        bot_normalized = bot_portfolio_aligned / bot_portfolio_aligned.iloc[0] * initial_capital if bot_portfolio_aligned.iloc[0] != 0 else bot_portfolio_aligned
        spy_normalized = spy_portfolio_aligned / spy_portfolio_aligned.iloc[0] * initial_capital if spy_portfolio_aligned.iloc[0] != 0 else spy_portfolio_aligned


        plt.figure(figsize=(14, 7))
        plt.plot(bot_normalized.index, bot_normalized, label='Bot Portfolio Value', color='blue')
        plt.plot(spy_normalized.index, spy_normalized, label='S&P 500 (SPY) Value', color='red')
        plt.title('Simulated Bot Portfolio vs. S&P 500 (SPY)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate some performance metrics
        # ... (same performance metric calculations as before) ...
        # Ensure you handle potential division by zero if portfolio values are constant or zero.
        # Example for total return:
        if not bot_normalized.empty and bot_normalized.iloc[0] != 0:
            bot_total_return = (bot_normalized.iloc[-1] / bot_normalized.iloc[0]) - 1
        else:
            bot_total_return = 0 # Or np.nan if you prefer
        
        if not spy_normalized.empty and spy_normalized.iloc[0] != 0:
            spy_total_return = (spy_normalized.iloc[-1] / spy_normalized.iloc[0]) - 1
        else:
            spy_total_return = 0

        print(f"\n--- Simulation Performance ---")
        print(f"Bot Total Return: {bot_total_return:.2%}")
        print(f"S&P 500 Total Return: {spy_total_return:.2%}")

        # 2. Annualized Return (CAGR - Compound Annual Growth Rate)
        num_years = (common_dates[-1] - common_dates[0]).days / 365.25 if not common_dates.empty else 0
        if num_years > 0 and bot_normalized.iloc[0] != 0:
            bot_cagr = (bot_normalized.iloc[-1] / bot_normalized.iloc[0])**(1/num_years) - 1
        else:
            bot_cagr = np.nan
        
        if num_years > 0 and spy_normalized.iloc[0] != 0:
            spy_cagr = (spy_normalized.iloc[-1] / spy_normalized.iloc[0])**(1/num_years) - 1
        else:
            spy_cagr = np.nan

        # 3. Maximum Drawdown
        def calculate_max_drawdown(prices):
            if prices.empty or prices.iloc[0] == 0: # Handle empty or zero-initial prices
                return 0
            cumulative_returns = prices / prices.iloc[0]
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - peak) / peak
            return drawdown.min()

        bot_max_drawdown = calculate_max_drawdown(bot_normalized)
        spy_max_drawdown = calculate_max_drawdown(spy_normalized)
        print(f"Bot Maximum Drawdown: {bot_max_drawdown:.2%}")
        print(f"S&P 500 Maximum Drawdown: {spy_max_drawdown:.2%}")

        # 4. Sharpe Ratio
        trading_days_per_year = 252
        def calculate_sharpe_ratio(daily_returns, risk_free_rate_daily=0):
            if daily_returns.empty or daily_returns.std() == 0: # Avoid division by zero
                return np.nan
            excess_returns = daily_returns - risk_free_rate_daily
            return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year)

        bot_daily_returns = bot_normalized.pct_change().dropna()
        spy_daily_returns = spy_normalized.pct_change().dropna()

        bot_sharpe_ratio = calculate_sharpe_ratio(bot_daily_returns)
        spy_sharpe_ratio = calculate_sharpe_ratio(spy_daily_returns)
        print(f"Bot Sharpe Ratio (Annualized, RF=0): {bot_sharpe_ratio:.2f}")
        print(f"S&P 500 Sharpe Ratio (Annualized, RF=0): {spy_sharpe_ratio:.2f}")
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

# Data fetching and preprocessing
ticker = yf.Ticker('AAPL')  # Import data
aapl_df = ticker.history(period="5y")  # Get 5 years of data in a DataFrame
open = aapl_df[['Close']].to_numpy()  # Using Close prices instead of Open

x = np.zeros((1260, 2))
y = [0] * 1260
for i in range(1240):
    x[i] = [open[i][0], open[i + 1][0]]
    y[i] = open[i + 2][0]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33
)
print(x_train[0], y_train[0])

# Model training and evaluation

# Linear Regression
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
y_pred_linear = linear.predict(x_test)
y_train_pred_linear = linear.predict(x_train)
linear_mse = mean_squared_error(y_test, y_pred_linear)
print("Linear MSE:", linear_mse, mean_squared_error(y_train, y_train_pred_linear))

# Decision Tree Regressor
max_depth = 25
random_state = 20
tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
y_train_pred_tree = tree.predict(x_train)
tree_mse = mean_squared_error(y_test, y_pred_tree)
print("Tree MSE:", tree_mse, mean_squared_error(y_train, y_train_pred_tree))

# Random Forest Regressor
forest = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
forest.fit(x_train, y_train)
y_pred_forest = forest.predict(x_test)
y_train_pred_forest = forest.predict(x_train)
forest_mse = mean_squared_error(y_test, y_pred_forest)
print("Forest MSE:", forest_mse, mean_squared_error(y_train, y_train_pred_forest))

# Neural Network Regressor
nn = MLPRegressor(random_state=1, max_iter=500)
nn.fit(x_train, y_train)
y_pred_nn = nn.predict(x_test)
y_train_pred_nn = nn.predict(x_train)
nn_mse = mean_squared_error(y_test, y_pred_nn)
print("Neural Net MSE:", nn_mse, mean_squared_error(y_train, y_train_pred_nn))

# Ensemble averaging
average_test = (y_pred_nn + y_pred_forest + y_pred_tree + y_pred_linear) / 4
average_train = (y_train_pred_nn + y_train_pred_forest + y_train_pred_tree + y_train_pred_linear) / 4
print("Ensemble MSE:", mean_squared_error(y_test, average_test), mean_squared_error(y_train, average_train))

# Dynamic weight computation based on inverse error
total_error = nn_mse + forest_mse + tree_mse + linear_mse
nn_error_inverse = total_error / nn_mse
forest_error_inverse = total_error / forest_mse
tree_error_inverse = total_error / tree_mse
linear_error_inverse = total_error / linear_mse
multiplier = 1 / (nn_error_inverse + forest_error_inverse + tree_error_inverse + linear_error_inverse)
nn_weight = nn_error_inverse * multiplier
forest_weight = forest_error_inverse * multiplier
tree_weight = tree_error_inverse * multiplier
linear_weight = linear_error_inverse * multiplier
print("Initial Weights:", nn_weight, forest_weight, tree_weight, linear_weight)

# Dynamic weighting loop based on percent error per day
days = 10
nn_weight = 0.25
forest_weight = 0.25
tree_weight = 0.25
linear_weight = 0.25
cumulative_errors = np.array([0.0, 0.0, 0.0, 0.0])
for day in range(days):
    print("Weights on day", day, ":", nn_weight, forest_weight, tree_weight, linear_weight)
    if y_test[day] != 0:
        nn_percent_error = abs((y_pred_nn[day] - y_test[day]) / y_test[day])
        forest_percent_error = abs((y_pred_forest[day] - y_test[day]) / y_test[day])
        tree_percent_error = abs((y_pred_tree[day] - y_test[day]) / y_test[day])
        linear_percent_error = abs((y_pred_linear[day] - y_test[day]) / y_test[day])
    cumulative_errors[0] += nn_percent_error
    cumulative_errors[1] += forest_percent_error
    cumulative_errors[2] += tree_percent_error
    cumulative_errors[3] += linear_percent_error

    avg_errors = cumulative_errors / (day + 1)
    inverse_errors = np.array([1 / (e + 1e-8) for e in avg_errors])
    weights = inverse_errors / np.sum(inverse_errors)
    nn_weight = weights[0]
    forest_weight = weights[1]
    tree_weight = weights[2]
    linear_weight = weights[3]

# Final predictions with weighted average
predictions = (nn.predict(x) * nn_weight) + (tree.predict(x) * tree_weight) + (forest.predict(x) * forest_weight) + (linear.predict(x) * linear_weight)

# Backtesting trading simulation
budget = 10000
stocks = 0
k = 1
max_k = 200
initial_budget = 10000
initial_stocks = 0

while k <= max_k:
    budget = initial_budget
    stocks = initial_stocks

    for i in range(1250):
        if (y[i] < predictions[i + 1]) and (budget > k * y[i]):
            budget -= k * y[i]
            stocks += k
        elif stocks > k - 1:
            budget += k * y[i]
            stocks -= k

        if i == 1249:
            budget += y[i] * stocks
            stocks = 0

    print(f"For k={k}: stocks={stocks}, budget={budget - 10000}")
    k += 1

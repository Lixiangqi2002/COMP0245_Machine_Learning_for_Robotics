import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
# use Adaboost d=10 to predict the house value using the California Housing Prices dataset
print("\n--- For California Housing Prices Dataset ---")
data = fetch_california_housing()

# print(data_cleaned.shape)
X = data.data
y = data.target
feature_names = data.feature_names
data = pd.DataFrame(X, columns=feature_names)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize and fit the Random Forest Regressor
adaboost_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=10),
                                     n_estimators=50, random_state=42)
adaboost_regressor.fit(X_train, y_train)

# Step 4: Make predictions
y_pred_train = adaboost_regressor.predict(X_train)
y_pred_test = adaboost_regressor.predict(X_test)

# Step 5: Evaluate the model performance
rmse_train = root_mean_squared_error(y_train, y_pred_train)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f" Training RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
print(f" Testing RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")

# Optional: Plot the predictions vs true values for the test set
# CHANGE FILEPATH TO RUN <---------------------------
file_path = "figures/bestModel_.png"

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color="blue", edgecolor="k", s=80, alpha=0.6, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="Perfect Fit")
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
# plt.title("Decision Tree Regression: Actual vs Predicted Values")
plt.title('AdaBoost Regressor with max_depth=10- Predictions vs True Values (Test Set)')
plt.show()
plt.legend()

# plt.savefig(file_path, format='png', dpi=300)
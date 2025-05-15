import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# Generate or reuse synthetic data
# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ada_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=25),
                                  n_estimators=50, random_state=42, loss='linear')

ada_regressor.fit(X_train, y_train)
y_ada_pred_train = ada_regressor.predict(X_train)

rmse_train = root_mean_squared_error(y_train, y_ada_pred_train)
r2_train = r2_score(y_train, y_ada_pred_train)

print(f"AdaBoost - Training Set RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")

# Optional: Visualize the training set performance
fig = plt.figure(figsize=(10, 6))

# 3D plot for AdaBoost predictions on training set
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_ada_pred_train, c='orange', marker='^', label='AdaBoost Predicted Values')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue', marker='o', label='True Values')
ax.set_title('AdaBoost Predictions vs True Values (Training Set)')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plt.legend()
plt.show()
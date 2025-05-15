import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


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

tree_0 = DecisionTreeRegressor(max_depth=1, splitter="best")
tree_0.fit(X_train, y_train)

tree_1 = DecisionTreeRegressor(max_depth=1, splitter="random")
# splitter: {"random", "best"}
# max_depth : int, default=None
tree_1.fit(X_train, y_train)

tree_2 = DecisionTreeRegressor(max_depth=5, splitter="random")
tree_2.fit(X_train, y_train)

tree_3 = DecisionTreeRegressor(max_depth=10, splitter="random")
tree_3.fit(X_train, y_train)



# Plotting the decision tree_0
plt.figure(figsize=(16, 10))  # Set the size of the figure
plot_tree(tree_0, fontsize=6)
plt.show()  # Show the plot

y_pred_0 = tree_0.predict(X_test)
# The mean squared error
print("Mean squared error for tree_0: %.5f" % root_mean_squared_error(y_test, y_pred_0))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination for tree_0: %.5f" % r2_score(y_test, y_pred_0))

# Plotting the decision tree_1
plt.figure(figsize=(16, 10))  # Set the size of the figure
plot_tree(tree_1, fontsize=6)
plt.show()  # Show the plot

y_pred_1 = tree_1.predict(X_test)
# The mean squared error
print("Mean squared error for tree_1: %.5f" % root_mean_squared_error(y_test, y_pred_1))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination for tree_1: %.5f" % r2_score(y_test, y_pred_1))

# Plotting the decision tree_2
plt.figure(figsize=(16, 10))  # Set the size of the figure
plot_tree(tree_2, fontsize=6)
plt.show()  # Show the plot

y_pred_2 = tree_2.predict(X_test)
# The mean squared error
print("Mean squared error for tree_2: %.5f" % root_mean_squared_error(y_test, y_pred_2))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination for tree_2: %.5f" % r2_score(y_test, y_pred_2))

# Plotting the decision tree_3
plt.figure(figsize=(16, 10))  # Set the size of the figure
plot_tree(tree_3, fontsize=6)
plt.show()  # Show the plot

y_pred_3 = tree_3.predict(X_test)
# The mean squared error
print("Mean squared error for tree_3: %.5f" % root_mean_squared_error(y_test, y_pred_3))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination for tree_3: %.5f" % r2_score(y_test, y_pred_3))

model = LinearRegression()
model.fit(X, y)

y_poly = model.predict(X_test)
mse_poly = mean_squared_error(y_poly, y_test)
r2_poly = r2_score(y_poly, y_test)

print("\n--- Performance Comparison ---")
print(f"Decision Tree Regressor - MSE: {root_mean_squared_error(y_test, y_pred_3):.4f}, R²: {r2_score(y_test, y_pred_3):.4f}")
print(f"Polynomial Regression (degree=3) - MSE: {mse_poly:.4f}, R²: {r2_poly:.4f}")


# Plot the predicted values for both models
fig = plt.figure(figsize=(18, 10))

# Subplot for Decision Tree predictions
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X_test[:, 0], X_test[:, 1], y_pred_3, c='red', marker='x', label='DT Predicted Values')
ax1.set_title('Decision Tree Predictions')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('y')

# Subplot for Polynomial Regression predictions
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], y_poly, c='green', marker='^', label='Poly Predicted Values')
ax2.set_title('Polynomial Regression Predictions')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('y')

# Subplot for True values
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', marker='o', label='True Values')
ax3.set_title('True Values')
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_zlabel('y')

plt.tight_layout()
plt.show()
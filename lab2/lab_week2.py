from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor

from sklearn.metrics import root_mean_squared_error, r2_score

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------ Functions ------------------------------

def regressor_metrics(_reg):

    _reg.fit(X_train, y_train)
    y_pred = _reg.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_pred, rmse, r2

def visualisor(y_pred, tree_type, max_depth):

    # CHANGE FILEPATH TO RUN <---------------------------
    file_path = "figures/" + tree_type + "_" + str(max_depth) + ".png"

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="blue", edgecolor="k", s=80, alpha=0.6, label="Predicted vs Actual")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="Perfect Fit")
    plt.xlabel("Actual Values (y_test)")
    plt.ylabel("Predicted Values (y_pred)")
    plt.title("Decision Tree Regression: Actual vs Predicted Values")
    plt.legend()
    plt.show()
    # plt.savefig(file_path, format='png', dpi=300)

# ------------------------------ Saving Results & Tree Depth Variations ------------------------------

tree_max_depth = [1, 5, 10, 15, 25 ]
polynomial_results = [0,0]
decision_tree_results, bagging_results, rand_forest_results, ada_results = np.empty((2, 5)), np.empty((2, 5)), np.empty((2, 5)), np.empty((2, 5))

# ------------------------------ Main Loop ------------------------------

for i in range(len(tree_max_depth)):

    # Initialise Regressors
    decision_tree_reg = DecisionTreeRegressor(max_depth=tree_max_depth[i], splitter="best")

    bagging_reg = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=tree_max_depth[i]), 
            n_estimators=50, 
            random_state=42)

    rand_forest_reg = RandomForestRegressor( 
            n_estimators=50, 
            random_state=42,
            max_depth=tree_max_depth[i])
    
    ada_boost_reg = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=tree_max_depth[i]), 
            n_estimators=50, 
            random_state=42, loss='linear')


    # Getting Results using regressor_metrics() method
    y_pred_dec_tree, decision_tree_results[0][i], decision_tree_results[1][i] = regressor_metrics(decision_tree_reg)
    y_pred_bagging, bagging_results[0][i], bagging_results[1][i] = regressor_metrics(bagging_reg)
    y_pred_forest, rand_forest_results[0][i], rand_forest_results[1][i] = regressor_metrics(rand_forest_reg)
    y_pred_ada, ada_results[0][i], ada_results[1][i] = regressor_metrics(ada_boost_reg)

    # Results to pandas dataframe
    decision_tree_df = pd.DataFrame(decision_tree_results, index=['RMSE', 'R2_ Score'], columns=tree_max_depth)
    bagging_df =       pd.DataFrame(bagging_results, index=['RMSE', 'R2_ Score'], columns=tree_max_depth)
    rand_forest_df =   pd.DataFrame(rand_forest_results, index=['RMSE', 'R2_ Score'], columns=tree_max_depth)
    ada_df =           pd.DataFrame(ada_results, index=['RMSE', 'R2_ Score'], columns=tree_max_depth)

    # Creating and saving Graphs
    visualisor(y_pred_dec_tree, "decision_tree", tree_max_depth[i])
    visualisor(y_pred_bagging, "bagging", tree_max_depth[i])
    visualisor(y_pred_forest, "rand_forest", tree_max_depth[i])
    visualisor(y_pred_ada, "ada_boost", tree_max_depth[i])

    # For polynomial regressor
    linear_reg = LinearRegression()
    y_pred_polynomial, polynomial_results[0], polynomial_results[1] = regressor_metrics(linear_reg)
    polynomial_df = pd.DataFrame(polynomial_results, index=['RMSE', 'R2_ Score'])
    visualisor(y_pred_polynomial, "polynomial_regression", 0)
    # combined_df = pd.concat()

    # Create combined Dataframe
    combined_df = pd.concat([decision_tree_df, bagging_df, rand_forest_df, ada_df, polynomial_df], keys=['decision_tree', 'bagging', 'rand_forest', 'ada_boost', 'polynomial_regression'])


# Save Dataframe
# combined_df.to_csv("figures/results.csv")

import numpy as np
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt



def fit_gp_model_1d(X_values, y_values, length_scale):

    rbf_kernel = RBF(
        length_scale=length_scale,  # Initial length scale
        length_scale_bounds="fixed"  # Bounds for length scale

    )

    # Create and fit the GP regressor
    gp = GaussianProcessRegressor(
        kernel=rbf_kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp.fit(X_values, y_values)

    return gp


def create_prediction_range_kp0(upper_bound):
    kp0_range = np.linspace(0.1, upper_bound, 200).reshape(-1, 1)
    return kp0_range


def create_prediction_range_kd0(upper_bound):
    kd0_range = np.linspace(0.0, upper_bound, 200).reshape(-1, 1)
    return kd0_range


def plot_gp_results_1d(kp0_values, kd0_values, tracking_errors, gp_kp0, gp_kd0, lower_bound_kp, lower_bound_kd, upper_bound_kp, upper_bound_kd, plot_name):
    """
    Function to plot the GP results for kp[0] and kd[0] along with tracking errors and confidence intervals.

    Parameters:
    - kp0_values: Array of kp[0] values (1D array).
    - kd0_values: Array of kd[0] values (1D array).
    - tracking_errors: Array of corresponding tracking errors (1D array).
    - gp_kp0: GP model for kp[0].
    - gp_kd0: GP model for kd[0].
    - upper_bound_kp: Upper bound for kp[0].
    - upper_bound_kd: Upper bound for kd[0].
    - plot_name: Name of the file to save the plot.
    """

    # Generate points to predict for plotting (for kp[0] and kd[0])
    x_kp0 = np.linspace(lower_bound_kp, upper_bound_kp, 1000).reshape(-1, 1)
    x_kd0 = np.linspace(lower_bound_kd, upper_bound_kd, 1000).reshape(-1, 1)

    # Predict mean and standard deviation using the GP models
    mu_kp0, sigma_kp0 = gp_kp0.predict(x_kp0, return_std=True)
    mu_kd0, sigma_kd0 = gp_kd0.predict(x_kd0, return_std=True)

    # Create the plot
    plt.figure(figsize=(16, 8))

    # Plot for kp[0]
    plt.subplot(1, 2, 1)
    plt.plot(x_kp0, mu_kp0, 'b-', label="Mean Prediction")
    plt.fill_between(x_kp0.flatten(), mu_kp0 - 1.96 * sigma_kp0, mu_kp0 + 1.96 * sigma_kp0,
                     alpha=0.2, color='blue', label="95% Confidence Interval")
    plt.scatter(kp0_values, tracking_errors, c='r', s=50, zorder=10, label="Observed Points")
    plt.title('GP Prediction for kp[0]')
    plt.xlabel('kp[0]')
    plt.ylabel('Tracking Error')
    plt.grid(True)
    plt.legend()

    # Plot for kd[0]
    plt.subplot(1, 2, 2)
    plt.plot(x_kd0, mu_kd0, 'g-', label="Mean Prediction")
    plt.fill_between(x_kd0.flatten(), mu_kd0 - 1.96 * sigma_kd0, mu_kd0 + 1.96 * sigma_kd0,
                     alpha=0.2, color='green', label="95% Confidence Interval")
    plt.scatter(kd0_values, tracking_errors, c='r', s=50, zorder=10, label="Observed Points")
    plt.title('GP Prediction for kd[0]')
    plt.xlabel('kd[0]')
    plt.ylabel('Tracking Error')
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified file
    plt.tight_layout()
    # plt.savefig(plot_name, format='png', dpi=300)
    plt.show()

    print(f"Plot saved to {plot_name}")



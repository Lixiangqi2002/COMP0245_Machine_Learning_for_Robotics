import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, medfilt, filtfilt, butter

from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models


# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)


def exponential_moving_average(data, alpha=0.3):
    """
    Applies an exponential moving average filter to smooth the data.

    Parameters:
    - data: Array of predicted joint positions (shape: [timesteps, joints]).
    - alpha: Smoothing factor (0 < alpha <= 1).

    Returns:
    - smoothed_data: Array of smoothed joint positions.
    """
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # Initialize with the first position
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    return smoothed_data

def load_data(file_path):
    """Load data from a given text file."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip the header
    return data

def plot_task3_1_compare_MLP_RF():
    for g in range(10):
        fig_image_dir = "task3/task3_1/neural_network/"
        predict_pos_MLP = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_MLP = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_MLP = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_MLP = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = "task3/task3_1/random_forest_depth_2/"
        predict_pos_RF_2 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_RF_2 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_RF_2 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_RF_2 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = "task3/task3_1/random_forest_depth_10/"
        predict_pos_RF_10 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_RF_10 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_RF_10 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_RF_10 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        # Load data from files
        predict_pos_MLP = load_data(predict_pos_MLP)
        predict_vel_MLP = load_data(predict_vel_MLP)
        real_pos_MLP = load_data(real_pos_MLP)
        real_vel_MLP = load_data(real_vel_MLP)

        predict_pos_RF_2 = load_data(predict_pos_RF_2)
        predict_vel_RF_2 = load_data(predict_vel_RF_2)
        real_pos_RF_2 = load_data(real_pos_RF_2)
        real_vel_RF_2 = load_data(real_vel_RF_2)

        predict_pos_RF_10 = load_data(predict_pos_RF_10)
        predict_vel_RF_10 = load_data(predict_vel_RF_10)
        real_pos_RF_10 = load_data(real_pos_RF_10)
        real_vel_RF_10 = load_data(real_vel_RF_10)

        # Determine number of joints based on the shape of the data
        num_joints = predict_pos_MLP.shape[1]
        # Plot Difference between RF-2 and RF-10
        for i in range(num_joints):
            plt.figure(figsize=(15, 10))

            # plot for compare predicted position
            plt.subplot(2, 2, 1)
            plt.plot(predict_pos_MLP[:, i], label=f'Predicted Position for MLP - Joint {i + 1}', color='r',
                     linestyle='-', linewidth=2)
            plt.plot(predict_pos_RF_2[:, i], label=f'Predicted Position for Random Forest 2 Depth - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(predict_pos_RF_10[:, i], label=f'Predicted Position for Random Forest 10 Depth - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)
            plt.title(f'Predicted Position Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Position')
            plt.legend()
            plt.grid()

            # plot for compare real position
            plt.subplot(2, 2, 2)
            plt.plot(real_pos_MLP[:, i], label=f'Measured Position for MLP - Joint {i + 1}', color='r',
                     linestyle='-', linewidth=2)
            plt.plot(real_pos_RF_2[:, i], label=f'Measured Position for Random Forest 2 Depth - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(real_pos_RF_10[:, i], label=f'Measured Position for Random Forest 10 Depth - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)
            plt.title(f'Measured Position Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # plot for compare predicted velocity
            plt.subplot(2, 2, 3)
            plt.plot(predict_vel_MLP[:, i], label=f'Predicted Velocity for MLP - Joint {i + 1}', color='r',
                     linestyle='-', linewidth=2)
            plt.plot(predict_vel_RF_2[:, i], label=f'Predicted Velocity for Random Forest 2 Depth - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1)
            plt.plot(predict_vel_RF_10[:, i], label=f'Predicted Velocity for Random Forest 10 Depth - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=1)
            plt.title(f'Predicted Velocity Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # plot for compare true velocity
            plt.subplot(2, 2, 4)
            plt.plot(real_vel_MLP[:, i], label=f'Measured Velocity for MLP - Joint {i + 1}', color='r',
                     linestyle='-', linewidth=2)
            plt.plot(real_vel_RF_2[:, i], label=f'Measured Velocity for Random Forest 2 Depth - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1)
            plt.plot(real_vel_RF_10[:, i], label=f'Measured Velocity for Random Forest 10 Depth - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=1)
            plt.title(f'Measured Velocity Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # Save the figure
            plt.tight_layout()
            image_dir = "task3/task3_1/comparison/"
            image_path = os.path.join(image_dir, f"goal_{g}_joint_{i + 1}_overall_comparison.png")
            if save_plot:
                plt.savefig(image_path, format='png', dpi=300)
            plt.show()
            plt.close()

def plot_task3_3_compare_RF_2_smoothed_RF(filter, depth, para_name, para_value):

    for g in range(10):
        fig_image_dir = f"task3/task3_1/neural_network/"
        predict_pos_1 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_1 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_1 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_1 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_1/random_forest_depth_{depth}/"
        predict_pos_4 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_4 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_4 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_4 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_3/{filter}/{para_name}_{para_value[0]}_random_forest_depth_{depth}/"
        predict_pos_2 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_2 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_2 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_2 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_3/{filter}/{para_name}_{para_value[1]}_random_forest_depth_{depth}/"
        predict_pos_3 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_3 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_3 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_3 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_3/{filter}/{para_name}_{para_value[2]}_random_forest_depth_{depth}/"
        predict_pos_5 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_5 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_5 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_5 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        # Load data from files
        predict_pos_1 = load_data(predict_pos_1)
        predict_vel_1 = load_data(predict_vel_1)
        real_pos_1 = load_data(real_pos_1)
        real_vel_1 = load_data(real_vel_1)

        predict_pos_2 = load_data(predict_pos_2)
        predict_vel_2 = load_data(predict_vel_2)
        real_pos_2 = load_data(real_pos_2)
        real_vel_2 = load_data(real_vel_2)

        predict_pos_3 = load_data(predict_pos_3)
        predict_vel_3 = load_data(predict_vel_3)
        real_pos_3 = load_data(real_pos_3)
        real_vel_3 = load_data(real_vel_3)

        predict_pos_4 = load_data(predict_pos_4)
        predict_vel_4 = load_data(predict_vel_4)
        real_pos_4 = load_data(real_pos_4)
        real_vel_4 = load_data(real_vel_4)

        predict_pos_5 = load_data(predict_pos_5)
        predict_vel_5 = load_data(predict_vel_5)
        real_pos_5 = load_data(real_pos_5)
        real_vel_5 = load_data(real_vel_5)

        # Determine number of joints based on the shape of the data
        num_joints = predict_pos_2.shape[1]
        # Plot Difference between RF-2 and RF-3
        for i in range(num_joints):
            plt.figure(figsize=(15, 10))

            # plot for compare predicted position
            plt.subplot(2, 2, 1)
            plt.plot(predict_pos_4[:, i],
                     label=f'Predicted Position for Random Forest {depth} Depth No Smoothing - Joint {i + 1}',
                     color='y', linestyle='-', linewidth=2)
            plt.plot(predict_pos_1[:, i], label=f'Predicted Position for MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.plot(predict_pos_5[:, i],
                     label=f'Predicted Position for Random Forest {depth} Depth {para_name} = {para_value[2]} - Joint {i + 1}',
                     color='orange', linestyle='-', linewidth=2)

            plt.plot(predict_pos_2[:, i], label=f'Predicted Position for Random Forest {depth} Depth {para_name} = {para_value[0]} - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(predict_pos_3[:, i], label=f'Predicted Position for Random Forest {depth} Depth {para_name} = {para_value[1]} - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)

            plt.title(f'Predicted Position Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Position')
            plt.legend()
            plt.grid()

            # plot for compare real position
            plt.subplot(2, 2, 2)
            plt.plot(real_pos_4[:, i],
                     label=f'Measured Position for Random Forest {depth} Depth No Smoothing - Joint {i + 1}',
                     color='y', linestyle='-', linewidth=2)
            plt.plot(real_pos_5[:, i],
                     label=f'Measured Position for Random Forest {depth} Depth {para_name} = {para_value[2]} - Joint {i + 1}', color='orange',
                     linestyle='-', linewidth=2)
            plt.plot(real_pos_1[:, i], label=f'Measured Position for MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.plot(real_pos_2[:, i], label=f'Measured Position for Random Forest {depth} Depth {para_name} = {para_value[0]} - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(real_pos_3[:, i], label=f'Measured Position for Random Forest {depth} Depth {para_name} = {para_value[1]} - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)

            plt.title(f'Measured Position Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # plot for compare predicted velocity
            plt.subplot(2, 2, 3)
            plt.plot(predict_vel_4[:, i],
                     label=f'Predicted Velocity for Random Forest {depth} Depth No Smoothing - Joint {i + 1}',
                     color='y', linestyle='-', linewidth=1)
            plt.plot(predict_vel_5[:, i],
                     label=f'Predicted Velocity for Random Forest {depth} Depth {para_name} = {para_value[2]} - Joint {i + 1}', color='orange',
                     linestyle='-', linewidth=2)
            plt.plot(predict_vel_1[:, i], label=f'Predicted Velocity for MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.plot(predict_vel_2[:, i], label=f'Predicted Velocity for Random Forest {depth} Depth {para_name} = {para_value[0]} - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1)
            plt.plot(predict_vel_3[:, i], label=f'Predicted Velocity for Random Forest {depth} Depth {para_name} = {para_value[1]} - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=1)

            plt.title(f'Predicted Velocity Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # plot for compare true velocity
            plt.subplot(2, 2, 4)
            plt.plot(real_vel_4[:, i],
                     label=f'Measured Velocity for Random Forest {depth} Depth No Smoothing - Joint {i + 1}',
                     color='y', linestyle='-', linewidth=1)
            plt.plot(real_vel_5[:, i],
                     label=f'Measured Velocity for Random Forest {depth} Depth {para_name} = {para_value[2]} - Joint {i + 1}', color='orange',
                     linestyle='-', linewidth=2)
            plt.plot(real_vel_1[:, i], label=f'Measured Velocity for MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.plot(real_vel_2[:, i], label=f'Measured Velocity for Random Forest {depth} Depth {para_name} = {para_value[0]} - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1)
            plt.plot(real_vel_3[:, i], label=f'Measured Velocity for Random Forest {depth} Depth {para_name} = {para_value[1]} - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=1)

            plt.title(f'Measured Velocity Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # Save the figure
            plt.tight_layout()
            image_dir = "task3/task3_3/comparison/"
            image_path = os.path.join(image_dir, f"goal_{g}_joint_{i + 1}_depth_{depth}_{filter}_overall_comparison.png")
            if save_plot:
                plt.savefig(image_path, format='png', dpi=300)
            plt.show()
            plt.close()

def plot_task3_3_overall_comparison():
    for g in range(10):
        fig_image_dir = f"task3/task3_1/neural_network/"
        predict_pos_1 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_1 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_1 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_1 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_1/random_forest_depth_2/"
        predict_pos_4 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_4 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_4 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_4 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_3/Gaussian/sigma_150_random_forest_depth_2/"
        predict_pos_2 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_2 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_2 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_2 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_3/EMA/alpha_0.001_random_forest_depth_2/"
        predict_pos_3 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_3 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_3 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_3 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_3/EMA/alpha_0.1_random_forest_depth_10/"
        predict_pos_5 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_5 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_5 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_5 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_1/random_forest_depth_10/"
        predict_pos_6 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_6 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_6 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_6 = fig_image_dir + f"goal_{g}_real_velocity.txt"

        fig_image_dir = f"task3/task3_3/Gaussian/sigma_150_random_forest_depth_10/"
        predict_pos_7 = fig_image_dir + f"goal_{g}_predicted_position.txt"
        real_pos_7 = fig_image_dir + f"goal_{g}_real_position.txt"
        predict_vel_7 = fig_image_dir + f"goal_{g}_predicted_velocity.txt"
        real_vel_7 = fig_image_dir + f"goal_{g}_real_velocity.txt"


        # Load data from files
        predict_pos_1 = load_data(predict_pos_1)
        predict_vel_1 = load_data(predict_vel_1)
        real_pos_1 = load_data(real_pos_1)
        real_vel_1 = load_data(real_vel_1)

        predict_pos_2 = load_data(predict_pos_2)
        predict_vel_2 = load_data(predict_vel_2)
        real_pos_2 = load_data(real_pos_2)
        real_vel_2 = load_data(real_vel_2)

        predict_pos_3 = load_data(predict_pos_3)
        predict_vel_3 = load_data(predict_vel_3)
        real_pos_3 = load_data(real_pos_3)
        real_vel_3 = load_data(real_vel_3)

        predict_pos_4 = load_data(predict_pos_4)
        predict_vel_4 = load_data(predict_vel_4)
        real_pos_4 = load_data(real_pos_4)
        real_vel_4 = load_data(real_vel_4)

        predict_pos_5 = load_data(predict_pos_5)
        predict_vel_5 = load_data(predict_vel_5)
        real_pos_5 = load_data(real_pos_5)
        real_vel_5 = load_data(real_vel_5)

        predict_pos_6 = load_data(predict_pos_6)
        predict_vel_6 = load_data(predict_vel_6)
        real_pos_6 = load_data(real_pos_6)
        real_vel_6 = load_data(real_vel_6)

        predict_pos_7 = load_data(predict_pos_7)
        predict_vel_7 = load_data(predict_vel_7)
        real_pos_7 = load_data(real_pos_7)
        real_vel_7 = load_data(real_vel_7)

        # Determine number of joints based on the shape of the data
        num_joints = predict_pos_2.shape[1]
        # Plot Difference between RF-2 and RF-3
        for i in range(num_joints):
            plt.figure(figsize=(17, 12))
            # plot for compare predicted position
            plt.subplot(2, 2, 1)

            plt.plot(predict_pos_4[:, i],
                     label=f'RF_2 No Smoothing - Joint {i + 1}',
                     color='gold', linestyle='-', linewidth=2)
            plt.plot(predict_pos_2[:, i],
                     label=f'RF_2 Gaussian sigma = 150 - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(predict_pos_3[:, i],
                     label=f'RF_2 EMA alpha = 0.001 - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)
            plt.plot(predict_pos_6[:, i],
                     label=f'RF_10 No Smoothing - Joint {i + 1}',
                     color='skyblue', linestyle='-', linewidth=2.5)
            plt.plot(predict_pos_7[:, i],
                     label=f'RF_10 Gaussian sigma = 150 - Joint {i + 1}',
                     color='purple', linestyle='-', linewidth=1.5)
            plt.plot(predict_pos_5[:, i],
                     label=f'RF_10 EMA alpha = 0.1 - Joint {i + 1}',
                     color='darkorange', linestyle='-', linewidth=2)
            plt.plot(predict_pos_1[:, i], label=f'MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.title(f'Predicted Position Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Position')
            plt.legend()
            plt.grid()

            # plot for compare real position
            plt.subplot(2, 2, 2)

            plt.plot(real_pos_4[:, i],
                     label=f'RF_2 No Smoothing - Joint {i + 1}',
                     color='gold', linestyle='-', linewidth=2)
            plt.plot(real_pos_6[:, i],
                     label=f'RF_10 No Smoothing - Joint {i + 1}',
                     color='skyblue', linestyle='-', linewidth=2)
            plt.plot(real_pos_2[:, i],
                     label=f'RF_2 Gaussian sigma = 150 - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(real_pos_3[:, i],
                     label=f'RF_2 EMA alpha = 0.001 - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)

            plt.plot(real_pos_7[:, i],
                     label=f'RF_10 Gaussian sigma = 150 - Joint {i + 1}',
                     color='purple', linestyle='-', linewidth=1.5)
            plt.plot(real_pos_5[:, i],
                     label=f'RF_10 EMA alpha = 0.1 - Joint {i + 1}',
                     color='darkorange', linestyle='-', linewidth=2)
            plt.plot(real_pos_1[:, i], label=f'MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.title(f'Measured Position Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # plot for compare predicted velocity
            plt.subplot(2, 2, 3)

            plt.plot(predict_vel_6[:, i],
                     label=f'RF_10 No Smoothing - Joint {i + 1}',
                     color='skyblue', linestyle='-', linewidth=2)
            plt.plot(predict_vel_4[:, i],
                     label=f'RF_2 No Smoothing - Joint {i + 1}',
                     color='gold', linestyle='-', linewidth=2)
            plt.plot(predict_vel_2[:, i],
                     label=f'RF_2 Gaussian sigma = 150 - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(predict_vel_3[:, i],
                     label=f'RF_2 EMA alpha = 0.001 - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)

            plt.plot(predict_vel_7[:, i],
                     label=f'RF_10 Gaussian sigma = 150 - Joint {i + 1}',
                     color='purple', linestyle='-', linewidth=1.5)
            plt.plot(predict_vel_5[:, i],
                     label=f'RF_10 EMA alpha = 0.1 - Joint {i + 1}',
                     color='darkorange', linestyle='-', linewidth=2)
            plt.plot(predict_vel_1[:, i], label=f'MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.title(f'Predicted Velocity Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # plot for compare true velocity
            plt.subplot(2, 2, 4)

            plt.plot(real_vel_4[:, i],
                     label=f'RF_2 No Smoothing - Joint {i + 1}',
                     color='gold', linestyle='-', linewidth=2)
            plt.plot(real_vel_6[:, i],
                     label=f'RF_10 No Smoothing - Joint {i + 1}',
                     color='skyblue', linestyle='-', linewidth=2)
            plt.plot(real_vel_2[:, i],
                     label=f'RF_2 Gaussian sigma = 150 - Joint {i + 1}',
                     color='b', linestyle='-', linewidth=1.5)
            plt.plot(real_vel_3[:, i],
                     label=f'RF_2 EMA alpha = 0.001 - Joint {i + 1}',
                     color='g', linestyle='-', linewidth=2)

            plt.plot(real_vel_7[:, i],
                     label=f'RF_10 Gaussian sigma = 150 - Joint {i + 1}',
                     color='purple', linestyle='-', linewidth=1.5)
            plt.plot(real_vel_5[:, i],
                     label=f'RF_10 EMA alpha = 0.1 - Joint {i + 1}',
                     color='darkorange', linestyle='-', linewidth=2)
            plt.plot(real_vel_1[:, i], label=f'MLP - Joint {i + 1}', color='r',
                     linestyle='--', linewidth=2)
            plt.title(f'Measured Velocity Comparison for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid()

            # Save the figure
            plt.tight_layout()
            image_dir = "task3/task3_3/comparison/"
            image_path = os.path.join(image_dir, f"goal_{g}_joint_{i + 1}_overall_comparison.png")
            if save_plot:
                plt.savefig(image_path, format='png', dpi=300)
            plt.show()
            plt.close()

def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])            # Shape: (N,)
        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Load all the models in a list
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            # Instantiate the model
            model = MLP()
            # Load the saved model
            model_dir = os.path.join(script_dir, 'task2/task2_1/model/')

            model_filename = os.path.join(model_dir, f'neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            # Load the saved Random Forest model
            model_dir = os.path.join(script_dir, 'task2/task2_2/model/')

            model_filename = os.path.join(model_dir, f'rf_depth_{depth}_joint{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)
    else:
        print("Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'")
        return

    # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    # Create a set of goal positions
    number_of_goal_positions_to_test = 10
    goal_positions = []
    seed = 0
    np.random.seed(seed)
    for i in range(number_of_goal_positions_to_test):
        goal_positions.append([
            np.random.uniform(*goal_position_bounds['x']),
            np.random.uniform(*goal_position_bounds['y']),
            np.random.uniform(*goal_position_bounds['z'])
        ])

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # PD controller gains
    kp = 1000  # Proportional gain
    kd = 100   # Derivative gain

    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()

    time_step = sim.GetTimeStep()
    # Generate test time array
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    error_data = {
        "Type": [],
        "Goal": [],
        "Joint Index": [],
        "Joint Position Error": [],
        "Joint Velocity Error": []
    }
    pos_error_data = {
        "Type": [],
        "Position Error": []
    }
    for g in range(len(goal_positions)):
        goal_position = goal_positions[g]
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        # Initialize the simulation
        sim.ResetPose()
        current_time = 0  # Initialize current time

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (num_points, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (num_points, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)
        real_joint_position_over_time = np.zeros((len(test_time_array), 7))
        for joint_idx in range(7):
            if neural_network_or_random_forest == "neural_network":
                # Prepare the test input
                test_input_tensor = torch.from_numpy(test_input).float()  # Shape: (num_points, 4)

                # Predict joint positions using the neural network
                with torch.no_grad():
                    predictions = models[joint_idx](test_input_tensor).numpy().flatten()  # Shape: (num_points,)
            elif neural_network_or_random_forest == "random_forest":
                # Predict joint positions using the Random Forest
                predictions = models[joint_idx].predict(test_input)  # Shape: (num_points,)

            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = predictions

        # Task 3-3 : smoothing filter
        if task_id == "3-3" and neural_network_or_random_forest == "random_forest":
            if filter=="EMA":
                # smaller α will provide greater smoothing, but will also slow down the response.
                predicted_joint_positions_over_time = exponential_moving_average(predicted_joint_positions_over_time, alpha=alpha)
            elif filter=="Gaussian":
                predicted_joint_positions_over_time = gaussian_filter1d(predicted_joint_positions_over_time, sigma=sigma, axis=0)

        # Compute qd_des_over_time by numerically differentiating the predicted joint positions
        qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
        # Clip the joint velocities to the joint limits
        qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))
        q_mes_all = []
        qd_mes_all = []
        q_d_all = []
        qd_d_all = []
        tau_cmd_arr = []
        # Data collection loop
        while current_time < test_time_array.max():
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

            # Get the index corresponding to the current time
            current_index = int(current_time / time_step)
            if current_index >= len(test_time_array):
                current_index = len(test_time_array) - 1

            # Get q_des and qd_des_clip from predicted data
            q_des = predicted_joint_positions_over_time[current_index, :]  # Shape: (7,)
            qd_des_clip = qd_des_over_time_clipped[current_index, :]      # Shape: (7,)

            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            tau_cmd_arr.append(tau_cmd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command

            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step
            # Store data for plotting
            q_mes_all.append(q_mes)
            qd_mes_all.append(qd_mes)
            q_d_all.append(q_des)
            qd_d_all.append(qd_des_clip)
            real_joint_position_over_time[current_index , :]= q_mes
        if save_txt:
            # Write q_mes_all and qd_mes_all to a text file for Q/R
            output_file_path = fig_image_dir+f"/goal_{g}_real_position.txt"
            with open(output_file_path, 'w') as f:
                for q in q_mes_all:
                    f.write(','.join(map(str, q)) + '\n')  # Write each position as a comma-separated string
            output_file_path = fig_image_dir + f"/goal_{g}_real_velocity.txt"
            with open(output_file_path, 'w') as f:
                for qd in qd_mes_all:
                    f.write(','.join(map(str, qd)) + '\n')  # Write each velocity as a comma-separated string

            output_file_path = fig_image_dir + f"/goal_{g}_predicted_position.txt"
            with open(output_file_path, 'w') as f:
                for q in q_d_all:
                    f.write(','.join(map(str, q)) + '\n')  # Write each position as a comma-separated string
            output_file_path = fig_image_dir + f"/goal_{g}_predicted_velocity.txt"
            with open(output_file_path, 'w') as f:
                for qd in qd_d_all:
                    f.write(','.join(map(str, qd)) + '\n')  # Write each velocity as a comma-separated string
            print(f"Data has been written to {output_file_path}.")

        # After the trajectory, compute the final cartesian position
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
        print(f"Final computed cartesian position: {final_cartesian_pos}")
        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")

        cartesian_prediction_positions_over_time = []
        cartesian_real_pos_MLPitions_over_time = []

        for i in range(len(test_time_array)):
            joint_positions = predicted_joint_positions_over_time[i, :]
            cartesian_pos, _ = dyn_model.ComputeFK(joint_positions, controlled_frame_name)
            cartesian_prediction_positions_over_time.append(cartesian_pos.copy())
        for i in range(len(test_time_array)):
            joint_positions_real = real_joint_position_over_time[i, :]
            cartesian_pos_real, _ = dyn_model.ComputeFK(joint_positions_real, controlled_frame_name)
            cartesian_real_pos_MLPitions_over_time.append(cartesian_pos_real.copy())

        cartesian_prediction_positions_over_time = np.array(cartesian_prediction_positions_over_time)  # Shape: (num_points, 3)
        cartesian_real_pos_MLPitions_over_time = np.array(cartesian_real_pos_MLPitions_over_time)  # Shape: (num_points, 3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(cartesian_real_pos_MLPitions_over_time[:, 0], cartesian_real_pos_MLPitions_over_time[:, 1],
                cartesian_real_pos_MLPitions_over_time[:, 2], label='Real Trajectory')
        ax.plot(cartesian_prediction_positions_over_time[:, 0], cartesian_prediction_positions_over_time[:, 1],
                cartesian_prediction_positions_over_time[:, 2], label='Predicted Trajectory', linestyle='--', color="red", linewidth=2)

        ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='red', label='Goal Position')
        ax.scatter(final_cartesian_pos[0], final_cartesian_pos[1], final_cartesian_pos[2], marker='*', color='blue',
                   label=f"Final Cartesian Position \nError : {position_error:.6f} ")  # Mark error position in X
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Predicted Cartesian Trajectory')
        plt.legend()
        if save_plot:
            image_path = os.path.join(fig_image_dir, f"goal_{g}_trajectory_comparison_cartesian.png")
            plt.savefig(image_path, format='png', dpi=300)  # Save the image
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 15))
        tau_cmd_arr = np.array(tau_cmd_arr)
        for i in range(tau_cmd_arr.shape[1]):
            plt.subplot(7, 1, i+1)
            plt.plot([t[i] for t in tau_cmd_arr], label=f"Joint {i + 1}")
        plt.title(f'Torque Command')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()
        image_path = os.path.join("task3", f"goal_{g}_torque_comparison.png")
        plt.savefig(image_path, format='png', dpi=300)
        plt.show()


        num_joints = len(q_mes)
        for i in range(num_joints):
            plt.figure(figsize=(10, 8))

            # Position plot for joint i
            q_mes_all = np.array(q_mes_all)
            q_d_all = np.array(q_d_all)
            joint_position_error = np.linalg.norm(q_mes_all[:,i] - q_d_all[:,i])
            print(f"Joint {i + 1} final position error between computed position and goal: {joint_position_error}")
            plt.subplot(2, 1, 1)
            plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Position Error {joint_position_error} ')
            plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i + 1}', linestyle='--', color="red", linewidth=2)
            plt.scatter(len(q_mes_all), q_mes_all[-1][i], label = f'Real Final Position - {q_mes_all[-1][i]}', color='blue')
            plt.scatter(len(q_d_all), q_d_all[-1][i], label = f'Desired Final Position - {q_d_all[-1][i]}', marker='*', color='red')
            plt.title(f'Position Tracking for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Position')
            plt.legend()

            # Velocity plot for joint i
            # Compute velocity error
            qd_mes_all = np.array(qd_mes_all)
            qd_d_all = np.array(qd_d_all)
            velocity_error = np.linalg.norm(qd_mes_all[:,i] - qd_d_all[:,i])
            print(f"Joint {i + 1} final velocity error between computed position and goal: {velocity_error}")
            plt.subplot(2, 1, 2)
            plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Velocity Error {velocity_error}')
            plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i + 1}', linestyle='--', color="red", linewidth=2)
            plt.scatter(len(qd_mes_all), qd_mes_all[-1][i], label=f'Real Final Vellocity - {qd_mes_all[-1][i]}',
                        color='blue')
            plt.scatter(len(qd_d_all), qd_d_all[-1][i], label=f'Desired Final Vellocity - {qd_d_all[-1][i]}', marker='*',
                        color='red')
            plt.title(f'Velocity Tracking for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()

            plt.tight_layout()
            if save_plot:
                image_path = os.path.join(fig_image_dir, f"goal_{g}_joint_{i + 1}_comparison.png")
                plt.savefig(image_path, format='png', dpi=300)  # Save the image
            plt.show()
            plt.close()

            # save to dictionary
            error_data["Type"].append(type)
            error_data["Goal"].append(g)
            error_data["Joint Index"].append(i + 1)
            error_data["Joint Position Error"].append(joint_position_error)
            error_data["Joint Velocity Error"].append(velocity_error)

        error_data["Type"].append(type)
        error_data["Goal"].append(g)
        error_data["Joint Index"].append("N/A")
        error_data["Joint Position Error"].append(np.nan)
        error_data["Joint Velocity Error"].append(np.nan)

        pos_error_data["Type"].append(type)
        pos_error_data["Position Error"].append(position_error)
        print(f"Position error between computed position and goal: {position_error}")

    if save_xlsx:
        if task_id == "3-1":
            df = pd.DataFrame(error_data)
            output_excel_path = os.path.join(f"task3/task3_1/", "joint_errors.xlsx")
            # df.to_excel(output_excel_path, index=False)
            with pd.ExcelWriter(output_excel_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
            print(f"Error data has been written to {output_excel_path}.")
        elif task_id == "3-3":
            df = pd.DataFrame(error_data)
            output_excel_path = os.path.join(f"task3/task3_3", "task3_3.xlsx")
            # df.to_excel(output_excel_path, index=False)
            with pd.ExcelWriter(output_excel_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
            print(f"Error data has been written to {output_excel_path}.")
        df = pd.DataFrame(pos_error_data)
        output_excel_path = os.path.join(f"task3/task3_1/", "position_error.xlsx")
        # df.to_excel(output_excel_path, index=False)
        with pd.ExcelWriter(output_excel_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)


if __name__ == '__main__':
    # Set the model type: "neural_network" or "random_forest"
    neural_network_or_random_forest = "random_forest"  # Change to "random_forest" to use Random Forest models
    task_id = "3-3"  # "3-1" / "3-3"
    filter = "Gaussian"  # EMA, Gaussian
    depth = 10
    save_xlsx = False
    save_txt = False
    save_plot = False
    if task_id == "3-1":
        if neural_network_or_random_forest == "neural_network":
            type = f"{neural_network_or_random_forest}"
            fig_image_dir = f"task3/task3_1/{neural_network_or_random_forest}"
        else:
            type = f"{neural_network_or_random_forest}_depth_{depth}"
            fig_image_dir = f"task3/task3_1/{neural_network_or_random_forest}_depth_{depth}"
    elif task_id == "3-3":
        output_excel_path = os.path.join("task3/task3_3", "task3_3.xlsx")
        if not os.path.isfile(output_excel_path) or not output_excel_path.endswith(".xlsx"):
            print("Error: The specified file is not a valid .xlsx file.")
        if neural_network_or_random_forest == "neural_network":
            type = f"{neural_network_or_random_forest}"
            fig_image_dir = f"task3/task3_3/{neural_network_or_random_forest}"
        else:
            if filter == "EMA":
                alpha = 0.1  # [0.1, 0.01, 0.001]
                type = f"alpha_{alpha}_{neural_network_or_random_forest}_depth_{depth}"
                fig_image_dir = f"task3/task3_3/EMA/alpha_{alpha}_{neural_network_or_random_forest}_depth_{depth}"
            elif filter == "Gaussian":
                sigma = 150  # [2, 75, 150]
                type = f"sigma_{sigma}_{neural_network_or_random_forest}_depth_{depth}"
                fig_image_dir = f"task3/task3_3/Gaussian/sigma_{sigma}_{neural_network_or_random_forest}_depth_{depth}"

    main()

    # make sure store enough .txt file then uncomment below codes for plotting comparison cross different models
    # TODO: task 3-1
    # compare three model without smoothing
    # plot_task3_1_compare_MLP_RF()
    # TODO: task 3-3
    # compare depth=2/10 for different smoothing effect
    # plot_task3_3_compare_RF_2_smoothed_RF(filter = "EMA", depth=2, para_name="alpha", para_value=[0.1, 0.01, 0.001])
    # plot_task3_3_compare_RF_2_smoothed_RF(filter = "EMA", depth=10, para_name="alpha", para_value=[0.1, 0.01, 0.001])
    # plot_task3_3_compare_RF_2_smoothed_RF(filter="Gaussian", depth=2, para_name="sigma", para_value=[2,75,150])
    # plot_task3_3_compare_RF_2_smoothed_RF(filter="Gaussian", depth=10, para_name="sigma", para_value=[2,75,150])
    # plot_task3_3_overall_comparison()

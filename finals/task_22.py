import pickle
import subprocess
import threading
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import psutil
import pynvml
from memory_profiler import memory_usage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models


# Set the visualization flag
visualize = True  # Set to True to enable visualization, False to disable
training_flag = True  # Set to True to train the models, False to skip training
test_cartesian_accuracy_flag = True  # Set to True to test the model with a new goal position, False to skip testing
max_depth = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]
depth = max_depth[9] # 0-9
fig_image_dir = f"task2/task2_2/depth_{depth}/"

with open(fig_image_dir + f"training_log_depth_{depth}.txt", "w") as file:
    file.write("#" * 50 + "\n")
    file.write(f"Depth: {depth} \n")
with open(fig_image_dir + f"testing_log_depth_{depth}.txt", "w") as file:
    file.write("#" * 50 + "\n")
    file.write(f"Depth: {depth} \n")


def monitor_resources(time_sample, cpu_percentages, mem_percentages, mem_usage, process, stop_event):
    while not stop_event.is_set():
        # result = subprocess.run(
        #     ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        #     stdout=subprocess.PIPE, text=True)
        # gpu_memory, gpu_utilization = result.stdout.strip().split(', ')
        # print(f"GPU Memory Usage (nvidia-smi): {gpu_memory} MB")
        # print(f"GPU Overall Utilization (nvidia-smi): {gpu_utilization}%")
        # result = subprocess.run(
        #     ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        #     stdout=subprocess.PIPE, text=True)
        # for line in result.stdout.strip().split('\n'):
        #     pid, gpu_memory_used = line.split(', ')
        #     if int(pid) == target_pid:
        #         percetage_gpu = 100 * float(gpu_memory_used) / float(gpu_memory)
        #         print(f"GPU Memory Usage (%): {percetage_gpu}%")
        #         gpu_percentages.append(float(percetage_gpu))
        #         print(f"GPU Memory Usage: {gpu_memory_used} MB")
        #         gpu_mem_usages.append(float(gpu_memory_used))
        cpu_percentages.append(process.cpu_percent(interval=None))
        mem_percentages.append(process.memory_percent())
        mem_usage.append(process.memory_info().rss / (1024 ** 2))
        time.sleep(time_sample)

if training_flag:
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])            # Shape: (N,)
        q_mes_all = np.array(data['q_mes_all'])        # Shape: (N, 7)
        goal_positions = np.array(data['goal_positions'])  # Shape: (N, 3)

        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

        # Combine time and goal data to form the input features
        X = np.hstack((time_array.reshape(-1, 1), goal_positions))  # Shape: (N, 4)

        # Split ratio
        split_ratio = 0.8

        # Initialize lists to hold training and test data for all joints
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        for joint_idx in range(7):
            cpu_percentages = []
            mem_usage = []
            mem_percentages = []

            # Extract joint data
            y = q_mes_all[:, joint_idx]  # Shape: (N,)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=split_ratio, shuffle=True, random_state=42
            )

            # Store split data
            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

            # Initialize the Random Forest regressor
            rf_model = RandomForestRegressor(
                n_estimators=100,    # Number of trees
                max_depth=depth,        # Maximum depth of the tree
                random_state=42,
                n_jobs=-1            # Use all available cores
            )
            ######################################################################################################
            ## Training:
            time_sample = 1
            process = psutil.Process()
            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=monitor_resources,
                                              args=(time_sample, cpu_percentages,
                                                    mem_percentages,
                                                    mem_usage, process, stop_event))
            monitor_thread.start()
            start_time = time.time()

            # Train the model
            rf_model.fit(X_train, y_train)
            stop_event.set()
            # Record end time
            end_time = time.time()
            monitor_thread.join()
            # Calculate average memory usage and CPU
            avg_memory_usage = max(mem_usage) - min(mem_usage)
            avg_memory_percentage = max(mem_percentages) - min(mem_percentages)
            avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)

            training_time = end_time - start_time

            print(f"Training time: {training_time:.2f} seconds")
            print(f"Average memory usage during training: {avg_memory_usage:.2f} MiB")
            print(f"Average memory percentage usage during training (%): {avg_memory_percentage:.2f}%")
            print(f"Average CPU usage during training: {avg_cpu_usage:.2f}%")

            # with open(fig_image_dir + f"training_log_depth_{depth}.txt", "a") as file:
            #     file.write(f"Training time for Joint {joint_idx + 1}: {training_time:.2f} seconds\n")
            #     file.write(f"Average CPU used percentage for Joint {joint_idx + 1}: {avg_cpu_usage:.2f}%\n")
            #     file.write(f"Average memory percentage for Joint {joint_idx + 1}: {avg_memory_percentage:.2f}%\n")
            #     file.write(f"Average memory usage for Joint {joint_idx + 1}: {avg_memory_usage:.2f} MiB\n")
            #     file.write("-" * 50 + "\n")

            ######################################################################################################
            ## Predicting:
            # Evaluate on training set
            y_train_pred = rf_model.predict(X_train)
            # Evaluate on test set
            y_test_pred = rf_model.predict(X_test)

            train_mse = np.mean((y_train - y_train_pred) ** 2)
            test_mse = np.mean((y_test - y_test_pred) ** 2)

            print(f'\nJoint {joint_idx+1}')
            print(f'Train MSE: {train_mse:.6f}')
            print(f'Test MSE: {test_mse:.6f}')

            # Save the trained model
            model_dir = os.path.join(script_dir, 'task2/task2_2/model/')

            model_filename = os.path.join(model_dir, f'rf_depth_{depth}_joint{joint_idx+1}.joblib')
            joblib.dump(rf_model, model_filename)
            print(f'Model for Joint {joint_idx+1} saved as {model_filename}')

            # Visualization (if enabled)
            if visualize:
                print(f'Visualizing results for Joint {joint_idx+1}...')

                # Plot true vs predicted positions on the test set
                sorted_indices = np.argsort(X_test[:, 0])
                X_test_sorted = X_test[sorted_indices]
                y_test_sorted = y_test[sorted_indices]
                y_test_pred_sorted = y_test_pred[sorted_indices]

                plt.figure(figsize=(10, 5))
                plt.plot(X_test_sorted[:, 0], y_test_sorted, label='True Joint Positions')
                plt.plot(X_test_sorted[:, 0], y_test_pred_sorted, label='Predicted Joint Positions', linestyle='--')
                plt.xlabel('Time (s)')
                plt.ylabel('Joint Position (rad)')
                plt.title(f'Joint {joint_idx+1} Position Prediction on Test Set')
                plt.legend()
                plt.grid(True)
                image_path = os.path.join(fig_image_dir, f"depth_{depth}_joint_{joint_idx + 1}_training.png")
                # plt.savefig(image_path, format='png', dpi=300)  # Save the image
                plt.show()

        print("Training and visualization completed.")
        plt.close()
if test_cartesian_accuracy_flag:

    if not training_flag:
        # Load the saved data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
        if not os.path.isfile(filename):
            print(f"Error: File {filename} not found in {script_dir}")
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # Extract data
            time_array = np.array(data['time'])            # Shape: (N,)

    # Testing with new goal positions
    print("\nTesting the model with new goal positions...")

    # Load all the models into a list
    models = []
    for joint_idx in range(7):
        # Load the saved model
        model_dir = os.path.join(script_dir, 'task2/task2_2/model/')

        # The name of the saved model
        model_filename = os.path.join(model_dir, f'rf_depth_{depth}_joint{joint_idx + 1}.joblib')

        try:
            rf_model = joblib.load(model_filename)

        except FileNotFoundError:
            print(f"Cannot find file {model_filename}")
            print("task_22_goal_pos needs to be run at least once with training_flag=True")
            quit()

        models.append(rf_model)

    # Generate new goal positions
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

    # Generate test time array
    test_time_array = np.linspace(time_array.min(), time_array.max(), 100)  # For example, 100 time steps

    # Initialize the dynamic model
    from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, CartesianDiffKin

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust root directory if necessary
    name_current_directory = "tests"
    root_dir = root_dir.replace(name_current_directory, "")
    # Initialize simulation interface
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)

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
    mse_errors = []
    for g in range(len(goal_positions)):
        # with open(fig_image_dir + f"testing_log_depth_{depth}.txt", "a") as file:
        #     file.write("#" * 50 + "\n")
            # file.write(f"Depth: {depth} \n")
        goal_position = goal_positions[g]
        print("\nTesting new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (100, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (100, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

        for joint_idx in range(7):
            cpu_percentages = []
            mem_usage = []
            mem_percentages = []

            process = psutil.Process()
            start_time = time.time()
            time_sample = 1
            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=monitor_resources,
                                              args=(time_sample, cpu_percentages,
                                                    mem_percentages,
                                                    mem_usage, process, stop_event))
            monitor_thread.start()
            start_time = time.time()

            # Predict joint positions
            y_pred = models[joint_idx].predict(test_input)  # Shape: (num_points,)

            stop_event.set()
            # Record end time
            end_time = time.time()
            monitor_thread.join()
            # Calculate average memory usage and CPU
            avg_memory_usage = max(mem_usage) - min(mem_usage)
            avg_memory_percentage = max(mem_percentages) - min(mem_percentages)
            avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)

            prediction_time = end_time - start_time

            print(f"Prediction time: {prediction_time:.2f} seconds")
            print(f"Average memory usage during prediction: {avg_memory_usage:.2f} MiB")
            print(f"Average memory percentage usage during prediction (%): {avg_memory_percentage:.2f}%")
            print(f"Average CPU usage during prediction: {avg_cpu_usage:.2f}%")

            # with open(fig_image_dir + f"testing_log_depth_{depth}.txt", "a") as file:
            #     file.write(f"Prediction time for Joint {joint_idx + 1}: {prediction_time:.2f} seconds\n")
            #     file.write(f"Average CPU used percentage for Joint {joint_idx + 1}: {avg_cpu_usage:.2f}%\n")
            #     file.write(f"Average memory percentage for Joint {joint_idx + 1}: {avg_memory_percentage:.2f}%\n")
            #     file.write(f"Average memory usage for Joint {joint_idx + 1}: {avg_memory_usage:.2f} MiB\n")
            #     file.write("-" * 50 + "\n")


            # Record end time
            end_time = time.time()
            # Calculate average memory usage and CPU
            avg_memory_usage = max(mem_usage) - min(mem_usage)
            avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
            prediction_time = end_time - start_time

            print(f"Prediction time: {prediction_time:.2f} seconds")
            print(f"Average memory usage during prediction: {avg_memory_usage:.2f} MiB")
            print(f"Average CPU usage during prediction: {avg_cpu_usage:.2f}%")
            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = y_pred

        # Get the final predicted joint positions (at the last time step)
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)

        # Compute forward kinematics
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)

        print(f"Computed cartesian position: {final_cartesian_pos}")
        print(f"Predicted joint positions at final time step: {final_predicted_joint_positions}")

        # Compute position error
        pred_error_xyz = final_cartesian_pos - goal_position
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")

        # Compute MSE
        true_positions = np.tile(goal_position, (len(test_time_array), 1))
        mse_error = np.mean((final_cartesian_pos - true_positions) ** 2)
        mse_errors.append(mse_error)
        print(f"MSE error for this goal position: {mse_error}")

        # Optional: Visualize the cartesian trajectory over time
        if visualize:
            cartesian_positions_over_time = []
            for i in range(len(test_time_array)):
                joint_positions = predicted_joint_positions_over_time[i, :]
                cartesian_pos, _ = dyn_model.ComputeFK(joint_positions, controlled_frame_name)
                cartesian_positions_over_time.append(cartesian_pos.copy())

            cartesian_positions_over_time = np.array(cartesian_positions_over_time)  # Shape: (num_points, 3)

            # Plot x, y, z positions over time
            plt.figure(figsize=(10, 5))
            plt.plot(test_time_array, cartesian_positions_over_time[:, 0], label='X Position')
            plt.plot(test_time_array, cartesian_positions_over_time[:, 1], label='Y Position')
            plt.plot(test_time_array, cartesian_positions_over_time[:, 2], label='Z Position')

            plt.text(test_time_array[-1], final_cartesian_pos[0], f"Error on X: {pred_error_xyz[0]:.3f} ", color='blue')
            plt.scatter(test_time_array[-1], final_cartesian_pos[0], color='blue')  # Mark error position in X
            plt.text(test_time_array[0], final_cartesian_pos[0], f"X goal: {goal_position[0]:.3f}", color='red')
            plt.plot(test_time_array, true_positions[:, 0], linestyle='--', color='red')  # Mark error position in X

            plt.text(test_time_array[-1], final_cartesian_pos[1], f"Error on Y: {pred_error_xyz[1]:.3f} ",
                     color='orange')
            plt.scatter(test_time_array[-1], final_cartesian_pos[1], color='orange')  # Mark error position in Y
            plt.text(test_time_array[0], final_cartesian_pos[1], f"Y goal: {goal_position[1]:.3f}", color='red')
            plt.plot(test_time_array, true_positions[:, 1], linestyle='--', color='red')  # Mark error position in Y

            plt.text(test_time_array[-1], final_cartesian_pos[2], f"Error on Z: {pred_error_xyz[2]:.3f} ",
                     color='green')
            plt.scatter(test_time_array[-1], final_cartesian_pos[2], color='green')  # Mark error position in Z
            plt.text(test_time_array[0], final_cartesian_pos[2] + 0.03, f"Z goal: {goal_position[2]:.3f}", color='red')
            plt.plot(test_time_array, true_positions[:, 2], linestyle='--', color='red')  # Mark error position in Z

            plt.xlabel('Time (s)')
            plt.ylabel('Cartesian Position (m)')
            plt.title('Predicted Cartesian Positions Over Time')
            plt.legend()
            plt.grid(True)
            image_path = os.path.join(fig_image_dir, f"depth_{depth}_goal_{g}_trajectory_prediction_cartesian.png")
            # plt.savefig(image_path, format='png', dpi=300)  # Save the image
            # plt.show()

            # Plot the trajectory in 3D space
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(cartesian_positions_over_time[:, 0], cartesian_positions_over_time[:, 1], cartesian_positions_over_time[:, 2], label='Predicted Trajectory')
            ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='red', label='Goal Position')
            ax.scatter(final_cartesian_pos[0], final_cartesian_pos[1], final_cartesian_pos[2], marker='*', color='blue', label=f"Final Cartesian Position\nError : {position_error:.6f} \nMSE : {mse_error:.6f}")  # Mark error position in X
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Z Position (m)')
            ax.set_title('Predicted Cartesian Trajectory')
            plt.legend()
            image_path = os.path.join(fig_image_dir, f"depth_{depth}_goal_{g}_trajectory_true_cartesian.png")
            # plt.savefig(image_path, format='png', dpi=300)  # Save the image
            plt.show()

    plt.close()

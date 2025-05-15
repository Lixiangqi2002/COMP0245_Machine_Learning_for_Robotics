import pickle
import subprocess
import threading

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import time
import psutil
from memory_profiler import memory_usage
import pynvml
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE: {device}")

# Set the visualization flag
visualize = True  # Set to True to enable visualization, False to disable
training_flag = True  # Set to True to train the models, False to skip training
test_cartesian_accuracy_flag = True  # Set to True to test the model with a new goal position, False to skip testing
fig_image_dir = "task2/task2_1"
plot_on_screen = True
process_name = "python"

process = psutil.Process()
pynvml.nvmlInit()
# find python process
target_process = None
for process in psutil.process_iter(['pid', 'name']):
    if process.info['name'] == process_name:
        target_process = process
        break
target_pid = target_process.info['pid']
if target_process:
    print(f"Find process: {target_process.info['name']} (PID: {target_pid})")
# find GPU handle
gpu_handle = None
gpu_count = pynvml.nvmlDeviceGetCount()
for gpu_id in range(gpu_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in processes:
        if p.pid == target_pid:
            gpu_handle = handle
            break
    if gpu_handle: break

def monitor_resources(time_sample, cpu_percentages, gpu_percentages, gpu_mem_usages, mem_percentages, mem_usage, process, stop_event):
    while not stop_event.is_set():
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True)
        gpu_memory, gpu_utilization = result.stdout.strip().split(', ')
        print(f"GPU Memory Usage (nvidia-smi): {gpu_memory} MB")
        print(f"GPU Overall Utilization (nvidia-smi): {gpu_utilization}%")
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True)
        for line in result.stdout.strip().split('\n'):
            pid, gpu_memory_used = line.split(', ')
            if int(pid) == target_pid:
                percetage_gpu = 100 * float(gpu_memory_used) / float(gpu_memory)
                print(f"GPU Memory Usage (%): {percetage_gpu}%")
                gpu_percentages.append(float(percetage_gpu))
                print(f"GPU Memory Usage: {gpu_memory_used} MB")
                gpu_mem_usages.append(float(gpu_memory_used))
        cpu_percentages.append(process.cpu_percent(interval=None))
        mem_percentages.append(process.memory_percent())
        mem_usage.append(process.memory_info().rss / (1024 ** 2))
        time.sleep(time_sample)

# MLP Model Definition
class JointAngleRegressor(nn.Module):
    def __init__(self, hidden_units=128):
        super(JointAngleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_units),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(hidden_units, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)


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

        # Custom Dataset Class
        class JointDataset(Dataset):
            def __init__(self, time_data, goal_data, joint_data):
                # Combine time and goal data to form the input features
                x = np.hstack((time_data.reshape(-1, 1), goal_data))  # Shape: (N, 4)
                self.x_data = torch.from_numpy(x).float().to(device)
                self.y_data = torch.from_numpy(joint_data).float().unsqueeze(1).to(device)  # Shape: (N, 1)

            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, idx):
                return self.x_data[idx], self.y_data[idx]

        # Split ratio
        split_ratio = 0.8

        # Initialize lists to hold datasets and data loaders for all joints
        train_loaders = []
        test_loaders = []
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []
        goal_train_list = []
        goal_test_list = []
        # with open("task2/task2_1/training_log.txt", "w") as file:
        #     file.write("#" * 50 + "\n")
        for joint_idx in range(7):
            # Extract joint data
            joint_positions = q_mes_all[:, joint_idx]  # Shape: (N,)

            # Split data
            x_train_time, x_test_time, y_train, y_test, goal_train, goal_test = train_test_split(
                time_array, joint_positions, goal_positions, train_size=split_ratio, shuffle=True
            )

            # Store split data for visualization
            x_train_list.append(x_train_time)
            x_test_list.append(x_test_time)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            goal_train_list.append(goal_train)
            goal_test_list.append(goal_test)

            # Create datasets
            train_dataset = JointDataset(x_train_time, goal_train, y_train)
            test_dataset = JointDataset(x_test_time, goal_test, y_test)

            # Create data loaders
            # train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
            train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

            # test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

            # Store loaders
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        # Training parameters
        epochs = 500
        learning_rate = 0.01

        for joint_idx in range(7):
            # Load the saved model
            model_dir = os.path.join(script_dir, 'task2/task2_1/model/')
            # The name of the saved model
            model_filename = os.path.join(model_dir, f'neuralq{joint_idx+1}.pt')

            # If the save model file exists, assume it's been trained already and skip training it
            if os.path.isfile(model_filename):
                print(f"File {model_filename} exists; assume trained already")
                continue

            print(f'\nTraining model for Joint {joint_idx+1}')

            # Initialize the model, criterion, and optimizer
            model = JointAngleRegressor().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            train_loader = train_loaders[joint_idx]
            test_loader = test_loaders[joint_idx]

            # Start timing
            start_time = time.time()

            # Training loop
            train_losses = []
            test_losses = []

            cpu_percentages = []
            gpu_percentages = []
            gpu_mem_usages = []
            mem_percentages = []
            mem_usage = []
            time_sample = 1
            # Training loop
            stop_event = threading.Event()

            monitor_thread = threading.Thread(target=monitor_resources,
                                              args=(time_sample, cpu_percentages, gpu_percentages, gpu_mem_usages, mem_percentages,
                                                    mem_usage, process, stop_event))
            monitor_thread.start()
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for data, target in train_loader:
                    data = data.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                train_loss = epoch_loss / len(train_loader)
                train_losses.append(train_loss)

                # Evaluate on test set for this epoch
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        test_loss += loss.item()
                test_loss /= len(test_loader)
                test_losses.append(test_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print("-" * 30)
                    print(f"process: {target_process.info['name']} (PID: {target_pid})")
                    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

            stop_event.set()
            # Record end time
            end_time = time.time()
            monitor_thread.join()

            # Calculate average memory usage
            avg_gpu_memory_usage = sum(gpu_mem_usages) / len(gpu_mem_usages)
            avg_gpu_percentage = sum(gpu_percentages) / len(gpu_percentages)
            avg_memory_usage = sum(mem_usage) / len(mem_usage)
            avg_mem_percent = sum(mem_percentages) / len(mem_percentages)
            avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
            training_time = end_time - start_time
            # with open("task2/task2_1/training_log.txt", "a") as file:
            #     file.write(f"Training time for Joint {joint_idx + 1}: {training_time:.2f} seconds\n")
            #     file.write(f"Average GPU memory usage for Joint {joint_idx + 1}: {avg_gpu_memory_usage:.2f} MiB\n")
            #     file.write(f"Average GPU used percentage for Joint {joint_idx + 1}: {avg_gpu_percentage:.2f}%\n")
            #     file.write(f"Average CPU used percentage for Joint {joint_idx + 1}: {avg_cpu_usage:.2f}%\n")
            #     file.write(f"Average memory percentage for Joint {joint_idx + 1}: {avg_mem_percent:.2f}%\n")
            #     file.write(f"Average memory usage for Joint {joint_idx + 1}: {avg_memory_usage:.2f} MiB\n")
            #     file.write("-" * 50 + "\n")

            print(f"Training time for Joint {joint_idx + 1}: {training_time:.2f} seconds")
            print(f"Average GPU memory usage for Joint {joint_idx + 1}: {avg_gpu_memory_usage:.2f}%")
            print(f"Average GPU used percentage for Joint {joint_idx + 1}: {avg_gpu_percentage:.2f}%")
            print(f"Average CPU used percentage for Joint {joint_idx + 1}: {avg_cpu_usage:.2f}%")
            print(f"Average memory percentage for Joint {joint_idx + 1}: {avg_mem_percent:.2f}%")
            print(f"Average memory usage for Joint {joint_idx + 1}: {avg_memory_usage:.2f} MiB")

            # Final evaluation on test set
            print(f'Final Test Loss for Joint {joint_idx + 1}: {test_losses[-1]:.6f}')

            # Save the trained model
            model_filename = os.path.join(model_dir, f'neuralq{joint_idx + 1}.pt')

            torch.save(model.state_dict(), model_filename)
            print(f'Model for Joint {joint_idx+1} saved as {model_filename}')

            # Visualization (if enabled)
            if visualize:
                print(f'Visualizing results for Joint {joint_idx+1}...')

                # Plot training and test loss over epochs
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epochs + 1), train_losses, label=f'Training Loss: {train_loss:.6f}')
                plt.plot(range(1, epochs + 1), test_losses, label=f'Test Loss: {test_loss:.6f}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Loss Curve for Joint {joint_idx+1}')
                plt.legend()
                plt.grid(True)
                image_path = os.path.join(fig_image_dir, f"joint_{joint_idx+1}_loss.png")
                # plt.savefig(image_path, format='png', dpi=300)  # Save the image
                plt.show()

                # Plot true vs predicted positions on the test set
                model.eval()
                with torch.no_grad():
                    x_test_time = x_test_list[joint_idx]
                    y_test = y_test_list[joint_idx]
                    goal_test = goal_test_list[joint_idx]
                    x_test = np.hstack((x_test_time.reshape(-1, 1), goal_test))
                    x_test_tensor = torch.from_numpy(x_test).float().to(device)
                    predictions = model(x_test_tensor).cpu().numpy().flatten()
                    # predictions = model(x_test_tensor).numpy().flatten()

                # Sort the test data for better visualization
                sorted_indices = np.argsort(x_test_time)
                x_test_time_sorted = x_test_time[sorted_indices]
                y_test_sorted = y_test[sorted_indices]
                predictions_sorted = predictions[sorted_indices]

                plt.figure(figsize=(10, 5))
                plt.plot(x_test_time_sorted, y_test_sorted, label='True Joint Positions')
                plt.plot(x_test_time_sorted, predictions_sorted, label='Predicted Joint Positions', linestyle='--')
                plt.xlabel('Time (s)')
                plt.ylabel('Joint Position (rad)')
                plt.title(f'Joint {joint_idx+1} Position Prediction on Test Set')
                plt.legend()
                plt.grid(True)
                image_path = os.path.join(fig_image_dir, f"joint_{joint_idx + 1}_prediction_error.png")
                # plt.savefig(image_path, format='png', dpi=300)  # Save the image
                plt.show()

        print("Training and visualization completed.")

if test_cartesian_accuracy_flag:

    if not training_flag:
        # Load the saved data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'task2/task2_1/model/')
        filename = os.path.join(model_dir, 'data.pkl')  # Replace with your actual filename
        if not os.path.isfile(filename):
            print(f"Error: File {filename} not found in {model_dir}")
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # Extract data
            time_array = np.array(data['time'])            # Shape: (N,)
            #q_mes_all = np.array(data['q_mes_all'])        # Shape: (N, 7)
            #goal_positions = np.array(data['goal_positions'])  # Shape: (N, 3)

            # Optional: Normalize time data for better performance
            # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Testing with a new goal position
    print("\nTesting the model with a new goal position...")
    with open("task2/task2_1/test_log.txt", "w") as file:
        file.write("#" * 50 + "\n")
    # load all the model in a list of models
    models = []
    for joint_idx in range(7):
        # Instantiate the model
        model = JointAngleRegressor().to(device)
        model_dir = os.path.join(script_dir, 'task2/task2_1/model/')

        # Load the saved model
        model_filename = os.path.join(model_dir, f'neuralq{joint_idx+1}.pt')
        try:
            model.load_state_dict(torch.load(model_filename, weights_only=False))

        except FileNotFoundError:
            print(f"Cannot find file {model_filename}")
            print("task_21_goal_pos needs to be run at least once with training_flag=True")
            quit()

        model.eval()
        models.append(model)

    # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    # create a set of goal positions 
    number_of_goal_positions_to_test = 10
    goal_positions = []
    seed = 0
    np.random.seed(seed)
    for i in range(number_of_goal_positions_to_test):
        goal_positions.append([np.random.uniform(*goal_position_bounds['x']),
        np.random.uniform(*goal_position_bounds['y']),
        np.random.uniform(*goal_position_bounds['z'])])
    
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

    position_errors = []
    mse_errors = []
    print(len(goal_positions))
    for g in range(len(goal_positions)):
        goal_position = goal_positions[g]
        print(f"Testing new goal position {g} - {goal_position}------------------------------------------------------------------------")

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (100, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (100, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)
        gpu_percentages = []
        gpu_mem_usages = []
        cpu_percentages = []
        mem_percentages = []
        mem_usage = []
        time_sample = 0.1
        for joint_idx in range(7):
            # Prepare the test input
            test_input_tensor = torch.from_numpy(test_input).float().to(device)  # Shape: (num_points, 4)
            start_time = time.time()

            process = psutil.Process()
            predictions = None

            stop_event = threading.Event()

            monitor_thread = threading.Thread(target=monitor_resources,
                                              args=(time_sample, cpu_percentages, gpu_percentages, gpu_mem_usages, mem_percentages, mem_usage, process, stop_event))
            monitor_thread.start()

            # Predict joint positions
            with torch.no_grad():
                predictions = models[joint_idx](test_input_tensor).cpu().numpy().flatten()  # Shape: (num_points,)

            stop_event.set()
            # Record end time
            end_time = time.time()
            monitor_thread.join()
            # Calculate average memory usage and CPU
            avg_memory_usage = max(mem_usage) - min(mem_usage)
            avg_memory_percentage = max(mem_percentages) - min(mem_percentages)
            avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
            avg_gpu_memory_usage = sum(gpu_mem_usages) / len(gpu_mem_usages)
            avg_gpu_percentage = sum(gpu_percentages) / len(gpu_percentages)
            prediction_time = end_time - start_time

            print(f"Prediction time: {prediction_time:.2f} seconds")
            print(f"Average memory usage during prediction: {avg_memory_usage:.2f} MiB")
            print(f"Average memory percentage usage during prediction (%): {avg_memory_percentage:.2f}%")
            print(f"Average CPU usage during prediction: {avg_cpu_usage:.2f}%")
            print(f"Average GPU usage during prediction: {avg_gpu_percentage:.2f}%")
            print(f"Average GPU usage during prediction: {avg_gpu_memory_usage:.2f} MiB")

            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = predictions

            # with open("task2/task2_1/test_log.txt", "a") as file:
            #     file.write(f"Prediction time for Joint {joint_idx + 1}: {prediction_time:.2f} seconds\n")
            #     file.write(f"Average GPU memory usage for Joint {joint_idx + 1}: {avg_gpu_memory_usage:.2f} MiB\n")
            #     file.write(f"Average GPU used percentage for Joint {joint_idx + 1}: {avg_gpu_percentage:.2f}%\n")
            #     file.write(f"Average CPU used percentage for Joint {joint_idx + 1}: {avg_cpu_usage:.2f}%\n")
            #     file.write(f"Average memory percentage for Joint {joint_idx + 1}: {avg_memory_percentage:.2f}%\n")
            #     file.write(f"Average memory usage for Joint {joint_idx + 1}: {avg_memory_usage:.2f} MiB\n")
            #     file.write("-" * 50 + "\n")

            cpu_percentages.clear()
            mem_percentages.clear()
            mem_usage.clear()
        # with open("task2/task2_1/test_log.txt", "a") as file:
        #     file.write("#" * 100 + "\n")

        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)

        # Compute forward kinematics
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)

        print(f"Goal position: {goal_position}")
        print(f"Computed cartesian position: {final_cartesian_pos}")
        print(f"Predicted joint positions at final time step: {final_predicted_joint_positions}")
        
        # Compute position error
        pred_error_xyz = final_cartesian_pos - goal_position
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")
        position_errors.append(position_error)

        # Compute MSE error
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
            plt.plot(test_time_array, cartesian_positions_over_time[:, 0], label=f'X Position')
            plt.plot(test_time_array, cartesian_positions_over_time[:, 1], label=f'Y Position')
            plt.plot(test_time_array, cartesian_positions_over_time[:, 2], label=f'Z Position')

            plt.xlabel('Time (s)')
            plt.ylabel('Cartesian Position (m)')
            plt.title(f'Predicted Cartesian Positions Over Time')

            # Annotate position error at the final time step
            final_cartesian_pos = cartesian_positions_over_time[-1, :]
            plt.text(test_time_array[-1], final_cartesian_pos[0], f"Error on X: {pred_error_xyz[0]:.3f} ", color='blue')
            plt.scatter(test_time_array[-1], final_cartesian_pos[0], color='blue')  # Mark error position in X
            plt.text(test_time_array[0], final_cartesian_pos[0], f"X goal: {goal_position[0]:.3f}", color='red')
            plt.plot(test_time_array, true_positions[:,0],linestyle='--', color='red')  # Mark error position in X

            plt.text(test_time_array[-1], final_cartesian_pos[1], f"Error on Y: {pred_error_xyz[1]:.3f} ", color='orange')
            plt.scatter(test_time_array[-1], final_cartesian_pos[1], color='orange')  # Mark error position in Y
            plt.text(test_time_array[0], final_cartesian_pos[1], f"Y goal: {goal_position[1]:.3f}", color='red')
            plt.plot(test_time_array, true_positions[:,1], linestyle='--',color='red')  # Mark error position in Y

            plt.text(test_time_array[-1], final_cartesian_pos[2], f"Error on Z: {pred_error_xyz[2]:.3f} ", color='green')
            plt.scatter(test_time_array[-1], final_cartesian_pos[2], color='green')  # Mark error position in Z
            plt.text(test_time_array[0], final_cartesian_pos[2]+0.03, f"Z goal: {goal_position[2]:.3f}", color='red')
            plt.plot(test_time_array, true_positions[:,2], linestyle='--',color='red')  # Mark error position in Z

            plt.legend()
            plt.grid(True)
            image_path = os.path.join(fig_image_dir, f"goal_{g}_trajectory_prediction_cartesian.png")
            # plt.savefig(image_path, format='png', dpi=300)  # Save the image
            if plot_on_screen:
                plt.show()


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
            image_path = os.path.join(fig_image_dir, f"goal_{g}_trajectory_true_cartesian.png")
            # plt.savefig(image_path, format='png', dpi=300)  # Save the image
            if plot_on_screen:
                plt.show()

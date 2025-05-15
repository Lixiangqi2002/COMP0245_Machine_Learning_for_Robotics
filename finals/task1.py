import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import product
import seaborn as sns
import pandas as pd
from pandas.plotting import parallel_coordinates
import time
import os

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10  # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    # m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m

    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real

    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)

    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Dataset
dataset = TensorDataset(X_tensor, Y_tensor)

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self, hidden_nodes, two_hidden_layers=False, three_hidden_layers=False):
        super(MLP, self).__init__()

        if two_hidden_layers:
            self.layers = nn.Sequential(
                nn.Linear(4, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, 1)
            )
        elif three_hidden_layers:
            self.layers = nn.Sequential(
                nn.Linear(4, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, 1)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(4, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, 1)
            )

    def forward(self, x):
        return self.layers(x)


def training_loop(model, optimizer, train_loader):
    criterion = nn.MSELoss()

    epochs = 1000  # 1000
    train_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}')

    end_time = time.time()

    training_time = end_time - start_time

    return train_losses, training_time


def testing_loop(model):
    criterion = nn.MSELoss()

    # Testing Phase: Simulate trajectory tracking
    q_test = 0
    dot_q_test = 0
    q_real = []
    q_real_corrected = []
    testing_loss = []

    # integration with only PD Control
    for i in range(len(t)):
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        ddot_q_real = (tau - b * dot_q_test) / m
        dot_q_test += ddot_q_real * dt
        q_test += dot_q_test * dt
        q_real.append(q_test)

    q_test = 0
    dot_q_test = 0
    for i in range(len(t)):
        # Apply MLP correction
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32)
        correction = model(inputs.unsqueeze(0)).item()
        ddot_q_corrected = (tau - b * dot_q_test + correction) / m

        dot_q_test += ddot_q_corrected * dt
        q_test += dot_q_test * dt
        q_real_corrected.append(q_test)

        target = torch.tensor([q_target[i]], dtype=torch.float32)
        predicted = torch.tensor([q_test], dtype=torch.float32)
        # loss = criterion(predicted, target).item()
        loss = mean_squared_error(predicted, target)
        testing_loss.append(loss)

    return q_real, q_real_corrected, testing_loss


def model_setup(task, param_lst, deep_nn_2=False, deep_nn_3=False):
    q_real_runs = []
    q_real_corrected_runs = []
    training_loss_runs = []
    training_time_runs = []
    testing_loss_runs = []

    for param in param_lst:
        match task:
            case 'hidden_nodes':
                model = MLP(param, deep_nn_2, deep_nn_3)
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            case 'learning_rate':
                model = MLP(32, deep_nn_2, deep_nn_3)
                optimizer = optim.Adam(model.parameters(), lr=param)
                train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            case 'batch_size':
                model = MLP(32, deep_nn_2, deep_nn_3)
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                train_loader = DataLoader(dataset, batch_size=param, shuffle=True)

        train_losses, training_time = training_loop(model, optimizer, train_loader)
        q_real, q_real_corrected, testing_loss = testing_loop(model)

        training_loss_runs.append(train_losses)
        testing_loss_runs.append(testing_loss)
        training_time_runs.append(training_time)
        q_real_runs.append(q_real)
        q_real_corrected_runs.append(q_real_corrected)

    # -------------- File + Folder Generation --------------

    folder_name = "task1"

    file_path_new = os.path.join("task1", task)

    if deep_nn_2:
        file_path_new = os.path.join(file_path_new, 'deep_2_layer')
    elif deep_nn_3:
        file_path_new = os.path.join(file_path_new, 'deep_3_layer')
    else:
        file_path_new = os.path.join(file_path_new, 'shallow')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists(file_path_new):
        os.makedirs(file_path_new)

    txt_title = f'MLP Correction {task}:'
    mse_values = []

    mse_file = os.path.join(file_path_new, 'mse.txt')
    open(mse_file, 'w').close()
    with open(mse_file, "a") as file:
        file.write("")
        file.write(txt_title + "\n")
        # for i in range(len(param_lst)):

    # -------------- Plotting --------------

    cmap = plt.get_cmap('viridis', len(param_lst) * 2)
    plt.figure(figsize=(12, 6))
    plt.plot(t, q_target, 'r-', label='Target')
    plt.plot(t, q_real_runs[-1], 'b--', label=f'PD Only')

    for i in range(len(param_lst)):
        plt.plot(t, q_real_corrected_runs[i], linestyle=':', color=cmap(i * 2),
                 label=f'PD + MLP Correction {task}: {param_lst[i]}')
        mse = mean_squared_error(q_target, q_real_corrected_runs[i])
        mse_values.append(({param_lst[i]}, mse))
        print(f'MSE {task} - {param_lst[i]}: ', mse)


    plt.title('Trajectory Tracking with and without MLP Correction')
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.legend()
    # plt.savefig((file_path_new + '/trajectory.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    for i in range(len(param_lst)):
        plt.plot(t, np.log(training_loss_runs[i]), linestyle='--', color=cmap(i * 2),
                 label=f'Training Loss {task}: {param_lst[i]}')
    plt.title('Log of Training Loss')
    plt.xlabel('Time [s]')
    plt.ylabel('MSE Loss')
    plt.legend()
    # plt.savefig(file_path_new + '/training_log_loss.png')

    plt.figure(figsize=(12, 6))
    for i in range(len(param_lst)):
        plt.plot(t, testing_loss_runs[i], linestyle='--', color=cmap(i * 2),
                 label=f'Testing Loss {task}: {param_lst[i]}')
    plt.title('Testing Loss')
    plt.xlabel('Time [s]')
    plt.ylabel('MSE Loss')
    plt.legend()
    # plt.savefig(file_path_new + '/testing_log_loss.png')
    plt.close()

    avg_train_loss = []
    training_loss_runs = np.array(training_loss_runs)
    for i in range(len(param_lst)):
        final_train_losses = training_loss_runs[i][-100:]  # 获取最后10个epoch的损失值
        avg_train_loss.append(sum(final_train_losses) / len(final_train_losses))
        print(f"Average Training Loss (last 100 epochs): {avg_train_loss[-1]}")
        # min_train_loss.append(np.array(training_loss_runs[i]).min())
    with open(mse_file, "a") as file:
        for i, tup in enumerate(mse_values):
            line_text = f"Trajectory MSE - {tup[0]}: {tup[1]}\nTraining Time: {training_time_runs[i]}\n"
            # file.write(line_text)
            # file.write(f'Average Training Loss (last 100 epochs) for {task}={tup[0]}: {avg_train_loss[i]} \n')


def run_experiment(num_hidden_nodes, learning_rate, batch_size, nn_layer):
    deep_n_2, deep_n_3 = False, False
    if nn_layer==1:
        deep_n_2, deep_n_3=False, False
    elif nn_layer==2:
        deep_n_2, deep_n_3 = True, False
    elif nn_layer==3:
        deep_n_2, deep_n_3 = False, True
    model = MLP(num_hidden_nodes, deep_n_2, deep_n_3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, training_time = training_loop(model, optimizer, train_loader)
    q_real, q_real_corrected, testing_loss = testing_loop(model)

    mse = mean_squared_error(q_target, q_real_corrected)
    return train_losses, training_time, q_real, q_real_corrected, testing_loss, mse


def run_all_combinations(model_dict):
    results = []

    keys = model_dict.keys()
    values = model_dict.values()

    counter = 0
    for combination in product(*values):
        counter += 1

    print('Length: ', counter)

    counter2 = 1
    for combination in product(*values):
        print('Comb: ', combination)
        print(f'Experiment {counter2} of {counter}')
        counter2 += 1

        train_losses, training_time, q_real, q_real_corrected, testing_loss, mse = run_experiment(combination[0],
                                                                                             combination[1],
                                                                                             combination[2],
                                                                                             combination[3])
        # Combinations 0: num_hidden_nodes, 1: learning_rate, 2: batch_size, 3: deep_nn
        # print(train_losses)
        train_losses.reverse()
        # print(train_losses)
        training_loss_runs = np.array(train_losses)
        f = 0
        final_train_losses = 0
        for cur in range(100):
            final_train_losses += training_loss_runs[cur]
            f+=1
        # final_train_losses = training_loss_runs[0:100]
        avg_train_loss= final_train_losses / f
        print(f"Average Training Loss 100  : {avg_train_loss}")
        print(f"MSE for trajectory testing : {mse}")
        result_entry = {
            'train_loss': avg_train_loss,
            'training_time': training_time,
            # 'q_real': q_real,
            # 'q_real_corrected': q_real_corrected,
            'testing_loss': mse,
            'learning_rate': combination[1],
            'batch_size': combination[2],
            'hidden_nodes': combination[0],
            'deep_nn': combination[3]
        }

        results.append(result_entry)

    df = pd.DataFrame(results)
    file_path = os.path.join("task1", "experiment_results_final.csv")
    # experiment_results = pd.read_csv(file_path)
    # df.to_csv(file_path, index=False)

    print('Results Saved ------------------------------------')

    return results


def plot_heatmaps_by_deep_nn(data, x_param, y_param, metric):
    df = pd.DataFrame(data)

    if metric == 'train_loss':
        # df[metric] = df[metric].apply(lambda x: np.log(eval(x)[-1]) if isinstance(x, str) else x)
        df[metric] = df[metric].apply(lambda x: np.log(x + 1e-8) if x > 0 else 0)


    elif metric == 'testing_loss':
        df[metric] = df[metric].apply(lambda x: np.log(x + 1e-8) if x > 0 else 0)

        # df[metric] = df[metric]#.apply(lambda x: np.mean([np.log(val) for val in eval(x)]))

    # Filter data for deep_nn=1,2,3
    for deep_nn_value in [1,2,3]:
        filtered_df = df[df['deep_nn'] == deep_nn_value]

        heatmap_data = filtered_df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc='mean')

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': metric})
        plt.title(f"Heatmap of average {metric} by {x_param} and {y_param} (deep_nn={deep_nn_value})")
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        # plt.savefig(f'task1/heatmap/{metric}_by_{x_param}_and_{y_param}_deepNN_{deep_nn_value}.png')
        plt.show()


if __name__ == '__main__':
    hidden_nodes = [32, 64, 96, 128]  # list shows values of the parameters
    learning_rate = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    batch_size = [64, 128, 256, 1000]

    model_dict = {
        'hidden_nodes': [32, 64, 96, 128, 256],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'batch_size': [32, 64, 128, 256, 1000],
        'nn_layer': [1, 2, 3]
    }

    # task 1.1 & task 1.2
    model_setup('hidden_nodes', hidden_nodes)
    model_setup('hidden_nodes', hidden_nodes, True)
    model_setup('hidden_nodes', hidden_nodes, False, True)

    # # task 1.3
    # model_setup('learning_rate', learning_rate)
    # model_setup('learning_rate', learning_rate, True)
    # model_setup('learning_rate', learning_rate, False, True)

    # #  task 1.4
    # model_setup('batch_size', batch_size)
    # model_setup('batch_size', batch_size, True)
    # model_setup('batch_size', batch_size, False, True)

    # # heatmap
    # file_path = 'task1/experiment_results_final.csv'
    # file_path = os.path.join("task1", "experiment_results_final.csv")
    # experiment_results = pd.read_csv(file_path)
    # experiment_results = run_all_combinations(model_dict)
    # for metric in ['train_loss', 'testing_loss', 'training_time']:
    #     # plot_heatmaps_by_deep_nn change: train_loss, testing_loss, training_time
    #     plot_heatmaps_by_deep_nn(experiment_results, 'hidden_nodes', 'learning_rate', metric)
    #     plot_heatmaps_by_deep_nn(experiment_results, 'hidden_nodes', 'batch_size', metric)
    #     plot_heatmaps_by_deep_nn(experiment_results, 'learning_rate', 'batch_size', metric)


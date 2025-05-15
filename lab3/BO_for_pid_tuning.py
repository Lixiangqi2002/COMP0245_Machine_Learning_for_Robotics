import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from skopt import gp_minimize
from plots import plot_objective, plot_evaluations, plot_convergence
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from gp_functions import fit_gp_model_1d, plot_gp_results_1d

damping = False

if damping==False:
    # Define the search space for Kp and Kd
    lower_bound_kp = 0.1  # 5.0
    lower_bound_kd = 0.0  # 5.0
    upper_bound_kp = 1000  # 50
    upper_bound_kd = 100  # 45
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot

else:
    lower_bound_kp = 5.0
    lower_bound_kd = 5.0
    upper_bound_kp = 50
    upper_bound_kd = 45
    conf_file_name = "pandaconfig_damping.json"  # Configuration file for the robot

# Configuration for the simulation
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")

# Sinusoidal reference
# Specify different amplitude values for each joint
amplitudes = [np.pi / 4, np.pi / 6, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4,
              np.pi / 4]  # Example amplitudes for joints
# Specify different frequency values for each joint
frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

# Convert lists to NumPy arrays for easier manipulation in computations
amplitude = np.array(amplitudes)
frequency = np.array(frequencies)
ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference

# Global lists to store data
kp0_values = []
kd0_values = []
tracking_errors = []


def simulate_with_given_pid_values(kp, kd, episode_duration, plot=True, joints_id=0, image_dir="report_figure/"):
    # here we reset the simulator each time we start a new test
    sim.ResetPose()

    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0] * dyn_model.getNumberofActuatedJoints())

    time_step = sim.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, = [], [], [], []

    steps = int(episode_duration / time_step)
    # print(steps)
    # testing loop
    for i in range(steps):
        if i<500:
            continue
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)

        # Compute sinusoidal reference trajectory
        q_des, qd_des = ref.get_values(current_time)  # Desired position and velocity
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step

    # Calculate tracking error
    q_mes_all = np.array(q_mes_all)
    q_des_all = np.array(q_d_all)


    if plot:
        plt.figure(figsize=(10, 5))

        time_vector = np.linspace(0, episode_duration, q_mes_all.shape[0])
        plt.plot(time_vector, q_mes_all[:,joints_id], label=f'Joint ' + str(joints_id + 1), color='b', linestyle='--', linewidth=2)
        plt.plot(time_vector, q_des_all[:,joints_id], label=f'Reference for Joint ' + str(joints_id + 1), color='r',
                 linestyle='-', linewidth=1)

        plt.title(f'Measured Joint Angles with Kp = {kp}, Kd = {kd}')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angle (rad)')
        plt.grid()
        plt.legend()
        # Ensure the directory exists before saving
        # image_dir = "report_figure/damping"

        image_path = os.path.join(image_dir, f"joint_{joints_id + 1}_kp_{kp}_kd_{kd}.png")
        # plt.savefig(image_path, format='png', dpi=300)  # Save the image
        print(f"Image saved!")
        plt.show()
        plt.close()

    tracking_error = np.sum((q_mes_all - q_des_all) ** 2/10)  # Sum of squared error as the objective
    # print tracking error
    print("###################################################################")
    print("Tracking error: ", tracking_error)
    # print PD gains
    print("kp: ", kp)
    print("kd: ", kd)

    return tracking_error


# Objective function for optimization
def objective(params):
    kp = np.array(params[:7])  # First 7 elements correspond to kp
    kd = np.array(params[7:])  # Last 7 elements correspond to kd
    episode_duration = 10

    # TODO Call the simulation with given kp and kd value
    tracking_error = simulate_with_given_pid_values(kp=kp, kd=kd, episode_duration=episode_duration, plot=False)
    # TODO Collect data for the first kp and kd
    # Store the current kp, kd, and tracking error values for further analysis
    kp0_values.append(kp[0])
    kd0_values.append(kd[0])
    tracking_errors.append(tracking_error)
    # return tracking_error
    return tracking_error


def main():
    space = [
                Real(lower_bound_kp, upper_bound_kp, name=f'kp{i}') for i in range(7)
            ] + [
                Real(lower_bound_kd, upper_bound_kd, name=f'kd{i}') for i in range(7)
            ]

    # RBF kernel for Gaussian Process
    rbf_kernel = RBF(
        length_scale=1.0,  # Initial length scale
        length_scale_bounds=(1e-2, 1e2)  # Bounds for length scale
    )

    # Gaussian Process Regressor
    gp = GaussianProcessRegressor(
        kernel=rbf_kernel,
        normalize_y=True,
        n_restarts_optimizer=10  # Optional for better hyperparameter optimization
    )

    # Perform Bayesian optimization for different acquisition functions
    acquisition_funcs = ['LCB', 'EI', 'PI']
    results = []
    for acq_func in acquisition_funcs:
        result = gp_minimize(
            objective,
            space,
            n_calls=20,
            base_estimator=gp,
            acq_func=acq_func,
            random_state=42
        )
        results.append(result)
        print(f"Optimization using {acq_func} completed.")

    # Compare convergence
    plt.figure(figsize=(10, 6))
    plot_convergence((acquisition_funcs[1], results[1]),
            (acquisition_funcs[0],results[0]),
                     (acquisition_funcs[2], results[2]))
    plt.title("Convergence Comparison for LCB, EI, and PI")
    # plt.savefig("report_figure/damping/convergence_comparison.png")
    plt.show()

    # Prepare data for further GP model fitting and simulation (as before)
    kp0_values_array = np.array(kp0_values).reshape(-1, 1)
    kd0_values_array = np.array(kd0_values).reshape(-1, 1)
    tracking_errors_array = np.array(tracking_errors)

    # Extract best results from each acquisition function
    for idx, acq_func in enumerate(acquisition_funcs):
        if damping :
            image_dir = "report_figure/damping/" + acquisition_funcs[idx]
        else:
            image_dir = "report_figure/no_damping/" + acquisition_funcs[idx]


        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        with open(image_dir+"/results.txt", "w") as f:
            best_kp = results[idx].x[:7]  # Optimal kp vector
            best_kd = results[idx].x[7:]  # Optimal kd vector
            optimal_tracking_error = results[idx].fun

            # Plot the objectives and evaluations for the acquisition function
            plot_objective(results[idx], save_diagonal_plots=True, save_contour_plots=True, save_dir=image_dir + "/One_Dimension_Effect")
            plot_evaluations(results[idx], save_diagonal_plots=True, save_scatter_plots=True, save_dir=image_dir + "/One_Dimension_Effect")

            print(f"Optimal Kp ({acq_func}): {best_kp}")
            print(f"Optimal Kd ({acq_func}): {best_kd}")
            print(f"Optimal tracking error ({acq_func}): {optimal_tracking_error}")
            # Write results to file
            f.write(f"Optimal Kp ({acq_func}): {best_kp}\n")
            f.write(f"Optimal Kd ({acq_func}): {best_kd}\n")
            f.write(f"Optimal tracking error ({acq_func}): {optimal_tracking_error}\n")
            f.write("\n")  # Add a blank line between each section

            # Fit GP models
            length_scale = [0.001, 0.01, 0.1, 1]
            for i in length_scale:
                # fit a new GP Model with different length_scale
                gp_kd0 = fit_gp_model_1d(kd0_values_array, tracking_errors_array, length_scale=i)
                gp_kp0 = fit_gp_model_1d(kp0_values_array, tracking_errors_array, length_scale=i)
                # Plot the results
                plot_name = image_dir + f"/RBF_parameter/cost_{result.fun}_LS_{i}.png"
                plot_gp_results_1d(kp0_values_array, kd0_values_array, tracking_errors_array, gp_kp0, gp_kd0, lower_bound_kp, lower_bound_kd, upper_bound_kp, upper_bound_kd, plot_name)

            # Simulation for each joint
            # for joint_id in range(7):
                # simulate_with_given_pid_values(best_kp[joint_id], best_kd[joint_id], 20, True, joint_id, image_dir)




if __name__ == "__main__": 
    main()

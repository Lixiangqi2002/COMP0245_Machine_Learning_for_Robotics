import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference

# Configuration for the simulation
conf_file_name = "pandaconfig_damping.json"  # Configuration file for the robot
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



def simulate_with_given_pid_values(sim_, kp, kd, episode_duration):
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()

    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0] * dyn_model.getNumberofActuatedJoints())

    time_step = sim_.GetTimeStep()
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
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)

        # Compute sinusoidal reference trajectory
        q_des, qd_des = ref.get_values(current_time)  # Desired position and velocity
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
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

    # # save the steady oscillation for current kp value
    # if plot:
    #     # for joints_id in range(7):
    #     plt.figure(figsize=(10, 5))
    #
    #     time_vector = np.linspace(0, episode_duration, q_mes_all.shape[0])
    #     plt.plot(time_vector, q_mes_all[:,joints_id], label=f'Joint ' + str(joints_id + 1), color='b', linestyle='--', linewidth=2)
    #     plt.plot(time_vector, q_des_all[:,joints_id], label=f'Reference for Joint ' + str(joints_id + 1), color='r',
    #              linestyle='-', linewidth=1)
    #
    #     plt.title(f'Measured Joint Angles with Kp = {kp}, Kd = {kd}')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Joint Angle (rad)')
    #     plt.grid()
    #     plt.legend()
    #     # Ensure the directory exists before saving
    #     # image_dir = "report_figure/noise"
    #
    #
    #     image_path = os.path.join(image_dir, f"joint_{joints_id + 1}_kp_{kp}_kd_{kd}.png")
    #     plt.savefig(image_path, format='png', dpi=300)  # Save the image
    #     print(f"Image saved!")
    #     # plt.show()
    #     plt.close()

    tracking_error = np.sum((q_mes_all - q_des_all) ** 2/10)  # Sum of squared error as the objective
    # print tracking error
    print("###################################################################")
    print("Tracking error: ", tracking_error)
    # print PD gains
    print("kp: ", kp)
    print("kd: ", kd)

    return q_mes_all, q_des_all, tracking_error


def main():
    global q_mes_zn, q_mes_gp, q_des_gp, tracking_errors_zn, tracking_errors_gp
    kp_candidate = [[13.28, 12.8, 6.76, 1.6, 13.28, 12.6, 13.2],[46.8363943554158, 41.36541708039877, 33.50316904296906, 44.21572655844731, 41.16524346046016,  13.395652649871614, 45.165154932049006]]
    kd_candidate = [[2.49, 2.67, 1.81, 0.8695, 2.49, 2.48, 2.60],[26.57368967662603, 37.29760620656251, 40.84365199693974, 17.720138998874557, 9.402076981107072,14.117406501677669, 22.084311545050255]]
    # condition = ['Z_N', 'GP']  # , 174.8268801631354
    episode_duration = 20
    for i in range(2):
        kp = kp_candidate[i]
        kp = np.array(kp)
        kd = kd_candidate[i]
        kd = np.array(kd)
        # for joint_id in range(7):
        if i == 0:
            q_mes_zn, q_des_zn, tracking_errors_zn = simulate_with_given_pid_values(sim, kp, kd, 20)
        elif i ==1:
            q_mes_gp, q_des_gp, tracking_errors_gp = simulate_with_given_pid_values(sim, kp, kd, 20)


    for i in range(7):
        image_dir = "report_figure/compare_GP_ZN/"
        plt.figure(figsize=(10, 5))

        time_vector = np.linspace(0, episode_duration, q_mes_zn.shape[0])
        plt.plot(time_vector, q_mes_zn[:, i], label=f'Z-N Joint ' + str(i + 1), color='b', linestyle='--',
                 linewidth=2)
        plt.plot(time_vector, q_mes_gp[:, i], label=f'GP Joint ' + str(i + 1), color='g', linestyle='--',
                 linewidth=2)
        plt.plot(time_vector, q_des_gp[:, i], label=f'Reference for Joint ' + str(i + 1), color='r',
                 linestyle='-', linewidth=1)

        plt.title(f'Measured Joint Angles Tracking Error for Z-N is {tracking_errors_zn}, while for GP is {tracking_errors_gp} ')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angle (rad)')
        plt.grid()
        plt.legend()
        # Ensure the directory exists before saving
        # image_dir = "report_figure/damping"

        image_path = os.path.join(image_dir,f"joint_{i + 1}.png")
        plt.savefig(image_path, format='png', dpi=300)  # Save the image
        print(f"Image saved!")
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
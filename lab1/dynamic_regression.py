import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity

        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)
        # print(tau_mes.shape) # 7
        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        # print(cur_regressor.shape) 
        current_time += time_step
        if current_time < 0.8: # 0.8 needs to be validated
            continue
        else:
            # Store the measured torque and regressor
            tau_mes_all.append(tau_mes)
            regressor_all.append(cur_regressor)
       
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    # stacking.....
    tau_mes_all = np.hstack(tau_mes_all)
    regressor_all = np.vstack(regressor_all)
    
    print(regressor_all.shape)
    print(tau_mes_all.shape)
    print(np.linalg.pinv(regressor_all).shape)

    inverse =  np.linalg.pinv(regressor_all)
    a = inverse @ tau_mes_all

    print("Computed coefficients:", a)
    print(a.shape) #(70,)

    # get predictions
    tau_pred = regressor_all @ a

    # TODO compute the metrics for the linear model
    residuals = tau_mes_all - tau_pred
    rss = np.sum(residuals ** 2)
    tss = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)
    n, p = regressor_all.shape

    # step 1: The r2 and adjusted r2
    r2 = 1 -(rss/tss)
    print(r2)
    r2 = r2_score(tau_mes_all, tau_pred)
    print(r2)
    adjusted_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    print("Coefficient of determination: %.4f" % r2)
    print("adjusted r2: %.4f" % adjusted_r2)

    # step 2: MSE
    mse = rss/(n-p-1)
    print(mse)
    mse = mean_squared_error(tau_mes_all, tau_pred)
    print(mse)
    print("Mean squared error: %.4f" % mean_squared_error(tau_mes_all, tau_pred))

    # step 3: compute F-statistics
    f_statistic = (tss-rss)/p/(rss/(n-p-1))
    p_value = 1 - stats.f.cdf(f_statistic, p, n-p-1)
    print("F-statistics: %.4f" % f_statistic)
    print("P-value: %.4f" % p_value)

    # step 4: confidence intervals for coeff
    se = np.diagonal(mse * np.linalg.pinv(regressor_all.T @ regressor_all))
    se = np.maximum(se, 0)
    se = np.sqrt(se)
    print(se.shape)
    print(tau_pred.shape)
    # Confidence intervals for each coeff
    coeff_i_lower = a - 1.96 * se
    coeff_i_upper = a + 1.96 * se

    # Step 5: Prediction intervals
    pred_se = np.sqrt(mse * (1 + np.sum(np.dot(regressor_all, np.linalg.pinv(np.dot(regressor_all.T, regressor_all))) * regressor_all, axis=1)))
    ci_lower = tau_pred - 1.96 * pred_se
    ci_upper = tau_pred + 1.96 * pred_se

    
    # TODO plot the torque prediction error for each joint (optional)
    # After computing tau_pred in your main function
    # Compute prediction error
    prediction_error = tau_mes_all - tau_pred
    prediction = tau_pred.reshape(-1, num_joints)
    # Reshape prediction_error
    prediction_error = prediction_error.reshape(-1, num_joints)
    # Check the shapes of tau_mes_all and tau_pred
    print(f"Shape of tau_mes_all: {tau_mes_all.shape}")
    print(f"Shape of tau_pred: {tau_pred.shape}")

    # Reshape prediction_error to match the number of joints
    num_measurements = tau_mes_all.shape[0]
    num_joints = len(amplitudes)
    if num_measurements % num_joints != 0:
        raise ValueError("The number of measurements is not divisible by the number of joints.")

    # Time vector for plotting
    time_vector_1 = np.linspace(0, max_time, prediction.shape[0])
    time_vector_2 = np.linspace(0, max_time, prediction_error.shape[0])

    plt.figure(figsize=(12, 8))
    for joint_index in range(num_joints):
        plt.subplot(num_joints, 1, joint_index + 1)
        # Get measured and predicted torque for the current joint
        measured_torque = tau_mes_all[joint_index::num_joints]
        predicted_torque = tau_pred[joint_index::num_joints]

        # Confidence intervals for the current joint
        ci_lower_joint = ci_lower[joint_index::num_joints]
        ci_upper_joint = ci_upper[joint_index::num_joints]

        # Plot measured torque
        plt.plot(time_vector_1, measured_torque, label='Measured Torque', color='blue', alpha=0.3)
        # Plot predicted torque
        plt.plot(time_vector_1, predicted_torque, label='Predicted Torque', color='red', linestyle='--', alpha=0.5)
        
        # Plot confidence intervals
        plt.fill_between(time_vector_1, ci_lower_joint, ci_upper_joint, color='gray', alpha=0.2, label='95% Confidence Interval')


        plt.title(f'Torque Prediction for Joint {joint_index + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Prediction Torque(Nm)')
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(12, 8))
    for joint_index in range(num_joints):
        plt.subplot(num_joints, 1, joint_index + 1)
        plt.plot(time_vector_2, prediction_error[:, joint_index], label=f'Joint {joint_index + 1}')
        plt.title(f'Torque Prediction Error for Joint {joint_index + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Prediction Error (Nm)')
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    main()

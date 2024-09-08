import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import generate_rssi_data 
from simulation import exponential_smoothing, kalman_filter, arima_model, lstm_model, particle_filter, evaluate_method

def run_simulation(method_name, method_func, data, args=[]):
    print(f"Running simulation for method: {method_name}")
    print(f"Type of data: {type(data)}")
    print(f"First 10 elements of data: {data[:10]}")
    name, mse, predicted = evaluate_method(method_name, method_func, data, args)
    return (name, mse, predicted)

if __name__ == "__main__":
    # Generate RSSI data
    rssi_data = generate_rssi_data()

    # Debugging statements
    print(f"Type of rssi_data: {type(rssi_data)}")
    print(f"First 10 elements of rssi_data: {rssi_data[:10]}")

    # Ensure rssi_data is a list or array
    if not isinstance(rssi_data, (list, np.ndarray)):
        raise TypeError("rssi_data should be a list or numpy array")

    # Define the methods to compare
    methods = [
        ("Exponential Smoothing", exponential_smoothing, [0.2]),
        ("Kalman Filter", kalman_filter, []),
        ("ARIMA", arima_model, []),
        ("LSTM", lstm_model, [10, 10]),
        ("Particle Filter", particle_filter, [])
    ]

    # Run simulations sequentially
    results = [run_simulation(name, func, rssi_data, args) for name, func, args in methods]

    # Calculate the index to start plotting and evaluating
    start_index = int(len(rssi_data) * 0.05)

    # Plot Results
    plt.figure(figsize=(10, 6))
    for name, mse, predicted in results:
        # Calculate MSE excluding the first 5%
        mse_excluding_start = np.mean((np.array(rssi_data[start_index:]) - np.array(predicted[start_index:])) ** 2)
        plt.plot(predicted[start_index:], label=f"{name} (MSE: {mse_excluding_start:.4f})")

    # Plot the original RSSI data excluding the first 5%
    plt.plot(rssi_data[start_index:], label="True RSSI", color="black", linestyle="dashed")

    plt.title("RSSI Prediction Comparison")
    plt.xlabel("Time")
    plt.ylabel("RSSI")
    plt.legend()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import generate_rssi_data 
from simulation import exponential_smoothing, kalman_filter, arima_model, lstm_model, particle_filter, run_simulation
import multiprocessing as mp

def run_multiple_simulations(num_runs=10):
    methods = [
        ("Exponential Smoothing", exponential_smoothing, [0.2]),
        ("Kalman Filter", kalman_filter, []),
        ("ARIMA", arima_model, []),
        ("LSTM", lstm_model, [20, 20]),
        ("Particle Filter", particle_filter, [])
    ]

    method_mse = {name: [] for name, _, _ in methods}

    for run in range(num_runs):
        # Generate random parameters for each run
        num_points = 3600
        base_rssi = np.random.uniform(-90, -40)
        trend_strength = np.random.uniform(0.005, 0.12)
        noise_std = np.random.uniform(0.1, 6)
        seasonal_period = np.random.randint(1000, 3000)
        seasonal_amplitude = np.random.uniform(0.01, 5)
        
        # Generate RSSI data with random parameters
        rssi_data = generate_rssi_data(
            num_points=num_points,
            base_rssi=base_rssi,
            trend_strength=trend_strength,
            noise_std=noise_std,
            seasonal_period=seasonal_period,
            seasonal_amplitude=seasonal_amplitude,
            random_seed=run
        )

        # Ensure rssi_data is a list or array
        if not isinstance(rssi_data, (list, np.ndarray)):
            raise TypeError("rssi_data should be a list or numpy array")

        # Create a queue to collect results
        result_queue = mp.Queue()

        # Create and start a process for each method
        processes = []
        for name, func, args in methods:
            p = mp.Process(target=run_simulation, args=(name, func, rssi_data, args, result_queue))
            processes.append(p)
            p.start()

        # Collect results from all processes
        results = [result_queue.get() for _ in methods]

        # Ensure all processes have finished
        for p in processes:
            p.join()

        # Calculate the index to start plotting and evaluating
        start_index = int(len(rssi_data) * 0.05)

        # Plot Results for the current run
        plt.figure(figsize=(10, 6))
        for name, mse, predicted in results:
            # Calculate MSE excluding the first 5%
            mse_excluding_start = np.mean((np.array(rssi_data[start_index:]) - np.array(predicted[start_index:])) ** 2)
            method_mse[name].append(mse_excluding_start)
            plt.plot(predicted[start_index:], label=f"{name} (MSE: {mse_excluding_start:.4f})")

        # Plot the original RSSI data excluding the first 5%
        plt.plot(rssi_data[start_index:], label="True RSSI", color="black", linestyle="dashed")

        plt.title(f"RSSI Prediction Comparison - Run {run + 1}")
        plt.xlabel("Time")
        plt.ylabel("RSSI")
        plt.legend()
        #plt.show()

    # Calculate average MSE for each method
    avg_mse = {name: np.mean(mse_list) for name, mse_list in method_mse.items()}

    # Sort methods by average MSE
    sorted_methods = sorted(avg_mse.items(), key=lambda item: item[1])

    # Display ranking
    print("Method Ranking based on Average MSE:")
    for rank, (name, mse) in enumerate(sorted_methods, start=1):
        print(f"{rank}. {name} - Average MSE: {mse:.4f}")

    # Plot aggregated MSE results
    plt.figure(figsize=(10, 6))
    for name, mse_list in method_mse.items():
        plt.plot(mse_list, label=f"{name} (Avg MSE: {avg_mse[name]:.4f})")

    plt.title("RSSI Prediction Comparison Across Multiple Runs")
    plt.xlabel("Run")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_multiple_simulations(num_runs=100)
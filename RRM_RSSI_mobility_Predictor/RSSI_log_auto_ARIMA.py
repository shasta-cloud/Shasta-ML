import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Auto ARIMA model
def auto_arima_model(data, steps=100):
    print("Starting Auto ARIMA model fitting...")
    model = auto_arima(data, seasonal=False, stepwise=True, suppress_warnings=True)
    print("Auto ARIMA model initialized.")
    model_fit = model.fit(data)
    print("Auto ARIMA model fitting completed.")
    print(model_fit.summary())
    predicted = model_fit.predict(n_periods=steps)
    print("Auto ARIMA model prediction completed.")
    return predicted

# Run simulation for a given method
def run_simulation(name, func, data, args, true_data, result_queue):
    try:
        predicted = func(data, steps=args[0])
        mse = np.mean((np.array(true_data) - np.array(predicted[-len(true_data):])) ** 2)
        result_queue.put((name, mse, predicted))
    except Exception as e:
        result_queue.put((name, float('inf'), []))
        print(f"Error in {name}: {e}")

def filter_and_predict(mac_address, df, time_window=4000, steps=10):  # Increased time_window to 4000
    # Filter data for the specific MAC address
    mac_data = df[df['Station MAC'] == mac_address]

    # Check for mobility
    mac_data['Signal Change'] = mac_data['Avg ACK Signal (dBm)'].diff().abs()
    mobility_threshold = 5
    mac_data['Is Moving'] = mac_data['Signal Change'] > mobility_threshold

    if not mac_data['Is Moving'].any():
        print(f"No mobility detected for MAC address {mac_address}.")
        return

    # Apply predictors and predict the next time steps
    rssi_data = mac_data['Avg ACK Signal (dBm)'].values
    timestamps = mac_data['Timestamp'].values

    # Ensure rssi_data is a list or array
    if not isinstance(rssi_data, (list, np.ndarray)):
        raise TypeError("rssi_data should be a list or numpy array")

    # Create a queue to collect results
    result_queue = mp.Queue()

    # Define fixed parameters for the Auto ARIMA method
    fixed_params = {
        "Auto ARIMA": [steps]
    }

    # Create and start a process for the Auto ARIMA method with fixed parameters
    methods = [
        ("Auto ARIMA", auto_arima_model, fixed_params["Auto ARIMA"])
    ]

    # Initialize predictions dictionary
    predictions = {name: [] for name, _, _ in methods}
    cumulative_mse = {name: 0 for name, _, _ in methods}

    # Iterate over multiple time windows
    num_windows = 20  # Number of time windows to evaluate
    for window_start in range(0, len(rssi_data) - time_window - steps, time_window // num_windows):
        window_end = window_start + time_window
        true_values = rssi_data[window_end:window_end + steps]
        true_timestamps = timestamps[window_end:window_end + steps]

        processes = []
        for name, func, args in methods:
            p = mp.Process(target=run_simulation, args=(name, func, rssi_data[window_start:window_end], args, true_values, result_queue))
            processes.append(p)
            p.start()

        # Collect results from all processes
        results = [result_queue.get() for _ in processes]

        # Ensure all processes have finished
        for p in processes:
            p.join()

        for name, mse, predicted in results:
            if len(predicted) > 0:
                predictions[name].extend(predicted[-steps:])
                cumulative_mse[name] += mse
            else:
                print(f"No predictions made for {name}")

        # Print progress
        print(f"Window {window_start // (time_window // num_windows) + 1}/{num_windows} completed")

        # Plot the actual vs predicted values for the current window
        plt.figure(figsize=(14, 8))
        plt.plot(true_timestamps, true_values, label='Actual', marker='o')
        plt.plot(true_timestamps, predicted, label='Predicted', marker='x')
        plt.title(f'Actual vs Predicted RSSI for MAC {mac_address} (Window {window_start // (time_window // num_windows) + 1})')
        plt.xlabel('Timestamp')
        plt.ylabel('Avg ACK Signal (dBm)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print the MSE for the current window
        for name in predictions:
            if len(predictions[name]) >= steps:
                mse = np.mean((np.array(true_values) - np.array(predictions[name][-steps:])) ** 2)
                print(f'{name} MSE for window {window_start // (time_window // num_windows) + 1}: {mse:.2f}')
            else:
                print(f"Not enough predictions for {name} to calculate MSE")

    # Calculate and print cumulative MSE
    for name in cumulative_mse:
        cumulative_mse[name] /= num_windows  # Average MSE over all windows
        print(f"{name} cumulative MSE: {cumulative_mse[name]:.2f}")

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('rssi_log_trej.csv')  # Replace with your actual file path
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Avg ACK Signal (dBm)'] = pd.to_numeric(df['Avg ACK Signal (dBm)'], errors='coerce')
    df.dropna(subset=['Avg ACK Signal (dBm)', 'Timestamp'], inplace=True)

    mac_addresses = df['Station MAC'].unique()  # Get all unique MAC addresses
    for mac_address in mac_addresses:
        print(f"Processing MAC address: {mac_address}")
        filter_and_predict(mac_address, df, time_window=4000, steps=10)  # Increased time_window to 4000
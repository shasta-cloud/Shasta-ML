import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter

# Simple Exponential Smoothing (SES)
def exponential_smoothing(data, alpha, steps=10):
    smoothed_data = [0] * len(data)
    smoothed_data[0] = data[0]
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    
    # Predict future values
    future_predictions = [smoothed_data[-1]] * steps
    return smoothed_data + future_predictions

# Kalman Filter
def kalman_filter(data, steps=10):
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data[0],
                      observation_covariance=1,
                      transition_covariance=0.01)
    smoothed_state_means, _ = kf.smooth(data)
    
    # Predict future values
    future_predictions = [smoothed_state_means[-1]] * steps
    return np.concatenate([smoothed_state_means, future_predictions])

# ARIMA model
def arima_model(data, steps=10):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    predicted = model_fit.predict(start=1, end=len(data) + steps, typ='levels')
    return predicted

# Run simulation for a given method
def run_simulation(name, func, data, args, result_queue):
    try:
        predicted = func(data, *args)
        mse = np.mean((np.array(data) - np.array(predicted[:len(data)])) ** 2)
        result_queue.put((name, mse, predicted))
    except Exception as e:
        result_queue.put((name, float('inf'), []))
        print(f"Error in {name}: {e}")

def filter_and_predict(mac_address, df, time_window=300, steps=10):
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

    # Ensure rssi_data is a list or array
    if not isinstance(rssi_data, (list, np.ndarray)):
        raise TypeError("rssi_data should be a list or numpy array")

    # Create a queue to collect results
    result_queue = mp.Queue()

    # Create and start a process for each method
    methods = [
        ("Exponential Smoothing", exponential_smoothing, [0.2]),
        ("Kalman Filter", kalman_filter, []),
        ("ARIMA", arima_model, [])
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    def animate(i):
        ax.clear()
        start = i * steps
        if start + time_window + steps > len(rssi_data):
            return

        data_used = rssi_data[start:start + time_window]
        true_data = rssi_data[start + time_window:start + time_window + steps]

        processes = []
        for name, func, args in methods:
            p = mp.Process(target=run_simulation, args=(name, func, data_used, args, result_queue))
            processes.append(p)
            p.start()

        # Collect results from all processes
        results = [result_queue.get() for _ in processes]

        # Ensure all processes have finished
        for p in processes:
            p.join()

        for name, mse, predicted in results:
            if len(predicted) >= time_window + steps:
                ax.plot(range(start, start + time_window + steps), predicted[:time_window + steps], label=f"{name} (MSE: {mse:.4f})")

        ax.plot(range(start, start + time_window), data_used, label="Data Used", color="blue")
        ax.plot(range(start + time_window, start + time_window + steps), true_data, label="True Data", color="green")
        
        # Highlight the prediction part with true values
        ax.axvspan(start + time_window, start + time_window + steps, color='yellow', alpha=0.2)

        ax.set_title(f"RSSI Prediction for MAC {mac_address} (Window Start: {start})")
        ax.set_xlabel("Time")
        ax.set_ylabel("RSSI")
        ax.legend()

    ani = animation.FuncAnimation(fig, animate, frames=range(0, len(rssi_data) // steps), repeat=False)
    ani.save(f'RSSI_prediction_{mac_address}.gif', writer=PillowWriter(fps=2))

if __name__ == "__main__":
    # Example usage
    df = pd.read_excel('RSSI_log_Efi_room2.ods', engine='odf')  # Replace with your actual file path
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Avg ACK Signal (dBm)'] = pd.to_numeric(df['Avg ACK Signal (dBm)'], errors='coerce')
    df.dropna(subset=['Avg ACK Signal (dBm)', 'Timestamp'], inplace=True)

    mac_addresses = df['Station MAC'].unique()  # Get all unique MAC addresses
    for mac_address in mac_addresses:
        filter_and_predict(mac_address, df, time_window=300, steps=10)
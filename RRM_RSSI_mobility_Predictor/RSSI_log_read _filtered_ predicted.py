import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# ARIMA model
def arima_model(train_data, order=(5, 1, 0)):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def predict_with_arima(train_data, test_data, order=(5, 1, 0), steps=10):
    predictions = []
    combined_data = np.concatenate([train_data, test_data])
    
    for i in range(0, len(test_data), steps):
        end_index = min(i + steps, len(test_data))
        window_data = combined_data[:len(train_data) + i]
        
        # Re-fit the model with the updated data
        model_fit = arima_model(window_data, order=order)
        
        predicted = model_fit.predict(start=len(window_data), end=len(window_data) + steps - 1, typ='levels')
        
        predictions.extend(predicted[:end_index - i])
    
    return np.array(predictions)

def main():
    # Read the CSV file
    df = pd.read_csv('rssi_log_long.csv')  # Replace with your actual file path
    # Filter for MAC address 'be:b3:a5:c8:06:85'
    df = df[df['Station MAC'] == 'be:b3:a5:c8:06:85']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Avg ACK Signal (dBm)'] = pd.to_numeric(df['Avg ACK Signal (dBm)'], errors='coerce')
    df.dropna(subset=['Avg ACK Signal (dBm)', 'Timestamp'], inplace=True)

    # Extract the RSSI data
    rssi_data = df['Avg ACK Signal (dBm)'].values

    # Split data into training (80%) and testing (20%) sets
    split_index = int(len(rssi_data) * 0.8)
    train_data = rssi_data[:split_index]
    test_data = rssi_data[split_index:]

    # Define the range of parameters for grid search
    p = range(20, 21, 1)
    d = range(0, 3)
    q = range(0, 2)
    pdq = list(itertools.product(p, d, q))

    best_order = None
    best_mse = float('inf')
    mse_results = []

    # Grid search for optimal order parameters with progress bar
    for idx, order in enumerate(tqdm(pdq, desc="Grid Search")):
        try:
            print(f"Evaluating ARIMA order {order} ({idx + 1}/{len(pdq)})")
            # Train ARIMA model on the training data
            model_fit = arima_model(train_data, order=order)

            # Predict on the test data
            predicted_test = predict_with_arima(train_data, test_data, order=order, steps=10)

            # Calculate MSE for all predictions
            mse = np.mean((test_data[:len(predicted_test)] - predicted_test) ** 2)
            mse_results.append((order, mse))

            # Print the resulting average MSE for the current order
            print(f'Order {order} - Average MSE: {mse:.2f}')

            # Update best order if current order has lower average MSE
            if mse < best_mse:
                best_mse = mse
                best_order = order

        except Exception as e:
            print(f"Error with order {order}: {e}")
            continue

    print(f'Best ARIMA order: {best_order} with average MSE: {best_mse:.2f}')

    # Plot MSE results for all orders
    orders, mses = zip(*mse_results)
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(mses)), mses, marker='o')
    plt.xlabel('Order Index')
    plt.ylabel('Average MSE')
    plt.title('Average MSE for Different ARIMA Orders')
    plt.xticks(range(len(orders)), labels=orders, rotation=90)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
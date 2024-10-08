import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import matplotlib.pyplot as plt
import cv2
import os

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

    # Fixed ARIMA order
    order = (20, 1, 1)

    print(f"Using ARIMA order {order}")
    # Train ARIMA model on the training data
    model_fit = arima_model(train_data, order=order)

    # Plot the training data and the model fitting
    plt.figure(figsize=(14, 7))
    plt.plot(train_data, label='Training Data')
    plt.plot(model_fit.fittedvalues, label='Fitted Values', color='red')
    plt.xlabel('Time')
    plt.ylabel('RSSI')
    plt.title('Training Data and Fitted Values')
    plt.legend()
    plt.savefig('training_data_fitted_values.png')
    plt.show()

    # Predict on the test data and create video frames
    steps = 10
    predictions = []
    combined_data = np.concatenate([train_data, test_data])
    frames = []

    for i in range(0, len(test_data), steps):
        end_index = min(i + steps, len(test_data))
        window_data = combined_data[:len(train_data) + i]
        
        # Re-fit the model with the updated data
        model_fit = arima_model(window_data, order=order)
        
        predicted = model_fit.predict(start=len(window_data), end=len(window_data) + steps - 1, typ='levels')
        
        predictions.extend(predicted[:end_index - i])
        
        # Plot and save the frame
        plt.figure(figsize=(14, 7))
        plt.plot(test_data, label='True Test Data')
        plt.plot(predictions, label='Predicted Test Data')
        plt.xlabel('Time')
        plt.ylabel('RSSI')
        plt.title(f'True vs Predicted RSSI for the Test Set (Step {i // steps + 1})')
        plt.legend()
        frame_filename = f'frame_{i // steps + 1}.png'
        plt.savefig(frame_filename)
        frames.append(frame_filename)
        plt.close()

    # Create video
    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter('test_set_predictions-5-2-0.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for frame_filename in frames:
        video.write(cv2.imread(frame_filename))
        os.remove(frame_filename)  # Remove the frame file after adding to video

    video.release()

    # Calculate MSE for all predictions
    predicted_test = np.array(predictions)
    mse = np.mean((test_data[:len(predicted_test)] - predicted_test) ** 2)
    print(f'Average MSE: {mse:.2f}')

    # Plot true and predicted values for the test set
    plt.figure(figsize=(14, 7))
    plt.plot(test_data, label='True Test Data')
    plt.plot(predicted_test, label='Predicted Test Data')
    plt.xlabel('Time')
    plt.ylabel('RSSI')
    plt.title(f'True vs Predicted RSSI for the Test Set (MSE: {mse:.2f})')
    plt.legend()
    plt.savefig('test_set_predictions.png')
    plt.show()

if __name__ == "__main__":
    main()
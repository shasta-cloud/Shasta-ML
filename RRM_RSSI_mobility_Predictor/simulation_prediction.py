import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter
import multiprocessing
from sklearn.metrics import mean_squared_error

# Simple Exponential Smoothing (SES)
def exponential_smoothing(data, alpha, steps=10):
    print(f"Running exponential_smoothing with alpha: {alpha}")
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
    print(f"Running ARIMA model")
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    predicted = model_fit.predict(start=1, end=len(data) + steps, typ='levels')
    return predicted

# LSTM model
def lstm_model(data, input_shape, output_shape, steps=10):
    print(f"Running LSTM model")
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare the data for LSTM
    X, y = [], []
    for i in range(len(data) - input_shape):
        X.append(data[i:i + input_shape])
        y.append(data[i + input_shape])
    X, y = np.array(X), np.array(y)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(50))
    model.add(Dense(output_shape))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Predict future values
    future_predictions = []
    last_sequence = data[-input_shape:]
    for _ in range(steps):
        next_pred = model.predict(last_sequence.reshape(1, input_shape, 1))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    return np.concatenate([predictions.flatten(), future_predictions])

# Particle Filter
def particle_filter(data, num_particles=1000, num_iterations=10, steps=10):
    print(f"Running Particle Filter with {num_particles} particles and {num_iterations} iterations")
    
    # Initialize particles
    particles = np.random.normal(data[0], 1, num_particles)
    weights = np.ones(num_particles) / num_particles

    # Particle filter algorithm
    for t in range(1, len(data)):
        # Predict
        particles += np.random.normal(0, 0.1, num_particles)
        
        # Update weights
        weights *= np.exp(-0.5 * (data[t] - particles)**2)
        weights += 1.e-300  # Avoid round-off to zero
        weights /= np.sum(weights)
        
        # Resample
        indices = np.random.choice(range(num_particles), num_particles, p=weights)
        particles = particles[indices]
        weights = weights[indices]
        weights /= np.sum(weights)
    
    # Estimate
    estimates = np.zeros(len(data))
    for t in range(len(data)):
        estimates[t] = np.mean(particles)
    
    # Predict future values
    future_predictions = [np.mean(particles)] * steps
    return np.concatenate([estimates, future_predictions])

# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    data = np.cumsum(np.random.randn(1000))  # Random walk data

    # Run each model
    alpha = 0.2
    smoothed_data = exponential_smoothing(data, alpha, steps=10)
    kalman_data = kalman_filter(data, steps=10)
    arima_steps = 10  # Number of future steps to predict
    arima_data = arima_model(data, steps=arima_steps)
    lstm_input_shape = 10
    lstm_output_shape = 1
    lstm_data = lstm_model(data, lstm_input_shape, lstm_output_shape, steps=10)
    particle_data = particle_filter(data, steps=10)

    # Plot the results
    plt.figure(figsize=(14, 10))
    plt.plot(data, label='Original Data')
    plt.plot(smoothed_data, label='Exponential Smoothing')
    plt.plot(kalman_data, label='Kalman Filter')
    plt.plot(arima_data, label='ARIMA Model')
    plt.plot(lstm_data, label='LSTM Model')
    plt.plot(particle_data, label='Particle Filter')
    plt.legend()
    plt.show()
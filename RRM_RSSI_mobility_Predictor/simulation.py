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
def exponential_smoothing(data, alpha):
    print(f"Running exponential_smoothing with alpha: {alpha}")
    smoothed_data = [0] * len(data)
    smoothed_data[0] = data[0]
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    return smoothed_data

# Kalman Filter
def kalman_filter(data):
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data[0],
                      observation_covariance=1,
                      transition_covariance=0.01)
    smoothed_state_means, _ = kf.smooth(data)
    return smoothed_state_means

# ARIMA model
def arima_model(data):
    print(f"Running ARIMA model")
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    predicted = model_fit.predict(start=1, end=len(data), typ='levels')
    return predicted

# LSTM model
def lstm_model(data, input_shape, output_shape):
    print(f"Running LSTM model")
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    # Prepare the data for LSTM
    X, y = [], []
    for i in range(len(data) - input_shape):
        X.append(data[i:i + input_shape, 0])
        y.append(data[i + input_shape, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the data into training and validation sets
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_data=(X_val, y_val))

    # Make predictions
    predicted = model.predict(X)
    predicted = scaler.inverse_transform(predicted)
    
    # Combine the initial data with the predictions
    combined = np.concatenate((data[:input_shape], predicted), axis=0)
    return combined.flatten()


# Particle Filter
def particle_filter(data):
    print(f"Running Particle Filter")
    num_particles = 1000
    particles = np.random.choice(data, size=num_particles)
    weights = np.ones(num_particles) / num_particles
    predicted = []

    for t in range(1, len(data)):
        particles += np.random.normal(0, 1, size=num_particles)
        weights *= np.exp(-0.5 * (data[t] - particles) ** 2)
        weight_sum = np.sum(weights)
        
        if weight_sum == 0 or np.any(np.isnan(weights)):
            weights = np.ones(num_particles) / num_particles
        else:
            weights /= weight_sum

        indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
        particles = particles[indices]
        predicted.append(np.mean(particles))

    predicted = np.concatenate(([data[0]], predicted))
    return predicted

def evaluate_method(method_name, method_func, data, args):
    print(f"Evaluating method: {method_name}")
    predicted = method_func(data, *args)
    mse = np.mean((np.array(data) - np.array(predicted)) ** 2)
    return method_name, mse, predicted

def run_simulation(method_name, method_func, data, args, result_queue):
    name, mse, predicted = evaluate_method(method_name, method_func, data, args)
    result_queue.put((name, mse, predicted))
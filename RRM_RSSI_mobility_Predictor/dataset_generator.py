import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def generate_rssi_data(num_points=3600, base_rssi=-60, trend_strength=0.01, noise_std=2, seasonal_period=500, seasonal_amplitude=5, random_seed=None):
    """
    Generates a synthetic RSSI data set.
    
    Parameters:
    - num_points: Number of RSSI data points to generate
    - base_rssi: The base RSSI value around which the values will fluctuate
    - trend_strength: The strength of the linear trend in RSSI values
    - noise_std: The standard deviation of the noise added to the RSSI
    - seasonal_period: Period of the seasonality (how often seasonal fluctuations repeat)
    - seasonal_amplitude: Amplitude of the seasonal fluctuations
    - random_seed: Seed for the random number generator (optional)
    
    Returns:
    - A NumPy array of generated RSSI values
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    time = np.arange(num_points)
    trend = trend_strength * time
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * time / seasonal_period)
    noise = np.random.normal(0, noise_std, num_points)
    
    rssi_values = base_rssi + trend + seasonal + noise
    return rssi_values


rssi_data = generate_rssi_data()
plt.plot(rssi_data)
plt.title("Simulated RSSI Data")
plt.xlabel("Time")
plt.ylabel("RSSI")
#plt.show()

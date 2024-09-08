import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def generate_rssi_data(num_points=500, base_rssi=-60, trend_strength=0.01, noise_std=2, seasonal_period=50, seasonal_amplitude=3):
    """
    Generates a synthetic RSSI data set.
    
    Parameters:
    - num_points: Number of RSSI data points to generate
    - base_rssi: The base RSSI value around which the values will fluctuate
    - trend_strength: The strength of the linear trend in RSSI values
    - noise_std: The standard deviation of the noise added to the RSSI
    - seasonal_period: Period of the seasonality (how often seasonal fluctuations repeat)
    - seasonal_amplitude: Amplitude of the seasonal fluctuations
    
    Returns:
    - A NumPy array of generated RSSI values
    """
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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import boxcox
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load the csv file (replace with your actual file path)
df = pd.read_csv('rssi_log_long.csv')

# Clean and preprocess the data
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.rename(columns=lambda x: x.strip(), inplace=True)

# Convert 'Avg ACK Signal (dBm)' to numeric, and drop rows with missing values
df['Avg ACK Signal (dBm)'] = pd.to_numeric(df['Avg ACK Signal (dBm)'], errors='coerce')
df_clean = df.dropna(subset=['Avg ACK Signal (dBm)', 'Timestamp'])

# Identify moving stations
df_clean['Signal Change'] = df_clean.groupby('Station MAC')['Avg ACK Signal (dBm)'].diff().abs()
mobility_threshold = 5
df_clean['Is Moving'] = df_clean['Signal Change'] > mobility_threshold

# Ensure all values are positive for Box-Cox transformation
min_value = df_clean['Avg ACK Signal (dBm)'].min()
if min_value <= 0:
    df_clean['Avg ACK Signal (dBm)'] += abs(min_value) + 1

# Apply Box-Cox transformation
df_clean['BoxCox Transformed'], lam = boxcox(df_clean['Avg ACK Signal (dBm)'])

# Differencing to make the data stationary
df_clean['Differenced'] = df_clean.groupby('Station MAC')['BoxCox Transformed'].diff()

# Calculate slopes
df_clean['Slope'] = df_clean.groupby('Station MAC')['Avg ACK Signal (dBm)'].diff()

# Plotting RSSI trajectory
plt.figure(figsize=(14, 8))
stations = df_clean['Station MAC'].unique()

for station in stations:
    station_data = df_clean[df_clean['Station MAC'] == station]
    
    # Plot line for RSSI trajectory
    plt.plot(station_data['Timestamp'], station_data['Avg ACK Signal (dBm)'], label=f'Station {station}', alpha=0.6)
    
    # Highlight moving points
    moving_data = station_data[station_data['Is Moving']]
    plt.scatter(moving_data['Timestamp'], moving_data['Avg ACK Signal (dBm)'], color='red', s=20, label=f'Moving {station}', alpha=0.8)
    
    # Highlight significant slopes
    slope_threshold = 2  # Define a threshold for significant slopes
    significant_slopes = station_data[station_data['Slope'].abs() > slope_threshold]
    plt.plot(significant_slopes['Timestamp'], significant_slopes['Avg ACK Signal (dBm)'], color='green', linestyle='None', marker='o', markersize=5, label=f'Significant Slope {station}')

# Configure plot details
plt.title('RSSI Trajectory of Stations Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Avg ACK Signal (dBm)')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-ticks

# Format the x-axis to show date and time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()

# Plotting flat parts of the data
plt.figure(figsize=(14, 8))

for station in stations:
    station_data = df_clean[df_clean['Station MAC'] == station]
    
    # Filter data to include only flat parts
    flat_parts = station_data[station_data['Slope'].abs() <= slope_threshold]
    
    # Plot line for flat parts
    plt.plot(flat_parts['Timestamp'], flat_parts['Avg ACK Signal (dBm)'], label=f'Station {station}', alpha=0.6)

# Configure plot details
plt.title('Flat Parts of RSSI Data Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Avg ACK Signal (dBm)')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-ticks

# Format the x-axis to show date and time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()

# Plotting slopes
plt.figure(figsize=(14, 8))

for station in stations:
    station_data = df_clean[df_clean['Station MAC'] == station]
    
    # Plot line for slopes
    plt.plot(station_data['Timestamp'], station_data['Slope'], label=f'Station {station}', alpha=0.6)

# Configure plot details
plt.title('Slopes of RSSI Data Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Slope of Avg ACK Signal (dBm)')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-ticks

# Format the x-axis to show date and time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()
'''
# Plot PACF for differenced data
for station in stations:
    station_data = df_clean[df_clean['Station MAC'] == station]
    plt.figure(figsize=(14, 8))
    plot_pacf(station_data['Differenced'].dropna(), ax=plt.gca(), title=f'PACF for Station {station}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plotting the ADF test results for the differenced data
for station in stations:
    station_data = df_clean[df_clean['Station MAC'] == station]
    adf_result = adfuller(station_data['Differenced'].dropna())
    
    # Plot ADF test statistic and critical values
    plt.figure(figsize=(14, 8))
    plt.plot(station_data['Timestamp'], station_data['Differenced'], label=f'Station {station}', alpha=0.6)
    plt.axhline(y=adf_result[4]['1%'], color='r', linestyle='--', label='1% Critical Value')
    plt.axhline(y=adf_result[4]['5%'], color='g', linestyle='--', label='5% Critical Value')
    plt.axhline(y=adf_result[4]['10%'], color='b', linestyle='--', label='10% Critical Value')
    plt.title(f'ADF Test for Station {station}')
    plt.xlabel('Timestamp')
    plt.ylabel('Differenced Box-Cox Transformed RSSI')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
'''
import pandas as pd
import matplotlib.pyplot as plt

# Load the ODS file (replace with your actual file path)
df = pd.read_excel('RSSI_log_Efi_room2.ods', engine='odf')

# Clean and preprocess the data
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.rename(columns=lambda x: x.strip(), inplace=True)

# Convert 'Avg ACK Signal (dBm)' to numeric, and drop rows with missing values
df['Avg ACK Signal (dBm)'] = pd.to_numeric(df['Avg ACK Signal (dBm)'], errors='coerce')
df_clean = df.dropna(subset=['Avg ACK Signal (dBm)'])

# Identify moving stations
df_clean['Signal Change'] = df_clean.groupby('Station MAC')['Avg ACK Signal (dBm)'].diff().abs()
mobility_threshold = 5
df_clean['Is Moving'] = df_clean['Signal Change'] > mobility_threshold

# Identify timestamps where RSSI data is missing
missing_rssi_data = df[df['Avg ACK Signal (dBm)'].isna()]

# Plotting
plt.figure(figsize=(14, 8))
stations = df_clean['Station MAC'].unique()

for station in stations:
    station_data = df_clean[df_clean['Station MAC'] == station]
    
    # Plot line for RSSI trajectory
    plt.plot(station_data['Timestamp'], station_data['Avg ACK Signal (dBm)'], label=f'Station {station}', alpha=0.6)
    
    # Highlight moving points
    moving_data = station_data[station_data['Is Moving']]
    plt.scatter(moving_data['Timestamp'], moving_data['Avg ACK Signal (dBm)'], color='red', s=20, label=f'Moving {station}', alpha=0.8)

# Adding indicators (e.g., black 'x' markers) for missing RSSI values
for idx, row in missing_rssi_data.iterrows():
    plt.axvline(row['Timestamp'], color='yellow', linestyle='--', alpha=0.3)  # Vertical line to indicate missing data period
    plt.scatter(row['Timestamp'], 0, color='black', marker='x', s=80, label='Missing RSSI' if idx == missing_rssi_data.index[0] else "")

# Configure plot details
plt.title('RSSI Trajectory of Stations Over Time (with Missing Data Indicated)')
plt.xlabel('Timestamp')
plt.ylabel('Avg ACK Signal (dBm)')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-ticks
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()

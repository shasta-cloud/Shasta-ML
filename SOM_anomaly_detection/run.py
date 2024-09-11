from minisom import MiniSom
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Ensure you are in the correct working directory
os.chdir('SOM_anomaly_detection')

# Load the data
file_path = 'ue_5_m.csv'
data = pd.read_csv(file_path)

def process_mac_address(mac_value):
    # Filter the data based on the 'mac' value
    filtered_data = data#[data['mac'] == mac_value]

    # Check if filtered data is empty
    if filtered_data.empty:
        raise ValueError(f"No data found for MAC value: {mac_value}")

    # Print the filtered data before dropping NaN values
    print(f"Filtered data for MAC {mac_value} before dropping NaN values:")
    print(filtered_data)

    # Print the count of NaN values in each column
    print("Count of NaN values in each column:")
    print(filtered_data.isna().sum())

    # Drop 'mac', 'port', and 'vlan' columns, and any columns with more than 50% NaN values
    columns_to_drop = ['mac', 'port', 'vlan']
    if 'datetime' in filtered_data.columns:
        columns_to_drop.append('datetime')

    # Drop specified columns and those with more than 50% NaNs
    filtered_data = filtered_data.drop(columns=columns_to_drop, errors='ignore')
    #filtered_data = filtered_data.loc[:, filtered_data.isna().mean() < 0.5]

    # Drop rows with NaN values
    filtered_data = filtered_data.dropna()

    # Check if filtered data is empty after dropping NaN values
    if filtered_data.empty:
        raise ValueError("No data available after dropping rows with NaN values.")

    # Update feature names after filtering and scaling
    feature_names = filtered_data.select_dtypes(include=[np.number]).columns

    # Initialize the scaler
    scaler = StandardScaler()

    # Normalize the filtered data
    data_array = scaler.fit_transform(filtered_data.select_dtypes(include=[np.number]))

    # Initialize and train the SOM
    som = MiniSom(x=10, y=10, input_len=data_array.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(data_array)
    som.train_random(data_array, num_iteration=1000)

    # BMU hit counts
    hits = np.zeros((10, 10))
    for d in data_array:
        winner = som.winner(d)
        hits[winner] += 1

    # Threshold to remove noise-dominated BMUs
    hit_threshold = np.percentile(hits, 90)
    valid_bmus = np.argwhere(hits >= hit_threshold)

    # Reference BMUs
    bmu_weights = np.array([som.get_weights()[bmu[0], bmu[1]] for bmu in valid_bmus])

    # Train KNN with valid BMUs
    k = 3  # Number of neighbors for KNN
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(bmu_weights)

    # Anomaly detection
    health_indicators = []
    anomalies = []
    for d in data_array:
        distances, _ = knn.kneighbors(d.reshape(1, -1))  # Reshape the input to fix the error
        avg_distance = np.mean(distances)
        health_indicators.append(avg_distance)

        if avg_distance > 1.0:  # Define your threshold based on healthy training data
            anomalies.append(d)

    # Convert anomalies to a DataFrame using updated feature names
    anomalies_df = pd.DataFrame(anomalies, columns=feature_names)

    # Calculate feature-wise statistics
    normal_data = data_array[np.isin(data_array, anomalies, invert=True).all(axis=1)]
    normal_df = pd.DataFrame(normal_data, columns=feature_names)

    normal_mean = normal_df.mean()
    normal_std = normal_df.std()
    anomalies_mean = anomalies_df.mean()
    anomalies_std = anomalies_df.std()

    # Print detected anomalies
    print("Detected anomalies:")
    print(anomalies_df)

    # Print feature-wise statistics
    print("\nFeature-wise statistics:")
    print("Normal Data Mean:\n", normal_mean)
    print("Normal Data Std Dev:\n", normal_std)
    print("Anomalies Mean:\n", anomalies_mean)
    print("Anomalies Std Dev:\n", anomalies_std)

    # Plot results using plotly.express
    som_positions = np.array([som.winner(x) for x in data_array])
    som_positions_df = pd.DataFrame(som_positions, columns=['SOM X-axis', 'SOM Y-axis'])
    som_positions_df['Health Indicator'] = health_indicators
    som_positions_df['Anomaly'] = ['Anomaly' if any(np.array_equal(x, anomaly) for anomaly in anomalies) else 'Normal' for x in data_array]

    fig = px.scatter(
        som_positions_df,
        x='SOM X-axis',
        y='SOM Y-axis',
        color='Anomaly',
        title=f'SOM Grid with Data Points and Anomalies for MAC {mac_value}',
        labels={'SOM X-axis': 'SOM X-axis', 'SOM Y-axis': 'SOM Y-axis'},
        marginal_x='histogram',
        marginal_y='rug'
    )

    fig.show()

    # Plot feature-wise mean comparison
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=feature_names,
        y=normal_mean,
        name='Normal Data Mean',
        marker_color='blue'
    ))
    fig2.add_trace(go.Bar(
        x=feature_names,
        y=anomalies_mean,
        name='Anomalies Mean',
        marker_color='red'
    ))

    fig2.update_layout(
        title=f'Feature-wise Mean Comparison for MAC {mac_value}',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Mean Value'),
        barmode='group'
    )

    fig2.show()

    return health_indicators, anomalies_df

# Process the first MAC address
mac_value_1 = '20:df:b9:3e:7b:e1'
print(f"Processing MAC address: {mac_value_1}")
health_indicators_1, anomalies_df_1 = process_mac_address(mac_value_1)

# Process the second MAC address
#mac_value_2 = '4c:b9:ea:07:f4:bc'
#print(f"Processing MAC address: {mac_value_2}")
#health_indicators_2, anomalies_df_2 = process_mac_address(mac_value_2)
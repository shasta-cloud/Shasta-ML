import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn import preprocessing
import tensorflow as tf
import tempfile
import keras
import os

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

data = pd.read_csv('dataset_903cb3bb245d_comb.csv')
data = data.drop(['mac'], axis=1)
data = data.drop(['uptime'], axis=1)
data = data.drop(['mem_buffered'], axis=1)
data = data.drop(['mem_cached'], axis=1)
data = data.drop(['mem_total'], axis=1)
data = data.drop(['sanity'], axis=1)
data = data.drop(['num_ifaces'], axis=1)
data['mem_free'] = data['mem_free'].astype(float)/973131776
data.info()
seq_length = 1

# Split your data into train and test sets
X_train, X_test, _, _ = train_test_split(data, data, test_size=0.2, random_state=42)
X_plot = X_test
'''
# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_train)
X_train = scaler.fit_transform(X_test)
X_test = scaler.transform(X_test)
'''
#print(X_test[0])
#pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
X_train = np.reshape(X_train, (int(len(X_train)/seq_length), seq_length, 12))
X_test = np.reshape(X_test, (int(len(X_test)/seq_length), seq_length, 12))
# Define the LSTM autoencoder model
input_shape = (X_train.shape[1], X_train.shape[2])

normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

model = keras.Sequential([
    normalizer,
    keras.layers.LSTM(12, activation='relu', return_sequences=True, input_shape=input_shape),
    keras.layers.LSTM(10, activation='relu', return_sequences=False),
    keras.layers.RepeatVector(X_train.shape[1]),
    keras.layers.LSTM(10, activation='relu', return_sequences=True),
    keras.layers.LSTM(12, activation='relu', return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(X_train.shape[2]))
])


model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
epochs = 1000
batch_size = 64

history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# Save the model
tf.saved_model.save(model, 'model_test')


# Plot the training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Detect anomalies using the reconstruction error
X_pred = model.predict(X_test)
mse = np.mean(np.power(X_test - X_pred, 2), axis=1)

# Define a threshold for anomaly detection (you can adjust this threshold)
threshold = np.mean(mse) + 5 * np.std(mse)

# Detect anomalies and raise an alarm
anomalies = mse > threshold
print("Number of anomalies:", np.sum(anomalies))

# Plot the histogram of MSE
plt.hist(mse, bins=50)
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.show()




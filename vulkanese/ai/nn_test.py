import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))

import vulkanese as ve

# Load your dataset
# For example, load it from a CSV file
data = pd.read_csv('home_prices.csv')

# Select features and target
features = data[['bedrooms', 'square_feet', 'location_index']]
target = data['price']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = ve.ai.Sequential([
    ve.ai.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    ve.ai.layers.Dense(64, activation='relu'),
    ve.ai.layers.Dense(1)  # Output layer: no activation because this is a regression problem
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',  # Common loss function for regression
    metrics=['mean_absolute_error']
)

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.1)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Set Mean Absolute Error: {mae}")

# Make predictions (for example, predict prices on the test set)
predictions = model.predict(X_test_scaled)
print(predictions[:5])  # Print the first 5 predictions

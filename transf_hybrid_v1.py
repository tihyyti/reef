import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Embedding
import tensorflow as tf

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise ModuleNotFoundError("TensorFlow is not installed. Please install it using: pip install tensorflow")

# Load and preprocess historical weather data (Example placeholder)
def load_historical_data():
    return np.random.rand(2000, 48, 5)  # Increased dataset size, 48-hour history, 5 features (temp, sunlight, wind, humidity, pressure)

# Load real-time sensor data (Example placeholder)
def load_realtime_data():
    return np.random.rand(2000, 5)  # Increased dataset size, 5 real-time features

# Load labels (Optimal tilt angles for training, Example placeholder)
def load_labels():
    return np.random.rand(2000, 1)  # 2000 samples, 1 output value (tilt angle)

# Positional Encoding Layer
def positional_encoding(seq_length, d_model):
    angles = np.arange(seq_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(angles, dtype=tf.float32)

# Define Transformer Encoder Layer
def transformer_encoder(inputs, head_size=128, num_heads=8, ff_dim=256, dropout=0.2):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

# Define Transformer Model for historical weather trends
historical_input = Input(shape=(48, 5))
position_encodings = positional_encoding(48, 5)
x = historical_input + position_encodings  # Adding positional encoding
x = transformer_encoder(x)
x = transformer_encoder(x)
x = transformer_encoder(x)  # Added extra transformer layer
x = GlobalAveragePooling1D()(x)
historical_output = Dense(64, activation='relu')(x)  # Increased layer size

# Define Feedforward Neural Network for real-time data
realtime_input = Input(shape=(5,))
realtime_layer = Dense(64, activation='relu')(realtime_input)
realtime_output = Dense(32, activation='relu')(realtime_layer)

# Merge both models
merged = tf.keras.layers.Concatenate()([historical_output, realtime_output])
final_layer = Dense(32, activation='relu')(merged)
output = Dense(1, activation='linear')(final_layer)  # Predict tilt angle

# Compile the pure Transformer model
model = Model(inputs=[historical_input, realtime_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Load Data
X_historical = load_historical_data()
X_realtime = load_realtime_data()
y = load_labels()

# Train Model
model.fit([X_historical, X_realtime], y, epochs=100, batch_size=64, validation_split=0.2)  # Increased epochs and batch size

# Save Model
model.save("transformer_solar_model.h5")

# Load Test Data (Placeholder Example)
X_test_historical = np.random.rand(400, 48, 5)  # 400 unseen test samples
X_test_realtime = np.random.rand(400, 5)
y_test = np.random.rand(400, 1)

# Load trained model and evaluate
model = load_model("transformer_solar_model.h5")
test_loss, test_mae = model.evaluate([X_test_historical, X_test_realtime], y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Predict values
y_pred = model.predict([X_test_historical, X_test_realtime])

# Compute RMSE and R2 Score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}, R2 Score: {r2}")

# Plot predicted vs actual tilt angles
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Tilt Angle")
plt.ylabel("Predicted Tilt Angle")
plt.title("Predicted vs Actual Tilt Angle")
plt.show()

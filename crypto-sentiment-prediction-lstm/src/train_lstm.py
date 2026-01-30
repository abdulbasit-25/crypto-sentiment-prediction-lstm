import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/data.csv")

data = df[['close']].values

# -----------------------------
# Data Scaling
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# -----------------------------
# Create Sequences
# -----------------------------
X, y = [], []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Build LSTM Model
# -----------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train, epochs=20, batch_size=32)

# -----------------------------
# Predictions
# -----------------------------
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)

real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label="Real Price", linewidth=2)
plt.plot(predicted, label="Predicted Price", linestyle="--", linewidth=2)
plt.title("Crypto Price Prediction (LSTM)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Next-Day Prediction
# -----------------------------
last_60 = scaled_data[-60:]
last_60 = last_60.reshape(1, 60, 1)

next_day_scaled = model.predict(last_60)
next_day_price = scaler.inverse_transform(next_day_scaled)

print("Predicted Next Day Price:", next_day_price[0][0])

# -----------------------------
# Save Model
# -----------------------------
model.save("models/crypto_lstm_model.h5")

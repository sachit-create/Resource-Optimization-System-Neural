import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# =========================
# LOAD DATA
# =========================
data_path = "../data/laptop_data.csv"
df = pd.read_csv(data_path)

print("Data loaded:", df.shape)

FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "net_upload_mbps",
    "net_download_mbps",
    "disk_read_mbps",
    "disk_write_mbps"
]

data = df[FEATURE_COLUMNS].values

# =========================
# SCALE DATA
# =========================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler (IMPORTANT for Flask)
joblib.dump(scaler, "scaler.save")

# =========================
# CREATE SEQUENCES
# =========================
SEQUENCE_LENGTH = 10

X = []
y = []

for i in range(len(data_scaled) - SEQUENCE_LENGTH):
    X.append(data_scaled[i:i + SEQUENCE_LENGTH])
    y.append(data_scaled[i + SEQUENCE_LENGTH][0])  # next CPU

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = Sequential([
    LSTM(64, input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================
# TRAIN
# =========================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# =========================
# EVALUATE
# =========================
loss, mae = model.evaluate(X_test, y_test)

print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# =========================
# SAVE MODEL
# =========================
model.save("model.h5")

print("✅ Model saved as model.h5")
print("✅ Scaler saved as scaler.save")
import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model("model/lstm_model.keras")
scaler = joblib.load("model/scaler.pkl")
config = joblib.load("model/config.pkl")

window_size = config["window_size"]

def forecast_future(data, days=30):

    data = np.array(data).reshape(-1, 1)
    scaled_data = scaler.transform(data)

    current_input = scaled_data[-window_size:].reshape(1, window_size, 1)

    predictions = []

    for _ in range(days):

        pred = model.predict(current_input, verbose=0)[0][0]  # ✔ FIX shape

        predictions.append(pred)

        # ✔ FIX: shift window safely (NO np.append)
        new_input = np.concatenate(
            (current_input[:, 1:, :], np.array(pred).reshape(1, 1, 1)),
            axis=1
        )

        current_input = new_input

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)
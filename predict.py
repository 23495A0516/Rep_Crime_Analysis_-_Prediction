import joblib
import numpy as np
import torch
from train_lstm import CrimeLSTM
from sklearn.preprocessing import StandardScaler

# Load Models
xgb_model = joblib.load("models/crime_prediction_xgb.joblib")
lstm_model = CrimeLSTM(input_size=4, hidden_size=64, num_layers=2, output_size=1)
lstm_model.load_state_dict(torch.load("models/crime_lstm.pth"))
lstm_model.eval()

# Load Encoder & Scaler
label_encoder_city = joblib.load("models/label_encoder_city.joblib")
scaler = StandardScaler()

# Predict Future Crimes
def predict_future_crime(city, day, month, year):
    city_encoded = label_encoder_city.transform([city])[0]

    future_years = list(range(year, year + 6))
    future_predictions = []

    prev_crime_count = xgb_model.predict(np.array([[city_encoded, day, month, year]]))[0]

    for future_year in future_years:
        X_future = np.array([[city_encoded, day, month, future_year]])
        X_future_scaled = scaler.fit_transform(X_future)
        X_future_tensor = torch.tensor(X_future_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            future_pred = lstm_model(X_future_tensor).item()
            prev_crime_count += future_pred * 0.1  # Slight increase in crime
            future_predictions.append(round(prev_crime_count))

    return future_years, future_predictions

# Test Prediction
future_years, future_predictions = predict_future_crime("New York", 15, 6, 2025)
print("Future Years:", future_years)
print("Crime Predictions:", future_predictions)

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
CSV_FILE_PATH = './crime_dataset.csv'

if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"❌ Error: CSV file '{CSV_FILE_PATH}' not found.")

df = pd.read_csv(CSV_FILE_PATH)

# Select relevant features
features = ["City", "Year", "Month", "Day"]
target = "Crime_Rate"

if target not in df.columns:
    raise ValueError("❌ Error: 'crime_rate' column is missing in the dataset.")

# Encode categorical data
label_encoders = {}
for col in ["City"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, "models/label_encoders.pkl")

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target].values

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Define the model
class CrimePredictionNN(nn.Module):
    def __init__(self, input_size):
        super(CrimePredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
input_size = X.shape[1]
crime_prediction_model = CrimePredictionNN(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(crime_prediction_model.parameters(), lr=0.01)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Train the model
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = crime_prediction_model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(crime_prediction_model.state_dict(), "models/crime_prediction_model.pth")
print("✅ Model trained and saved successfully at 'models/crime_prediction_model.pth'")

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample training data (Replace this with actual training features)
X_train = np.array([[1, 2024, 2], [2, 2023, 4], [3, 2025, 6]])

# Train the scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

print("Scaler saved successfully!")

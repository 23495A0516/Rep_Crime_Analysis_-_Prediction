import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Sample training data (Replace with actual dataset)
X_train = np.array([[1, 2024, 2], [2, 2023, 4], [3, 2025, 6]])
y_train = np.array([0, 1, 2])

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/crime_analysis_model.pkl")
print("Model saved successfully!")

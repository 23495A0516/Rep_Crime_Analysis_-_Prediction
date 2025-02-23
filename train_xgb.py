import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("crime_dataset.csv")

# Encode categorical variables
label_encoder_city = LabelEncoder()
df["City"] = label_encoder_city.fit_transform(df["City"])

# Save the encoder
joblib.dump(label_encoder_city, "models/label_encoder_city.joblib")

# Define features and labels
X = df[["City", "Day", "Month", "Year"]]
y = df["Crime Count"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/crime_prediction_xgb.joblib")
print("✅ XGBoost Model Saved Successfully!")

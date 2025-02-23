import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset (Replace with actual dataset path)
df = pd.read_csv("crime_dataset.csv")  

# Create label encoders for categorical columns
label_encoders = {}
categorical_columns = ["City", "Crime_Type"]  # Update column names as per dataset

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])  # Transform the column
    label_encoders[col] = encoder  # Store the encoder

# Save label encoders to a file
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Label encoders saved successfully!")

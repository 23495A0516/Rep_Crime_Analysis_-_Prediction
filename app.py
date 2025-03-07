from flask import Flask, render_template, request, jsonify, redirect, send_from_directory, url_for
from pymongo import MongoClient
from predict import predict_future_crime
from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib
# import qrcode
matplotlib.use("Agg")  # Fix for Matplotlib GUI issues
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from geopy.geocoders import Nominatim
import io
import base64
import time

app = Flask(__name__)

# Load CSV data into a DataFrame
CSV_FILE_PATH = './crime_dataset.csv'
FEEDBACK_FILE_PATH = './submit_feedback.txt'
CSV_FILE_PATH1 = pd.read_csv(CSV_FILE_PATH)

# Load dataset at the start of the application
df = pd.read_csv(CSV_FILE_PATH)

# Load the ML model for crime analysis
crime_analysis_model = joblib.load("models/crime_analysis_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Load the DL model for crime prediction
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

input_size = 4
crime_prediction_model = CrimePredictionNN(input_size)
crime_prediction_model.load_state_dict(torch.load("models/crime_prediction_model.pth"))
crime_prediction_model.eval()

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['crime_analysis']
crime_data_collection = db['crime_data']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_crime_data():
    try:
        crime_type = request.form['crimeType']
        incident_date = request.form['incidentDate']
        incident_time = request.form['incidentTime']
        location = request.form['location']
        
        crime_data = {
            "crime_type": crime_type,
            "incident_date": datetime.strptime(incident_date, '%Y-%m-%d'),
            "incident_time": incident_time,
            "location": location
        }
        
        crime_data_collection.insert_one(crime_data)
        return render_template('thank_you_Data.html', name="Reporter 🕵🏻️")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

city_coordinates = {
    "Seattle": (47.6062, -122.3321),
    "Houston": (29.7604, -95.3698),
    "Boston": (42.3601, -71.0589),
    "Miami": (25.7617, -80.1918),
    "Dallas": (32.7767, -96.7970),
    "Chicago": (41.8781, -87.6298),
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "New York": (40.7128, -74.0060),
    "Denver": (39.7392, -104.9903)
}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form.to_dict()
    City = data.get("City").strip().title()  # Ensure proper case
    Crime_Type = data.get("Crime_Type").strip().title()

    # Normalize column names to match dataset
    filtered_data = CSV_FILE_PATH1[
        (CSV_FILE_PATH1['City'].str.title() == City) & 
        (CSV_FILE_PATH1['Crime_Type'].str.title() == Crime_Type)
    ]

    if not filtered_data.empty:
        Crime_Rate = filtered_data.iloc[0]['Crime_Rate']
        Crime_Count = filtered_data.iloc[0]['Crime Count']
    else:
        return render_template('error.html', message="No data found for the selected city and crime type.")

    # Get coordinates for selected city
    lat, lon = city_coordinates.get(City, (0, 0))

    return render_template('analysis_results.html', 
                           City=City, 
                           Crime_Type=Crime_Type, 
                           Crime_Rate=Crime_Rate, 
                           Crime_Count=Crime_Count, 
                           lat=lat, lon=lon)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    City = data.get("City")
    date = data.get("date")

    # Validate input
    if not City:
        return render_template("error.html", message="❌ Error: City is required.")
    if City not in label_encoders["City"].classes_:
        return render_template("error.html", message=f"❌ Error: City '{City}' is not recognized.")
    
    City_encoded = label_encoders["City"].transform([City])[0]
    
    try:
        Year, Month, Day = map(int, date.split("-"))
    except ValueError:
        return render_template("error.html", message="❌ Error: Invalid date format. Use YYYY-MM-DD.")
    
    # Prepare data for prediction
    X = np.array([[City_encoded, Day, Month, Year]])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict crime count
    with torch.no_grad():
        crime_prediction = crime_prediction_model(X_tensor).item()
    
    # Fetch past crime data
    city_data = CSV_FILE_PATH1[CSV_FILE_PATH1["City"] == City]
    if city_data.empty:
        return render_template("error.html", message=f"❌ No crime data found for {City}.")
    
    city_data = city_data.groupby("Year")["Crime Count"].sum().reset_index()
    
    # Create graphs directory
    graph_dir = "static/graphs"
    os.makedirs(graph_dir, exist_ok=True)
    timestamp = int(time.time())
    
    # Generate Past Crime Graph
    past_graph_path = f"{graph_dir}/{City}_past_{timestamp}.png"
    plt.figure(figsize=(8, 4))
    plt.plot(city_data["Year"], city_data["Crime Count"], marker='o', linestyle='-', color="red", label="Past Crime")
    plt.xlabel("Year")
    plt.ylabel("Crime Count")
    plt.title(f"📊 Past Crime Growth in {City}")
    plt.legend()
    plt.grid()
    plt.savefig(past_graph_path)
    plt.close()
    
    # Generate Future Crime Prediction
    future_years = list(range(Year, Year + 6))
    future_predictions = []
    
    for future_year in future_years:
        X_future = np.array([[City_encoded, Day, Month, future_year]])
        X_future_scaled = scaler.transform(X_future)
        X_future_tensor = torch.tensor(X_future_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            future_pred = crime_prediction_model(X_future_tensor).item()
            future_predictions.append(round(future_pred))
    
    future_graph_path = f"{graph_dir}/{City}_future_{timestamp}.png"
    plt.figure(figsize=(8, 4))
    plt.plot(future_years, future_predictions, marker='o', linestyle='--', color="blue", label="Predicted Future Crime")
    plt.xlabel("Year")
    plt.ylabel("Crime Count")
    plt.title(f"📊 Future Crime Prediction in {City}")
    plt.legend()
    plt.grid()
    plt.savefig(future_graph_path)
    plt.close()
    
    # Get city coordinates for map
    lat, lon = city_coordinates.get(City, (None, None))
    
    return render_template(
        "prediction_result.html", 
        City=City, 
        date=date, 
        predicted_crime=max(0, round(crime_prediction)),
        past_crime_graph=past_graph_path, 
        future_crime_graph=future_graph_path,
        lat=lat, lon=lon,
        future_years=future_years,
        future_crimes=future_predictions
    )

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        name = request.form['name']
        mail_id = request.form['email']
        feedback = request.form['feedback']

        with open(FEEDBACK_FILE_PATH, 'a') as f:
            f.write(f"Name: {name}\nEmail: {mail_id}\nFeedback: {feedback}\n\n")

        return render_template('thank_you_FB.html', name=name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(os.path.join(app.root_path, 'static'), path)

if __name__ == '__main__':
    app.run(debug=True)

import pickle
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS



# Initialize Flask app
app = Flask(__name__)

CORS(app)

# Load the trained model
with open("glof_xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the saved scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# OpenWeatherMap API Key (Replace with your actual API key)
OPENWEATHER_API_KEY = "4b4ef5677d44496ca3d61cd3cb9d3241"

# Function to get elevation from Open-Elevation API
def get_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses
        return response.json()["results"][0]["elevation"]
    except Exception as e:
        print(f"Elevation API Error: {e}")
        return None

# Function to get weather data from OpenWeatherMap
def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "precipitation": data.get("rain", {}).get("1h", 0),  # Rainfall in last hour
        }
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None

def get_weather_tomorrow(lat, lon):
    api_key = "YOUR_TOMORROW_IO_API_KEY"
    url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={"qBleeXvBQt2wCWBzzEvAwtLQQlEB7z3w"}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()["data"]["values"]
        return {
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "precipitation": data.get("precipitationIntensity", 0),
        }
    except Exception as e:
        print(f"Tomorrow.io API Error: {e}")
        return None


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract location from request
        lat, lon = data.get("Lat_lake"), data.get("Lon_lake")

        if lat is None or lon is None:
            return jsonify({"error": "Latitude and Longitude are required"}), 400

        # Fetch additional features from APIs
        elev = get_elevation(lat, lon)
        weather = get_weather_tomorrow(lat, lon)

        if elev is None or weather is None:
            return jsonify({"error": "Failed to fetch external data"}), 500

        # Prepare input DataFrame
        df = pd.DataFrame([{
            "Lat_lake": lat,
            "Lon_lake": lon,
            "Elev_lake": elev,
            "Temperature": weather["temperature"],
            "Humidity": weather["humidity"],
            "Precipitation": weather["precipitation"]
        }])

        # Standardize input features
        X_input = scaler.transform(df)

        # Predict GLOF impact level
        prediction = model.predict(X_input)[0]
        impact_classes = {0: "Low", 1: "Medium", 2: "High"}
        result = impact_classes.get(prediction, "Unknown")

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

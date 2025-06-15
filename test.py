import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "Lat_lake": 27.98,
    "Lon_lake": 86.92,
    "Elev_lake": 4800,
    "Driver_lake": "Surging glacier",
    "Driver_GLOF": "Moraine Dam Failure",
    "Mechanism": "Overtopping",
    "Impact_type": "Infrastructure Damage",
    "Repeat": "Yes",
    "Region_RGI": "Himalayas"
}

response = requests.post(url, json=data)
print(response.json())

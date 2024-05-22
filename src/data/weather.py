import requests
import csv
import os

# API URL
url = 'https://api.open-meteo.com/v1/forecast?latitude=46.5547&longitude=15.6467&current=temperature_2m,rain,weather_code&timezone=Europe%2FBerlin&forecast_days=1'

data_dir = 'data/raw'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    desired_data = {
        "time": data["current"]["time"],
        "temperature_2m": data["current"]["temperature_2m"],
        "rain": data["current"]["rain"],
        "weather_code": data["current"]["weather_code"]
    }

    csv_filename = "weather_data.csv"
    csv_filepath = os.path.join(data_dir, csv_filename)

    with open(csv_filepath, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=desired_data.keys())
        writer.writeheader()
        writer.writerow(desired_data)

    print(f"Data has been saved to {csv_filepath}")
else:
    print(f"Failed to fetch data from {url}. Status code: {response.status_code}")

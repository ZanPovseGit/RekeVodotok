import csv
from datetime import datetime
import os

reka_data = []
with open('data/raw/reka_data.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row['Datum'] = datetime.strptime(row['Datum'], '%Y-%m-%d %H:%M')
        reka_data.append(row)

weather_data = []
with open('data/raw/weather_data.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row['time'] = datetime.strptime(row['time'], '%Y-%m-%dT%H:%M')
        weather_data.append(row)

matched_data = []
for reka_row in reka_data:
    for weather_item in weather_data:
        if reka_row['Datum'].strftime('%Y-%m-%d %H') == weather_item['time'].strftime('%Y-%m-%d %H'):
            matched_row = reka_row.copy()
            matched_row['Temperature 2m'] = weather_item['temperature_2m']
            matched_row['Rain'] = weather_item['rain']
            matched_row['Weather Code'] = weather_item['weather_code']
            matched_data.append(matched_row)
            break

headers = [
    'Postaja Sifra', 'Ge Dolzina', 'Ge Sirina', 'Kota 0', 'Reka', 'Merilno Mesto', 'Ime Kratko', 'Datum', 'Datum CET', 
    'Vodostaj', 'Pretok', 'Pretok Znacilni', 'Temp Vode', 'Prvi VV Pretok', 'Drugi VV Pretok', 'Tretji VV Pretok', 
    'Temperature 2m', 'Rain', 'Weather Code'
]

output_csv_file = 'data/processed/learning_data.csv'
file_exists = os.path.exists(output_csv_file) and os.path.getsize(output_csv_file) > 0

with open(output_csv_file, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    if not file_exists:
        writer.writeheader()
    
    if file_exists:
        for row in matched_data:
            row['Datum'] = row['Datum'].strftime('%Y-%m-%d %H:%M')
            writer.writerow(row)
        print(f"Data has been appended to {output_csv_file}")
    else:
        print("No existing data found in the output file. No new data appended.")


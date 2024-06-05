from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import folium
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import requests
import joblib

def encode_pretok_znacilni(column):
    mapping = {'mali pretok': 0, 'srednji pretok': 1, 'velik pretok': 3, '': 4}
    return column.replace(mapping)


model_znacilni = tf.keras.models.load_model('src/models/model.h5')
model_pretok = tf.keras.models.load_model('src/models/modelRob.h5')

preprocessor = joblib.load('models/preprocessor_pipeline.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map():
    river_drava_coords = (46.5547, 15.6459)
    m = folium.Map(location=river_drava_coords, zoom_start=12)
    folium.Marker(river_drava_coords, popup='Loading...', tooltip='Click for Prediction', icon=folium.Icon(color='blue')).add_to(m)
    return m._repr_html_()

@app.route('/predict', methods=['GET'])
def predict():
    api_url = 'https://api.open-meteo.com/v1/forecast?latitude=46.5547&longitude=15.6467&current_weather=true&timezone=Europe%2FBerlin'
    response = requests.get(api_url)
    api_data = response.json()

    temperature = api_data['current_weather']['temperature']
    rain = api_data['current_weather'].get('rain', 0)
    weather_code = api_data['current_weather']['weathercode']

    input_data = pd.DataFrame({
        "Temperature 2m": [temperature],
        "Rain": [rain],
        "Weather Code": [weather_code],
        "Merilno Mesto": ["Drava"],
        "Pretok Znacilni": ["srednji pretok"]
    })

    input_data['Pretok Znacilni'] = encode_pretok_znacilni(input_data['Pretok Znacilni'])
    input_data['Pretok Znacilni'] = pd.to_numeric(input_data['Pretok Znacilni'], errors='coerce')
    X_processed = preprocessor.transform(input_data)


    X_processed = np.expand_dims(X_processed, axis=0)

    pred_znacilni = model_znacilni.predict(X_processed)
    pred_pretok = model_pretok.predict(X_processed)

    pred_znacilni = float(pred_znacilni[0][0])
    pred_pretok = float(pred_pretok[0][0])

    if pred_znacilni <= 0.5:
        flow_description = "Mali pretok"
    elif 0.5 < pred_pretok <= 1.5:
        flow_description = "Srednji pretok"
    else:
        flow_description = "Velik pretok"

    return jsonify({
        'pretok': pred_pretok,
        'znacilni': flow_description
    })

if __name__ == '__main__':
    app.run(debug=True)

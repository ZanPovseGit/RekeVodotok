import joblib
from flask import Flask, request, jsonify
from keras.models import load_model
import mlflow.keras
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "761209f44c347c69076d1becee6ca6c1d9257e4f"

mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/RekeVodotok.mlflow")

def encode_pretok_znacilni(column):
    mapping = {'mali pretok': 0, 'srednji pretok': 1, 'velik pretok': 3, '': 4}
    return column.replace(mapping)

preprocessor = joblib.load('models/preprocessor_pipeline.pkl')

def load_model_pret():
    try:
        previous_production_run = mlflow.search_runs(filter_string="tags.environment = 'production'", order_by=["start_time DESC"]).iloc[0]
        previous_production_run_id = previous_production_run["run_id"]
        previous_model_path = f"runs:/{previous_production_run_id}/lstmPretok"
        model_url = mlflow.artifacts.download_artifacts(run_id=previous_production_run_id, artifact_path="lstmPretok", dst_path="src/models/")
        #onnx_model_path = f"src/models/{previous_production_run_id}/lstmPretok/onnx/model.onnx"
        model_path = "src/models/model.h5"
        model = load_model(model_path)
        return model
    except IndexError:
        print("No previous model found.")
    return None

model =load_model_pret()

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.json
    df = pd.DataFrame([data])

    df['Pretok Znacilni'] = encode_pretok_znacilni(df['Pretok Znacilni'])
    df['Pretok Znacilni'] = pd.to_numeric(df['Pretok Znacilni'], errors='coerce')

    X_processed = preprocessor.transform(df)

    X_processed = np.expand_dims(X_processed, axis=0)

    prediction = model.predict(X_processed)


    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

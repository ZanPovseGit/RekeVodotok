import mlflow.keras
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import os


os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "761209f44c347c69076d1becee6ca6c1d9257e4f"

mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/RekeVodotok.mlflow")

with mlflow.start_run(run_name="GrajenjeModela"):

    train_data = pd.read_csv('data/processed/learning_data.csv')
    eval_data = pd.read_csv('data/processed/evaluation_data.csv')

    print(train_data.head())
    print(eval_data.head())

    # Custom encoding function
    def encode_pretok_znacilni(column):
        mapping = {'mali pretok': 0, 'srednji pretok': 1, 'velik pretok': 3,'':4}
        return column.replace(mapping)

    # Apply the custom encoding function before converting to numeric
    train_data['Pretok Znacilni'] = encode_pretok_znacilni(train_data['Pretok Znacilni'])
    eval_data['Pretok Znacilni'] = encode_pretok_znacilni(eval_data['Pretok Znacilni'])

    print(eval_data['Pretok Znacilni'])

    # Check and convert target columns to numeric
    train_data['Pretok'] = pd.to_numeric(train_data['Pretok'], errors='coerce')
    train_data['Pretok Znacilni'] = pd.to_numeric(train_data['Pretok Znacilni'], errors='coerce')
    eval_data['Pretok'] = pd.to_numeric(eval_data['Pretok'], errors='coerce')
    eval_data['Pretok Znacilni'] = pd.to_numeric(eval_data['Pretok Znacilni'], errors='coerce')

    # Define feature columns
    numerical_features = ['Temperature 2m', 'Rain']
    categorical_features = ['Weather Code', 'Merilno Mesto']
    pretok_znacilni_feature = ['Pretok Znacilni']

    # Define preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pretok_znacilni_pipeline = Pipeline([
        ('encode', FunctionTransformer(encode_pretok_znacilni, validate=False)),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features),
        ('pretok_znacilni', pretok_znacilni_pipeline, pretok_znacilni_feature)
    ])

    # Split features and target
    X_train = train_data[numerical_features + categorical_features + pretok_znacilni_feature]
    y_train_pretok = train_data['Pretok']
    y_train_znacilni = train_data['Pretok Znacilni']

    X_eval = eval_data[numerical_features + categorical_features + pretok_znacilni_feature]
    y_eval_pretok = eval_data['Pretok']
    y_eval_znacilni = eval_data['Pretok Znacilni']

    # Preprocess features
    X_train_processed = preprocessor.fit_transform(X_train)
    X_eval_processed = preprocessor.transform(X_eval)

    # Check lengths of processed evaluation data
    print(f'Length of X_eval_processed: {len(X_eval_processed)}')
    print(f'Length of y_eval_pretok: {len(y_eval_pretok)}')

    # Ensure sequence length is appropriate
    sequence_length = 1
    batch_size = 32

    # Create TimeseriesGenerator for training and evaluation
    train_gen_pretok = TimeseriesGenerator(X_train_processed, y_train_pretok, length=sequence_length, batch_size=batch_size)
    eval_gen_pretok = TimeseriesGenerator(X_eval_processed, y_eval_pretok, length=sequence_length, batch_size=batch_size)

    train_gen_znacilni = TimeseriesGenerator(X_train_processed, y_train_znacilni, length=sequence_length, batch_size=batch_size)
    eval_gen_znacilni = TimeseriesGenerator(X_eval_processed, y_eval_znacilni, length=sequence_length, batch_size=batch_size)

    # PRETOK Model
    model_pretok = Sequential()
    model_pretok.add(LSTM(50, activation='relu', input_shape=(sequence_length, X_train_processed.shape[1])))
    model_pretok.add(Dense(1))
    model_pretok.compile(optimizer='adam', loss='mse')

    model_pretok.fit(train_gen_pretok, epochs=50, validation_data=eval_gen_pretok)

    # ZNACILNOST PRETOKA Model
    model_znacilni = Sequential()
    model_znacilni.add(LSTM(50, activation='relu', input_shape=(sequence_length, X_train_processed.shape[1])))
    model_znacilni.add(Dense(1))
    model_znacilni.compile(optimizer='adam', loss='mse')

    model_znacilni.fit(train_gen_znacilni, epochs=50, validation_data=eval_gen_znacilni)

    loss_pretok = model_pretok.evaluate(eval_gen_pretok)
    loss_znacilni = model_znacilni.evaluate(eval_gen_znacilni)

    print(f'Pretok model loss: {loss_pretok}')
    print(f'Pretok Znacilni model loss: {loss_znacilni}')

    mlflow.log_metric("LossPretoka",loss_pretok)
    mlflow.log_metric("Loss znacilnosti pretoka",loss_znacilni)
    mlflow.log_param("Sekvenca dolzina",sequence_length)
    mlflow.log_param("Batch size",batch_size)


    try:
        previous_production_run = mlflow.search_runs(filter_string="tags.environment = 'production'",order_by=["start_time DESC"]).iloc[0]
        print(previous_production_run.to_string())
        previous_production_run_id = previous_production_run["run_id"]
        previous_model_path = f"runs:/{previous_production_run_id}/lstm_model"
        prev_loss = previous_production_run["metrics.LossPretoka"]
        print(f"Previous model accuracy: {prev_loss}")
        
        if loss_pretok >= prev_loss:
            mlflow.log_param("environment", "production")
            mlflow.keras.log_model(model_pretok,"lstmPretok")
            mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/lstmPretok", "GradenjeModela")
            print("New model saved.")
        else:
            print("New model is not better than the previous one. Keeping the old model.")
    except IndexError:
        print("No previous model found.")
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.preprocessing.sequence import TimeseriesGenerator
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import base64
import io

def encode_pretok_znacilni(column):
    mapping = {'mali pretok': 0, 'srednji pretok': 1, 'velik pretok': 3, '': 4}
    return column.replace(mapping)

model_znacilni = load_model('src/models/model.h5')

train_data = pd.read_csv('data/processed/learning_data.csv')
eval_data = pd.read_csv('data/processed/evaluation_data.csv')

train_data['Pretok Znacilni'] = encode_pretok_znacilni(train_data['Pretok Znacilni'])
eval_data['Pretok Znacilni'] = encode_pretok_znacilni(eval_data['Pretok Znacilni'])

train_data['Pretok Znacilni'] = pd.to_numeric(train_data['Pretok Znacilni'], errors='coerce').fillna(4)
eval_data['Pretok Znacilni'] = pd.to_numeric(eval_data['Pretok Znacilni'], errors='coerce').fillna(4)

numerical_features = ['Temperature 2m', 'Rain']
categorical_features = ['Weather Code', 'Merilno Mesto']
pretok_znacilni_feature = ['Pretok Znacilni']

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

X_train = train_data[numerical_features + categorical_features + pretok_znacilni_feature]
y_train_znacilni = train_data['Pretok Znacilni']

preprocessor.fit(X_train)

X_eval = eval_data[numerical_features + categorical_features + pretok_znacilni_feature]
y_eval_znacilni = eval_data['Pretok Znacilni']

X_eval_processed = preprocessor.transform(X_eval)

categorical_feature_names = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
processed_columns = numerical_features + list(categorical_feature_names) + pretok_znacilni_feature

sequence_length = 1
batch_size = 32
eval_gen_znacilni = TimeseriesGenerator(X_eval_processed, y_eval_znacilni, length=sequence_length, batch_size=batch_size)

predictions = model_znacilni.predict(eval_gen_znacilni).flatten()
y_eval_znacilni_seq = y_eval_znacilni[sequence_length:].values 

errors = predictions - y_eval_znacilni_seq

mse = mean_squared_error(y_eval_znacilni_seq, predictions)
mae = mean_absolute_error(y_eval_znacilni_seq, predictions)

scatter_data = pd.DataFrame(X_eval_processed, columns=processed_columns)
scatter_data = scatter_data.iloc[sequence_length:] 
scatter_data['Predicted'] = predictions
scatter_data['Actual'] = y_eval_znacilni_seq

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

X_sample = X_eval_processed[:100]
X_sample_reshaped = X_sample.reshape((X_sample.shape[0], 1, X_sample.shape[1]))

def model_predict(data):
    data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
    return model_znacilni.predict(data_reshaped).flatten()

explainer = shap.KernelExplainer(model_predict, X_sample)
shap_values = explainer.shap_values(X_sample, nsamples=100)

shap.summary_plot(shap_values, X_sample, feature_names=processed_columns, show=False)
summary_fig = plt.gcf()
summary_fig_base64 = plot_to_base64(summary_fig)

shap.initjs()
force_plot_html = shap.force_plot(explainer.expected_value, shap_values[0], X_sample[0], feature_names=processed_columns).html()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Pretok Znacilni Model Evaluation Dashboard'),
    html.Div(f'Mean Squared Error: {mse:.4f}', id='mse-output'),
    html.Div(f'Mean Absolute Error: {mae:.4f}', id='mae-output'),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in scatter_data.columns],
        value=processed_columns[0],
        clearable=False
    ),
    dcc.Graph(id='line-chart'),
    dcc.Graph(id='scatter-matrix'),
    dcc.Graph(id='error-violin'),
    dcc.Graph(id='residual-plot'),
    html.Div([
        html.H3('SHAP Summary Plot'),
        html.Img(src=f'data:image/png;base64,{summary_fig_base64}')
    ]),
    html.Div([
        html.H3('SHAP Force Plot'),
        html.Iframe(srcDoc=force_plot_html, style={"width": "100%", "height": "500px"})
    ]),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  
        n_intervals=0
    )
])

@app.callback(
    [Output('line-chart', 'figure'),
     Output('scatter-matrix', 'figure'),
     Output('error-violin', 'figure'),
     Output('residual-plot', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('feature-dropdown', 'value')]
)
def update_graphs(n, selected_feature):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(len(y_eval_znacilni_seq))), y=y_eval_znacilni_seq, mode='lines', name='Actual'))
    fig1.add_trace(go.Scatter(x=list(range(len(predictions))), y=predictions, mode='lines', name='Predicted'))
    fig1.update_layout(
        title='Actual vs Predicted Pretok Znacilni',
        xaxis_title='Time',
        yaxis_title='Pretok Znacilni',
        xaxis=dict(rangeslider=dict(visible=True))
    )

    fig2 = px.scatter_matrix(scatter_data, dimensions=scatter_data.columns[:-2], color='Predicted')
    fig2.update_layout(title='Scatter Matrix Plot')

    fig3 = go.Figure()
    fig3.add_trace(go.Violin(y=errors, box_visible=True, line_color='blue'))
    fig3.update_layout(title='Error Distribution (Violin Plot)', yaxis_title='Error')

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=scatter_data[selected_feature], y=errors, mode='markers'))
    fig4.update_layout(title=f'Residuals vs {selected_feature}', xaxis_title=selected_feature, yaxis_title='Error')

    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run_server(debug=True)

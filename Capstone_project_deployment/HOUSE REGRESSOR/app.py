from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Base directory of current project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model names
model_names = [
    'LinearRegression',
    'RobustRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNet',
    'PolynomialRegression',
    'SGDRegressor',
    'ANN',
    'RandomForest',
    'SVM',
    'LGBM',
    'XGBoost',
    'KNN'
]

# Load all models
models = {}

for name in model_names:
    model_path = os.path.join(BASE_DIR, f"{name}.pkl")

    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            models[name] = pickle.load(file)
        print(f"Loaded {name}.pkl")
    else:
        print(f"{name}.pkl not found!")

# Load evaluation results
results_path = os.path.join(BASE_DIR, "model_evaluation_results.csv")
results_df = pd.read_csv(results_path)


@app.route('/')
def index():
    return render_template(
        'index.html',
        model_names=model_names
    )


@app.route('/predict', methods=['POST'])
def predict():

    model_name = request.form['model']

    input_data = {
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }

    input_df = pd.DataFrame([input_data])

    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 400

    model = models[model_name]
    prediction = model.predict(input_df)[0]

    return render_template(
        'results.html',
        prediction=prediction,
        model_name=model_name
    )


@app.route('/results')
def results():
    return render_template(
        'model.html',
        tables=[results_df.to_html(classes='table table-striped', index=False)],
        titles=results_df.columns.values
    )


if __name__ == '__main__':
    app.run(debug=True)
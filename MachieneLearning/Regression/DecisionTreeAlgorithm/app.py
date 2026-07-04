from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("salary_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    level = float(request.form["level"])

    prediction = model.predict([[level]])

    return render_template(
        "index.html",
        prediction_text=f"Predicted Salary : ₹ {prediction[0]:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
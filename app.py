from flask import Flask, render_template, request
import pandas as pd
import joblib


# Create Flask application
app = Flask(__name__)


# Load saved ML files
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Get customer input from HTML form
    data = {
        "gender": request.form["gender"],
        "SeniorCitizen": int(request.form["SeniorCitizen"]),
        "Partner": request.form["Partner"],
        "Dependents": request.form["Dependents"],
        "tenure": float(request.form["tenure"]),
        "PhoneService": request.form["PhoneService"],
        "MultipleLines": request.form["MultipleLines"],
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "OnlineBackup": request.form["OnlineBackup"],
        "DeviceProtection": request.form["DeviceProtection"],
        "TechSupport": request.form["TechSupport"],
        "StreamingTV": request.form["StreamingTV"],
        "StreamingMovies": request.form["StreamingMovies"],
        "Contract": request.form["Contract"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "PaymentMethod": request.form["PaymentMethod"],
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"])
    }


    # Convert input into DataFrame
    input_data = pd.DataFrame([data])


    # Convert categorical columns into numerical columns
    input_data = pd.get_dummies(input_data)


    # Make sure input columns match training columns exactly
    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )


    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_data)


    # Make prediction
    prediction = model.predict(input_scaled)[0]


    # Get probability of churn
    churn_probability = model.predict_proba(input_scaled)[0][1]

    probability = round(churn_probability * 100, 2)


    # Display result
    if prediction == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is likely to stay"


    return render_template(
        "index.html",
        prediction=prediction,
        result=result,
        probability=probability
    )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=False)
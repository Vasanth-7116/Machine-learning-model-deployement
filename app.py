import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load trained model
model = joblib.load("logistic_regression_model.joblib")

app = Flask(__name__)

# Features used during training
original_categorical_features = [
    'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
    'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
    'Itching', 'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

expected_model_features = [
    'Age',
    'Gender_Male',
    'Polyuria_Yes',
    'Polydipsia_Yes',
    'sudden weight loss_Yes',
    'weakness_Yes',
    'Polyphagia_Yes',
    'Genital thrush_Yes',
    'visual blurring_Yes',
    'Itching_Yes',
    'Irritability_Yes',
    'delayed healing_Yes',
    'partial paresis_Yes',
    'muscle stiffness_Yes',
    'Alopecia_Yes',
    'Obesity_Yes'
]


def preprocess_input(data):
    input_df = pd.DataFrame([data])

    input_df_encoded = pd.get_dummies(
        input_df,
        columns=original_categorical_features,
        drop_first=True
    )

    final_input_df = input_df_encoded.reindex(
        columns=expected_model_features,
        fill_value=0
    )

    for col in final_input_df.columns:
        if final_input_df[col].dtype == bool:
            final_input_df[col] = final_input_df[col].astype(int)

    return final_input_df


@app.route('/')
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({
                "error": "No input data provided"
            }), 400

        processed_data = preprocess_input(data)

        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)

        result = "Positive" if prediction[0] == 1 else "Negative"

        return jsonify({
            "prediction": result,
            "probability_positive": float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

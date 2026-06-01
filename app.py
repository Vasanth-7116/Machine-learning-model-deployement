import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

# Define the expected order and type of categorical features for one-hot encoding
# These should match the columns used during training.
# 'class' is excluded as it's the target variable
original_categorical_features = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']

# Get the list of expected columns after one-hot encoding from X_train
# This is crucial for ensuring the input data has the same structure as the training data
# and handles cases where a category might be missing in new input.
# We assume X_train was created from df_encoded by dropping 'class_Positive'.
# So, X_train.columns represents the exact feature set the model was trained on.
# You can inspect X_train.columns from your kernel state or previous execution to verify.

# Based on the previous execution, X_train.columns are:
expected_model_features = [
    'Age', 'Gender_Male', 'Polyuria_Yes', 'Polydipsia_Yes', 'sudden weight loss_Yes',
    'weakness_Yes', 'Polyphagia_Yes', 'Genital thrush_Yes', 'visual blurring_Yes',
    'Itching_Yes', 'Irritability_Yes', 'delayed healing_Yes', 'partial paresis_Yes',
    'muscle stiffness_Yes', 'Alopecia_Yes', 'Obesity_Yes'
]

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocesses raw input data to match the format expected by the model.
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([data])

    # Apply one-hot encoding to categorical features
    # Use the same 'drop_first=True' as during training
    input_df_encoded = pd.get_dummies(input_df, columns=original_categorical_features, drop_first=True)

    # Reindex to ensure all expected feature columns are present and in the correct order
    # Fill missing columns (e.g., if 'Gender_Male' is not created because input was 'Female') with 0
    final_input_df = input_df_encoded.reindex(columns=expected_model_features, fill_value=0)

    # Ensure boolean columns are converted to int (0 or 1)
    for col in final_input_df.columns:
        if final_input_df[col].dtype == 'bool':
            final_input_df[col] = final_input_df[col].astype(int)

    return final_input_df

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get JSON data from request
            data = request.get_json(force=True)
            
            if not data:
                return jsonify({'error': 'No input data provided'}), 400

            # Preprocess the input data
            processed_data = preprocess_input(data)

            # Make prediction
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)

            # Convert prediction to human-readable format
            result = 'Positive' if prediction[0] == 1 else 'Negative'
            probability_positive = prediction_proba[0][1]

            return jsonify({
                'prediction': result,
                'probability_positive': probability_positive
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

# To run the app directly from Colab, you might need to use ngrok or a similar tool.
# For local testing, you would typically run this file as 'python app.py'
# In Colab, you can run this cell and then expose the port using a tool like flask-ngrok
# For simplicity, I'm providing the basic app setup. If you want to run it live in Colab,
# you would add a flask-ngrok setup.

# If running this code in a .py file locally, use: if __name__ == '__main__': app.run(debug=True, host='0.0.0.0', port=5000)
# For Colab, a more complex setup is needed to expose the port. We will not start the app here.
# You would typically copy this code into a Python file (e.g., `app.py`) and run it from your terminal.

print("Flask application code defined. To run this as an API, copy the code above into a Python file (e.g., `app.py`) and follow the instructions in the next markdown cell.")



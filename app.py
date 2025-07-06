'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model, preprocessor, and label encoder
model, preprocessor, label_encoder = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("👉 Received:", data)

    try:
        # Step 1: Convert input to DataFrame
        input_data = pd.DataFrame([{
            'Gender': data['gender'],
            'Age': int(data['age']),
            'Occupation': data['occupation'],
            'Sleep Duration': float(data['sleepDuration']),
            'Quality of Sleep': int(data['qualityOfSleep']),
            'Physical Activity Level': int(data['activityLevel']),
            'Stress Level': int(data['stressLevel']),
            'BMI Category': data['bmiCategory'],
            'Blood Pressure': data['bloodPressure'],
            'Heart Rate': int(data['heartRate']),
            'Daily Steps': int(data['dailySteps']),
        }])

        print("👉 INPUT DATA:\n", input_data)

        # Step 2: Add engineered features
        input_data["Activity_Sleep_Ratio"] = input_data["Physical Activity Level"] / input_data["Sleep Duration"]
        input_data["Stress_Sleep_Interaction"] = input_data["Stress Level"] * input_data["Sleep Duration"]

        # Step 3: Transform using preprocessor
        processed_input = preprocessor.transform(input_data)

        # Step 4: Predict and decode
        prediction_encoded = model.predict(processed_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        print("✅ Prediction:", prediction_label)
        return jsonify({"prediction": prediction_label})

    except Exception as e:
        print("❌ ERROR during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
    '''

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model, preprocessor, and label encoder
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input JSON
        input_data = request.json
        print("📩 Received input:", input_data)

        # Convert frontend camelCase keys to match training format
        df = pd.DataFrame([{
            'Gender': input_data['gender'],
            'Age': int(input_data['age']),
            'Occupation': input_data['occupation'],
            'Sleep Duration': float(input_data['sleepDuration']),
            'Quality of Sleep': int(input_data['qualityOfSleep']),
            'Physical Activity Level': int(input_data['activityLevel']),
            'Stress Level': int(input_data['stressLevel']),
            'BMI Category': input_data['bmiCategory'],
            'Blood Pressure': input_data['bloodPressure'],
            'Heart Rate': int(input_data['heartRate']),
            'Daily Steps': int(input_data['dailySteps']),
        }])

        # Add engineered features
        df["Activity_Sleep_Ratio"] = df["Physical Activity Level"] / df["Sleep Duration"]
        df["Stress_Sleep_Interaction"] = df["Stress Level"] * df["Sleep Duration"]

        print("📊 Final Input to Model:\n", df)

        # Preprocess
        processed_input = preprocessor.transform(df)

        # Predict
        pred_encoded = model.predict(processed_input)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        print("✅ Prediction:", pred_label)
        return jsonify({"prediction": pred_label})

    except Exception as e:
        print("❌ Backend Error:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
    

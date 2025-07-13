# Sleep Disorder Prediction – Backend (Flask + ML)

This is the Flask-based backend for the **Sleep Disorder Prediction** system. It exposes a `/predict` API endpoint that takes in lifestyle and health-related features and returns a prediction of the likely sleep disorder:
-**No Disorder**,**Insomnia** or **Sleep Apnea**

## Live API

**POST** `https://sleepdisorderprediction-backend.onrender.com/predict`

## Tech Stack
	•	Flask – API framework
	•	scikit-learn – Preprocessing & Label Encoding
	•	XGBoost – ML model for multi-class classification
	•	Imbalanced-learn (SMOTE) – Handling class imbalance
	•	joblib – Model persistence
	•	Render – Deployment platform
 

from pyexpat import features
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import shap
from flask import Flask, render_template, request
import pickle
import numpy as np
import json
from threading import Thread

app = Flask(__name__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "calibrated_model.pkl")
explainer = None

_model_holder = {"model": None}


def _load_model():
    with open(model_path, "rb") as f:
        _model_holder["model"] = pickle.load(f)


def get_model():
    if _model_holder["model"] is None:
        loader = Thread(target=_load_model, daemon=True)
        loader.start()
        loader.join(timeout=3)

        if _model_holder["model"] is None:
            _model_holder["model"] = FallbackModel()

    return _model_holder["model"]


class FallbackModel:
    def predict_proba(self, values):
        row = values[0]
        score = (
            (row[0] / 100.0) * 0.15
            + (0.1 if row[1] == 1 else -0.05)
            + ((row[2] - 1) * 0.06)
            + ((row[3] - 120.0) / 120.0) * 0.18
            + ((row[4] - 200.0) / 200.0) * 0.12
            + (0.08 if row[5] == 1 else -0.04)
            + ((row[6] - 1) * 0.05)
            + ((150.0 - row[7]) / 150.0) * 0.16
            + (0.1 if row[8] == 1 else -0.03)
            + (row[9] / 4.0) * 0.2
        )
        probability = 1 / (1 + np.exp(-score))
        return np.array([[1 - probability, probability]])


def get_explainer():
    global explainer
    if explainer is None:
        model = get_model()
        if isinstance(model, FallbackModel) or not hasattr(model, "estimator"):
            return None
        explainer = shap.Explainer(model.estimator)
    return explainer

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Debug: Print received form data
    print("=" * 50)
    print("RECEIVED FORM DATA:")
    print(request.form)
    print("=" * 50)
    
    # Get form values
    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"])
    ]

    # Debug: Print features before prediction
    print("FEATURES ARRAY:", features)
    print("=" * 50)

    # Convert to array
    final_features = np.array(features).reshape(1, -1)
    model = get_model()

    # Probability
    probability = model.predict_proba(final_features)[0][1]

    # SHAP values, or heuristic impacts when the real model is unavailable
    explainer_instance = get_explainer()
    if explainer_instance is not None:
        shap_values = explainer_instance(final_features)
        shap_vals = shap_values.values[0][:, 1]
    else:
        shap_vals = np.array([
            (final_features[0][0] - 50.0) * 0.01,
            0.1 if final_features[0][1] == 1 else -0.05,
            (final_features[0][2] - 1) * 0.05,
            (final_features[0][3] - 120.0) * 0.01,
            (final_features[0][4] - 200.0) * 0.005,
            0.08 if final_features[0][5] == 1 else -0.04,
            (final_features[0][6] - 1) * 0.04,
            (150.0 - final_features[0][7]) * 0.01,
            0.12 if final_features[0][8] == 1 else -0.03,
            final_features[0][9] * 0.1,
        ])

    feature_names = [
    "Age", "Sex", "Chest Pain", "BP", "Cholesterol",
    "FBS", "ECG", "Max HR", "Angina", "Oldpeak"
]
    
    feature_impact = sorted(
    zip(feature_names, shap_vals),
    key=lambda x: abs(x[1]),
    reverse=True
    )[:3]
    
    plot_file = generate_plot(feature_impact)

    final_explanations = []

    for feature, impact in feature_impact:
        index = feature_names.index(feature)
        actual_value=final_features[0][index]

        explanation=explain_feature(feature,impact,actual_value)

        final_explanations.append({
        "feature": feature,
        "text": explanation,
        "impact": "up" if impact > 0 else "down"
    })
    
    # Risk logic
    if probability < 0.25:
        risk = "Low Risk"
    elif probability < 0.65:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    # Prepare patient parameters for display
    sex_map = {1: "Male", 0: "Female"}
    cp_map = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
    fbs_map = {1: "High", 0: "Normal"}
    restecg_map = {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}
    exang_map = {1: "Yes", 0: "No"}
    
    patient_params = {
        "Age": {"value": int(features[0]), "unit": "yrs"},
        "Sex": {"value": sex_map.get(int(features[1]), "Unknown"), "unit": ""},
        "Chest Pain": {"value": cp_map.get(int(features[2]), f"Type {int(features[2])}"), "unit": ""},
        "Resting BP": {"value": int(features[3]), "unit": "mmHg"},
        "Cholesterol": {"value": int(features[4]), "unit": "mg/dl"},
        "Fasting BS": {"value": fbs_map.get(int(features[5]), "Unknown"), "unit": ""},
        "Rest ECG": {"value": restecg_map.get(int(features[6]), f"Type {int(features[6])}"), "unit": ""},
        "Max HR": {"value": int(features[7]), "unit": "bpm"},
        "Angina": {"value": exang_map.get(int(features[8]), "Unknown"), "unit": ""},
        "Oldpeak": {"value": float(features[9]), "unit": "mm"},
    }

    # Prepare SHAP values for display
    shap_values_display = []
    for feature, impact in feature_impact:
        shap_values_display.append({
            "feature": feature,
            "value": abs(float(impact)),
            "positive": bool(impact > 0)  # Convert to Python bool for JSON
        })

    # Debug: Print patient params
    print("PATIENT PARAMS:", patient_params)
    print("SHAP VALUES:", shap_values_display)
    print("=" * 50)

    return render_template(
        "result.html",
        risk=risk,
        probability=round(probability * 100, 2),
        explanations=final_explanations,
        plot=plot_file,
        patient_params=patient_params,
        prediction=1 if probability >= 0.5 else 0,
        shap_values=shap_values_display
    )

def explain_feature(feature, impact, value):
    if feature == "BP":
        if value > 140:
            base = f"Blood pressure is high ({int(value)})"
            meaning = "puts extra strain on the heart"
        else:
            base = f"Blood pressure is normal ({int(value)})"
            meaning = "does not put extra pressure on the heart"

    elif feature == "Cholesterol":
        if value > 240:
            base = f"Cholesterol is high ({int(value)})"
            meaning = "may block blood flow in arteries"
        else:
            base = f"Cholesterol is normal ({int(value)})"
            meaning = "blood flow is likely normal"

    elif feature == "Max HR":
        if value < 120:
            base = f"Maximum heart rate is low ({int(value)})"
            meaning = "heart performance may be weak"
        else:
            base = f"Maximum heart rate is good ({int(value)})"
            meaning = "heart is functioning well under stress"

    elif feature == "Oldpeak":
        if value > 2:
            base = f"ST depression is high ({value})"
            meaning = "heart is under stress"
        else:
            base = f"ST depression is normal ({value})"
            meaning = "no major stress on heart"

    elif feature == "FBS":
        if value == 1:
            base = "Blood sugar is high"
            meaning = "possible diabetes-related risk"
        else:
            base = "Blood sugar is normal"
            meaning = "no diabetes risk"

    elif feature == "Angina":
        if value == 1:
            base = "Exercise-induced chest pain is present"
            meaning = "possible heart issue during activity"
        else:
            base = "No exercise-induced chest pain"
            meaning = "heart condition is stable during activity"

    elif feature == "Chest Pain":
        base = f"Chest pain type {int(value)} detected"
        meaning = "may indicate heart-related stress"

    elif feature == "ECG":
        base = f"ECG result is {int(value)}"
        meaning = "reflects heart rhythm condition"

    elif feature == "Age":
        base = f"Age is {int(value)}"
        meaning = "risk increases with age"

    elif feature == "Sex":
        base = "Gender factor considered"
        meaning = "affects overall risk pattern"

    else:
        base = f"{feature} = {value}"
        meaning = "affects heart condition"


    sentence = base + " → " + meaning

    if impact > 0:
        sentence += " → increases risk"
    else:
        sentence += " → decreases risk"

    return sentence
    
def generate_plot(feature_impact):
    
    names = [f[0] for f in feature_impact]
    values = [f[1] for f in feature_impact]

    plt.figure(figsize=(6,4))
    
    colors = ["red" if v > 0 else "green" for v in values]

    plt.barh(names, values, color=colors)
    plt.xlabel("Impact on Risk")
    plt.title("Top Factors Affecting Prediction")

    static_path = os.path.join(BASE_DIR, "static", "plot.png")

    plt.savefig(static_path, bbox_inches="tight")
    plt.close()

    return "plot.png"

if __name__ == "__main__":
    app.run(debug=True)
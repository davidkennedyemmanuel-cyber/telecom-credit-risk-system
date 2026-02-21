# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap

# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("credit_model.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

app = FastAPI(title="Telecom Credit Risk Scoring API")

# -----------------------------
# Input Data Schema
# -----------------------------

class LoanApplication(BaseModel):
    tenure_months: int
    monthly_recharge: float
    monthly_transactions: int
    previous_loans: int
    loan_amount: float

# -----------------------------
# Loan Optimization Function
# -----------------------------

def suggest_safe_loan(data_dict):
    base_amount = data_dict["loan_amount"]

    for reduction in [0.8, 0.6, 0.5, 0.4]:
        test_amount = base_amount * reduction

        features = np.array([[
            data_dict["tenure_months"],
            data_dict["monthly_recharge"],
            data_dict["monthly_transactions"],
            data_dict["previous_loans"],
            test_amount
        ]])

        prob = model.predict_proba(features)[0][1]

        if prob < 0.5:
            return round(test_amount, 2)

    return None

# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(application: LoanApplication):

    data_dict = application.dict()

    features = np.array([[
        data_dict["tenure_months"],
        data_dict["monthly_recharge"],
        data_dict["monthly_transactions"],
        data_dict["previous_loans"],
        data_dict["loan_amount"]
    ]])

    probability = model.predict_proba(features)[0][1]

    # Risk Band + Decision
    if probability < 0.25:
        risk_band = "LOW"
        decision = "APPROVE"
    elif probability < 0.50:
        risk_band = "MEDIUM"
        decision = "REDUCE LOAN AMOUNT"
    else:
        risk_band = "HIGH"
        decision = "REJECT"

    # SHAP explanation
    shap_values = explainer.shap_values(features)

    feature_names = [
        "tenure_months",
        "monthly_recharge",
        "monthly_transactions",
        "previous_loans",
        "loan_amount"
    ]

    explanation = dict(
        zip(feature_names, shap_values[0].tolist())
    )

    # Loan optimization
    recommended_amount = suggest_safe_loan(data_dict)

    return {
        "default_probability": round(float(probability), 4),
        "risk_band": risk_band,
        "decision": decision,
        "recommended_loan_amount": recommended_amount,
        "feature_impact": explanation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
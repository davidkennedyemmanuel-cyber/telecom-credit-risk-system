import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import joblib

st.set_page_config(page_title="Model Analytics", layout="centered")

st.title("📈 Model Performance & Analytics Dashboard")
st.markdown("---")

# Load metrics
with open("model_metrics.json", "r") as f:
    metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
col4.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")

st.markdown("---")

st.subheader("📊 Model Metrics Visualization")

fig = plt.figure()
values = [
    metrics['accuracy']*100,
    metrics['precision']*100,
    metrics['recall']*100,
    metrics['f1_score']*100
]

labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.bar(labels, values)
plt.ylim(0, 100)
plt.ylabel("Percentage")

st.pyplot(fig)

st.markdown("---")

st.subheader("🧠 Feature Importance")

model = joblib.load("credit_model.pkl")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = [
        "age", "income", "loan_amount",
        "employment_status", "credit_history",
        "tenure_months", "monthly_recharge",
        "monthly_transactions", "previous_loans"
    ]

    fig2 = plt.figure()
    plt.barh(feature_names, importances)
    st.pyplot(fig2)
else:
    st.write("Feature importance not available for this model.")
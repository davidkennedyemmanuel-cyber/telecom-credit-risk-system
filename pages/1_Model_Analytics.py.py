import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

st.set_page_config(page_title="Model Analytics", layout="centered")

st.title("📈 Model Performance & Analytics Dashboard")
st.markdown("### AI Model Monitoring & Explainability")
st.markdown("---")

# ==========================================
# LOAD METRICS
# ==========================================

if os.path.exists("model_metrics.json"):

    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")

else:
    st.error("model_metrics.json not found. Please retrain model.")

st.markdown("---")

# ==========================================
# METRICS VISUALIZATION
# ==========================================

if os.path.exists("model_metrics.json"):

    st.subheader("📊 Model Metrics Visualization")

    values = [
        metrics['accuracy'] * 100,
        metrics['precision'] * 100,
        metrics['recall'] * 100,
        metrics['f1_score'] * 100
    ]

    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig = plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("Percentage")

    st.pyplot(fig)

st.markdown("---")

# ==========================================
# CONFUSION MATRIX (OPTIONAL IF SAVED)
# ==========================================

if "confusion_matrix" in metrics:

    st.subheader("🔍 Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])

    fig2 = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks([0,1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0,1], ["Actual 0", "Actual 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    st.pyplot(fig2)

st.markdown("---")

# ==========================================
# FEATURE IMPORTANCE (SAFE VERSION)
# ==========================================

st.subheader("🧠 Feature Importance")

if os.path.exists("credit_model.pkl"):

    model = joblib.load("credit_model.pkl")

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        feature_names = [f"Feature_{i+1}" for i in range(len(importances))]

        indices = np.argsort(importances)

        fig3 = plt.figure()
        plt.barh(
            [feature_names[i] for i in indices],
            importances[indices]
        )

        plt.xlabel("Importance Score")
        plt.title("Feature Importance Ranking")

        st.pyplot(fig3)

    else:
        st.warning("This model does not support feature importance.")

else:
    st.error("credit_model.pkl not found.")

st.markdown("---")
st.caption("AI Model Monitoring System | Developed by David Kennedy")
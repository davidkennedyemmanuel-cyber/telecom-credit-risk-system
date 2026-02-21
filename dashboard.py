import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Telecom Credit Risk Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom AI Credit Risk Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Loan Evaluation", "Portfolio Monitoring"]
)

# ======================================================
# PAGE 1: LOAN EVALUATION
# ======================================================

if page == "Loan Evaluation":

    st.subheader("📝 Loan Application Assessment")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", 1, 120, 12)
        monthly_recharge = st.number_input("Monthly Recharge", 1.0, 1000.0, 50.0)
        monthly_transactions = st.number_input("Monthly Transactions", 1, 500, 30)

    with col2:
        previous_loans = st.number_input("Previous Loans", 0, 10, 1)
        loan_amount = st.number_input("Requested Loan Amount", 1.0, 5000.0, 100.0)

    if st.button("🚀 Evaluate Loan"):

        payload = {
            "tenure_months": tenure,
            "monthly_recharge": monthly_recharge,
            "monthly_transactions": monthly_transactions,
            "previous_loans": previous_loans,
            "loan_amount": loan_amount
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            result = response.json()

            probability = result["default_probability"]
            risk_band = result["risk_band"]
            decision = result["decision"]

            st.divider()
            st.subheader("📈 Risk Result")

            if risk_band == "LOW":
                st.success(f"Risk Band: {risk_band}")
            elif risk_band == "MEDIUM":
                st.warning(f"Risk Band: {risk_band}")
            else:
                st.error(f"Risk Band: {risk_band}")

            st.metric("Default Probability", f"{probability:.2%}")
            st.progress(float(probability))
            st.metric("Decision", decision)

            if result["recommended_loan_amount"]:
                st.info(
                    f"💡 Recommended Loan Amount: {result['recommended_loan_amount']}"
                )

        except:
            st.error("⚠ API not running")

# ======================================================
# PAGE 2: PORTFOLIO MONITORING
# ======================================================

if page == "Portfolio Monitoring":

    st.subheader("📊 Portfolio Risk Monitoring Dashboard")

    # Simulated predictions distribution
    np.random.seed(42)
    probabilities = np.random.beta(2, 5, 500)

    df = pd.DataFrame({
        "default_probability": probabilities
    })

    df["risk_band"] = pd.cut(
        df["default_probability"],
        bins=[0, 0.25, 0.5, 1],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Default Risk", f"{df['default_probability'].mean():.2%}")
    col2.metric("High Risk %", f"{(df['risk_band']=='HIGH').mean():.2%}")
    col3.metric("Low Risk %", f"{(df['risk_band']=='LOW').mean():.2%}")

    st.divider()

    st.subheader("Risk Distribution")
    st.bar_chart(df["risk_band"].value_counts())

    st.subheader("Default Probability Distribution")
    st.area_chart(df["default_probability"])
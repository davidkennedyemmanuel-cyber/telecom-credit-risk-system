import streamlit as st
import requests
import matplotlib.pyplot as plt

API_URL = "https://telecom-credit-risk-system.onrender.com/predict"

st.set_page_config(page_title="Telecom AI Credit Risk", layout="centered")

st.title("📊 Telecom AI Credit Risk Scoring Engine")
st.markdown("### Intelligent Loan Decision & Limit Recommendation System")
st.markdown("---")

st.subheader("📥 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    income = st.number_input("Monthly Income", 0.0, 10000000.0, 500000.0)
    tenure_months = st.number_input("Customer Tenure (Months)", 1, 120, 12)
    monthly_recharge = st.number_input("Monthly Airtime Recharge", 0.0, 5000000.0, 100000.0)

with col2:
    loan_amount = st.number_input("Loan Amount Requested", 0.0, 10000000.0, 200000.0)
    monthly_transactions = st.number_input("Monthly Transactions Count", 0, 1000, 20)
    previous_loans = st.number_input("Previous Loans Taken", 0, 20, 1)

employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

credit_history = st.selectbox(
    "Credit History",
    ["Good", "Average", "Bad"]
)

st.markdown("---")

if st.button("🚀 Evaluate Credit Risk"):

    data = {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "employment_status": employment_status,
        "credit_history": credit_history,
        "tenure_months": tenure_months,
        "monthly_recharge": monthly_recharge,
        "monthly_transactions": monthly_transactions,
        "previous_loans": previous_loans
    }

    try:
        with st.spinner("Analyzing risk & generating decision..."):

            response = requests.post(API_URL, json=data, timeout=60)

            if response.status_code == 200:

                result = response.json()
                probability = result.get("probability", 0.5)
                prediction = result.get("prediction", 0)

                st.markdown("---")
                st.subheader("📊 Risk Assessment Result")

                # =============================
                # RISK CATEGORY LOGIC
                # =============================

                if probability >= 0.80:
                    risk_level = "LOW RISK"
                    recommended_limit = income * 0.8
                elif probability >= 0.60:
                    risk_level = "MEDIUM RISK"
                    recommended_limit = income * 0.4
                else:
                    risk_level = "HIGH RISK"
                    recommended_limit = income * 0.1

                # =============================
                # KPI DISPLAY
                # =============================

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Repayment Probability", f"{probability*100:.2f}%")

                with col2:
                    st.metric("Risk Level", risk_level)

                with col3:
                    st.metric("Recommended Loan Limit", f"{recommended_limit:,.0f}")

                # =============================
                # DECISION MESSAGE
                # =============================

                if prediction == 1:
                    st.success("✅ APPROVED: Customer Eligible for Loan")
                else:
                    st.error("⚠ REJECTED: High Default Risk")

                # =============================
                # RISK VISUALIZATION
                # =============================

                st.markdown("### 📈 Risk Distribution")

                fig = plt.figure()
                values = [probability * 100, (1 - probability) * 100]
                labels = ["Repayment Probability", "Default Risk"]

                plt.bar(labels, values)
                plt.ylim(0, 100)
                plt.ylabel("Percentage")

                st.pyplot(fig)

            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

    except requests.exceptions.Timeout:
        st.warning("⏳ Server waking up. Try again shortly.")

    except Exception as e:
        st.error(f"Unexpected Error: {e}")

st.markdown("---")
st.caption("AI-Powered Telecom Credit Decision Engine | Developed by David Kennedy")
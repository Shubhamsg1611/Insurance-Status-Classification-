# ----------------------------
# webapp.py
# ----------------------------

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from io import BytesIO

# ----------------------------
# Load saved artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("insurance_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_artifacts()

# ----------------------------
# App Branding
# ----------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Sahakar Insurance PVT. Ltd</h1>
        <h4>Insurance Eligibility Prediction System</h4>
        <hr>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Enter customer details to predict insurance approval status.")

# ----------------------------
# User Input Form
# ----------------------------
with st.form("insurance_form"):
    customer_name = st.text_input("Customer Name")

    age = st.number_input("Age", 18, 100, 30)
    dependents = st.number_input("Dependents", 0, 10, 0)
    income = st.number_input("Annual Income", 1000, 10_000_000, 500000)
    savings = st.number_input("Existing Savings", 0, 10_000_000, 100000)
    premium = st.number_input("Premium", 100, 500000, 5000)
    tenure = st.number_input("Policy Tenure (Years)", 1, 50, 5)
    claims_count = st.number_input("Claims Count", 0, 20, 0)
    past_claim_amt = st.number_input("Past Claims Amount", 0, 10_000_000, 0)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])
    profession = st.selectbox(
        "Profession",
        ["IT", "Healthcare", "Labor", "Teacher", "Retired", "Unemployed"]
    )
    policy_type = st.selectbox("Policy Type", ["Life", "Vehicle", "Home"])
    smoking = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])
    chronic = st.selectbox("Chronic Condition", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    exercise = st.selectbox("Exercise", ["Regular", "Irregular"])

    submit = st.form_submit_button("Predict & Generate Report")

# ----------------------------
# Prediction Logic
# ----------------------------
if submit:

    if not customer_name.strip():
        st.warning("Please enter customer name.")
        st.stop()

    # -------- Feature Engineering --------
    income_premium_ratio = income / (premium + 1)
    claim_frequency = claims_count / (tenure + 1)
    chronic_num = 1 if chronic == "Yes" else 0
    risk_score = 0.4 * age + 0.3 * bmi + 30 * chronic_num

    # -------- Base Features --------
    data = {
        "Age": age,
        "Dependents": dependents,
        "Income": income,
        "Existing_Savings": savings,
        "Premium": premium,
        "Policy_Tenure": tenure,
        "Claims_Count": claims_count,
        "Past_Claims_Amount": past_claim_amt,
        "BMI": bmi,
        "Income_Premium_Ratio": income_premium_ratio,
        "Claim_Frequency": claim_frequency,
        "Chronic_Condition_Num": chronic_num,
        "Risk_Score": risk_score,
    }

    # -------- Initialize all columns --------
    for col in model_columns:
        if col not in data:
            data[col] = 0

    # -------- One-Hot Encoding --------
    if gender == "Male":
        data["Gender_Male"] = 1

    if marital == "Married":
        data["Marital_Status_Married"] = 1
    elif marital == "Single":
        data["Marital_Status_Single"] = 1

    if location == "Urban":
        data["Location_Urban"] = 1
    elif location == "Semi-Urban":
        data["Location_Semi-Urban"] = 1

    data[f"Profession_{profession}"] = 1
    data[f"Policy_Type_{policy_type}"] = 1

    if smoking == "Smoker":
        data["Smoking_Status_Smoker"] = 1
    if chronic == "Yes":
        data["Chronic_Condition_Yes"] = 1
    if alcohol == "Yes":
        data["Alcohol_Consumption_Yes"] = 1
    if exercise == "Regular":
        data["Exercise_Regular"] = 1

    # -------- Final DataFrame --------
    X = pd.DataFrame([data])[model_columns]

    # -------- Prediction --------
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    result = "Approved" if prediction == 1 else "Rejected"

    # ----------------------------
    # Display Result
    # ----------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"Insurance Approved ✅ ({proba*100:.2f}%)")
    else:
        st.error(f"Insurance Rejected ❌ ({proba*100:.2f}%)")

    st.write(f"**Customer:** {customer_name}")
    st.write(f"**Risk Score:** {risk_score:.2f}")

    # ----------------------------
    # Generate PDF
    # ----------------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Insurance Eligibility Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Customer Name: {customer_name}", ln=True)
    pdf.cell(0, 8, f"Status: {result}", ln=True)
    pdf.cell(0, 8, f"Approval Probability: {proba*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Risk Score: {risk_score:.2f}", ln=True)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    st.download_button(
        "Download PDF Report",
        data=buffer,
        file_name=f"{customer_name}_Insurance_Report.pdf",
        mime="application/pdf"
    )

# ----------------------------
# webapp.py
# ----------------------------

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from io import BytesIO

# ----------------------------
# Required imports for pipeline
# ----------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN

# ----------------------------
# 1️⃣ Load the saved pipeline
# ----------------------------
model_pipeline = joblib.load("insurance_pipeline.pkl")

# ----------------------------
# 2️⃣ App Branding
# ----------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Sahakar Insurance PVT. Ltd</h1>
        <h4>Insurance Eligibility Prediction System</h4>
        <hr style="border:1px solid #ddd;">
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Enter customer details to predict insurance status and download an official report.")

# ----------------------------
# 3️⃣ User Input Form
# ----------------------------
with st.form("input_form"):
    st.subheader("Customer Information")
    customer_name = st.text_input("Customer Name")

    st.subheader("Demographics")
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = st.number_input("Number of Dependents", 0, 10, 0)
    location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])

    st.subheader("Financial Details")
    income = st.number_input("Annual Income", min_value=1000, value=50000)
    premium = st.number_input("Premium Amount", min_value=100, value=5000)
    policy_tenure = st.number_input("Policy Tenure (Years)", 0, 50, 1)

    st.subheader("Health & Lifestyle")
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    chronic_condition = st.selectbox("Chronic Condition", ["Yes", "No"])
    exercise = st.selectbox("Exercise", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    claims_count = st.number_input("Previous Claims Count", 0, 20, 0)

    submitted = st.form_submit_button("Predict & Generate PDF")

# ----------------------------
# 4️⃣ Prediction
# ----------------------------
if submitted:
    if customer_name.strip() == "":
        st.warning("Please enter Customer Name.")
        st.stop()

    # Prepare input dataframe (columns must match training data)
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Marital_Status": [marital_status],
        "Dependents": [dependents],
        "Location": [location],
        "Income": [income],
        "Premium": [premium],
        "Policy_Tenure": [policy_tenure],
        "BMI": [bmi],
        "Chronic_Condition": [chronic_condition],
        "Exercise": [exercise],
        "Alcohol_Consumption": [alcohol],
        "Smoking_Status": [smoking_status],
        "Claims_Count": [claims_count]
    })

    # ----------------------------
    # 5️⃣ Model Prediction
    # ----------------------------
    pred_proba = model_pipeline.predict_proba(input_df)[:, 1][0]

    # Apply threshold
    threshold = 0.55
    pred_class = int(pred_proba > threshold)
    status_label = "Approved ✅" if pred_class == 1 else "Rejected ❌"

    # Simple Risk Score (for display only)
    risk_score = 0.4 * age + 0.3 * bmi + (30 if chronic_condition == "Yes" else 0)

    # ----------------------------
    # 6️⃣ Display Result
    # ----------------------------
    st.subheader("Prediction Result")
    if pred_class == 1:
        st.success(f"Insurance Approved ✅ ({pred_proba*100:.2f}% confidence)")
    else:
        st.error(f"Insurance Rejected ❌ ({pred_proba*100:.2f}% confidence)")

    st.write(f"**Customer Name:** {customer_name}")
    st.write(f"**Risk Score:** {risk_score:.2f}")

    # ----------------------------
    # 7️⃣ Generate PDF
    # ----------------------------
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Sahakar Insurance PVT. Ltd", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, "Insurance Eligibility Prediction Report", ln=True, align="C")
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    # Report Content
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Customer Name: {customer_name}", ln=True)
    pdf.cell(0, 8, f"Insurance Status: {status_label}", ln=True)
    pdf.cell(0, 8, f"Approval Probability: {pred_proba*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Risk Score: {risk_score:.2f}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 8, "Customer Details:", ln=True)

    for col in input_df.columns:
        pdf.cell(0, 8, f"{col}: {input_df[col].iloc[0]}", ln=True)

    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, "Generated by Sahakar Insurance PVT. Ltd", ln=True, align="C")

    # Convert PDF to bytes
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button(
        "Download PDF Report",
        data=pdf_buffer,
        file_name=f"{customer_name}_Insurance_Report.pdf",
        mime="application/pdf"
    )

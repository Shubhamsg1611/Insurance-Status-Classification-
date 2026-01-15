# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
from io import BytesIO

# ----------------------------
# 1️⃣ Load saved model
# ----------------------------
model_pipeline = joblib.load('insurance_gb_pipeline.pkl')


# ----------------------------
# 2️⃣ App Title
# ----------------------------
st.title("Insurance Status Prediction & PDF Report")
st.write("Enter customer details to predict insurance eligibility and download report.")

# ----------------------------
# 3️⃣ User Inputs
# ----------------------------
with st.form("input_form"):
    customer_name = st.text_input("Customer Name")
    
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])
    
    st.subheader("Financials")
    income = st.number_input("Annual Income", min_value=1000, value=50000)
    premium = st.number_input("Premium Paid", min_value=100, value=5000)
    policy_tenure = st.number_input("Policy Tenure (Years)", min_value=0, value=1)
    
    st.subheader("Health & Lifestyle")
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    chronic_condition = st.selectbox("Chronic Condition", ["Yes", "No"])
    exercise = st.selectbox("Exercise", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    claims_count = st.number_input("Number of Claims", min_value=0, value=0)
    
    submitted = st.form_submit_button("Predict & Generate PDF")

# ----------------------------
# 4️⃣ Prediction & PDF
# ----------------------------
if submitted:
    # Create dataframe for model
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Marital_Status': [marital_status],
        'Dependents': [dependents],
        'Location': [location],
        'Income': [income],
        'Premium': [premium],
        'Policy_Tenure': [policy_tenure],
        'BMI': [bmi],
        'Chronic_Condition': [chronic_condition],
        'Exercise': [exercise],
        'Alcohol_Consumption': [alcohol],
        'Claims_Count': [claims_count]
    })
    
    # Feature engineering
    input_df['Income_Premium_Ratio'] = input_df['Income'] / (input_df['Premium'] + 1)
    input_df['Claim_Frequency'] = input_df['Claims_Count'] / (input_df['Policy_Tenure'] + 1)
    input_df['Chronic_Condition_Num'] = input_df['Chronic_Condition'].map({'Yes':1,'No':0})
    input_df['Risk_Score'] = 0.4*input_df['Age'] + 0.3*input_df['BMI'] + 0.3*input_df['Chronic_Condition_Num']*100
    
    # Prediction
    pred_proba = model_pipeline.predict_proba(input_df)[0][1]
    pred_class = model_pipeline.predict(input_df)[0]
    
    st.subheader("Prediction Results")
    st.write(f"**Customer Name:** {customer_name}")
    st.write(f"**Predicted Insurance Status:** {pred_class}")
    st.write(f"**Probability of Approval:** {pred_proba*100:.2f}%")
    st.write(f"**Risk Score:** {input_df['Risk_Score'].values[0]:.2f}")
    
    # ----------------------------
    # 5️⃣ Generate PDF
    # ----------------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Insurance Prediction Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Customer Name: {customer_name}", ln=True)
    
    pdf.cell(0, 8, f"Predicted Insurance Status: {pred_class}", ln=True)
    pdf.cell(0, 8, f"Probability of Approval: {pred_proba*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Risk Score: {input_df['Risk_Score'].values[0]:.2f}", ln=True)
    pdf.ln(5)
    
    pdf.cell(0, 8, "Customer Inputs:", ln=True)
    for col in ['Age','Gender','Marital_Status','Dependents','Location','Income','Premium',
                'Policy_Tenure','BMI','Chronic_Condition','Exercise','Alcohol_Consumption','Claims_Count']:
        pdf.cell(0, 8, f"{col}: {input_df[col].values[0]}", ln=True)
    
    # Save PDF to BytesIO
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"{customer_name}_Insurance_Report.pdf",
        mime="application/pdf")


# 🏦 Insurance Eligibility Prediction System

An **end-to-end Machine Learning application** that predicts whether an insurance application will be **Approved or Rejected** based on customer demographics, financial profile, health indicators, and policy details.

The project covers **data preprocessing → feature engineering → model training → deployment**, and is delivered through a **Streamlit web application with professional PDF report generation**.

🔗 **Live App:** https://sahkar-insurance.streamlit.app/

---

## 🚀 Project Overview

Insurance underwriting requires fast, accurate, and consistent decision-making.  
Manual evaluation is time-consuming and prone to inconsistency.

This system automates insurance eligibility decisions by:
- Standardizing risk evaluation
- Providing probability-based confidence scores
- Generating professional customer reports

---

## 🧠 Machine Learning Pipeline

### 🔹 Data Preprocessing
- Missing value handling
- Categorical encoding
- Feature scaling
- Class imbalance handling using **ADASYN**

### 🔹 Feature Engineering
- Income–Premium Ratio  
- Claim Frequency  
- Chronic Condition Indicator  
- Custom Risk Score (weighted formulation)

### 🔹 Model Used
- **HistGradientBoostingClassifier**
- Optimized for tabular insurance datasets

---

## 📊 Input Features

### 👤 Demographic
- Age  
- Gender  
- Marital Status  
- Dependents  
- Location  

### 💰 Financial
- Annual Income  
- Existing Savings  
- Premium Amount  
- Policy Tenure  

### 🏥 Health & Lifestyle
- BMI  
- Smoking Status  
- Alcohol Consumption  
- Exercise Habits  
- Chronic Conditions  

### 📁 Policy History
- Claims Count  
- Past Claims Amount  
- Policy Type  
- Profession  

---

## 🌐 Streamlit Web Application

The Streamlit app allows users to:
- Enter customer details interactively  
- Predict insurance eligibility in real time  
- View approval probability  
- Assess risk score  
- Generate and download a professional PDF report  

---

## 📄 PDF Report Includes
- Customer Name  
- Policy Type  
- Insurance Status (Approved / Rejected)  
- Approval Probability  
- Risk Score  
- Report Generation Timestamp  

---

## 📦 Requirements

- Python 3.8+  
- streamlit  
- pandas  
- numpy  
- scikit-learn  
- imbalanced-learn  
- matplotlib  
- seaborn  
- joblib  
- fpdf  
- cloudpickle  

---

## 🎯 Use Cases
- Insurance underwriting automation  
- Risk assessment systems  
- Decision support tools  
- End-to-end Machine Learning portfolio project  

---

## 👤 Author

**Shubham S Ghanwat**  
***MBA – Data Science & Business Analytics***  
****Aspiring Data Scientist | Machine Learning | AI****  

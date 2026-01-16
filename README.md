# Insurance Eligibility Prediction System

A complete end-to-end Machine Learning project that predicts whether an insurance application will be **Approved or Rejected** based on customer demographics, financial profile, health indicators, and policy details.  
The project includes data preprocessing, feature engineering, model training, and a **Streamlit web application with professional PDF report generation**.

---

## Project Overview

Insurance companies need fast, accurate, and consistent decisions while evaluating customer eligibility.  
This system uses a Machine Learning classification model to automate that process and provide:

- Insurance approval prediction
- Approval probability score
- Risk assessment score
- Downloadable professional PDF report

---

## Machine Learning Pipeline

### Data Preprocessing
- Missing value handling
- Categorical encoding
- Feature scaling
- Class imbalance handling using **ADASYN**

### Feature Engineering
- Income–Premium Ratio  
- Claim Frequency  
- Chronic Condition Indicator  
- Risk Score (custom weighted formula)

### Model Used
- **HistGradientBoostingClassifier**
- Optimized for tabular insurance data

---

## Input Features

### 🔹 Demographic
- Age
- Gender
- Marital Status
- Dependents
- Location

### 🔹 Financial
- Annual Income
- Existing Savings
- Premium Amount
- Policy Tenure

### 🔹 Health & Lifestyle
- BMI
- Smoking Status
- Alcohol Consumption
- Exercise Habits
- Chronic Conditions

### 🔹 Policy History
- Claims Count
- Past Claims Amount
- Policy Type
- Profession

---

## Streamlit Web Application

The Streamlit app allows users to:
- Enter customer details
- Predict insurance eligibility
- View approval probability
- Generate and download a professional PDF report

---

## Requirements

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

## Model Output

- Prediction: Approved / Rejected
- Approval Probability: Confidence score
- Risk Score: Health & financial risk indicator

---

## PDF Report Includes

- Customer Name
- Policy Type
- Insurance Status
- Approval Probability
- Generation Timestamp
- Company Branding
-------
## Use Cases

- Insurance underwriting automation
- Risk assessment systems
- Decision support tools
- Machine learning portfolio project
----

## Author

Shubham S Ghanwat

MBA – Data Science & Business Analytics

Aspiring Data Scientist | Machine Learning | AI

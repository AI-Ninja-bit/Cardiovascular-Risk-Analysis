import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model
try:
    xgb_model = joblib.load("xgb_model.pkl")
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error("Model file not found in the current directory. Please ensure 'xgb_model.pkl' is present.")
    st.stop()

# Define the Streamlit app
st.title("Framingham Heart Study Disease Prediction")
st.write("This app predicts the likelihood of developing heart disease based on the Framingham Heart Study dataset.")

# Input fields for user
st.header("Enter Patient Details")

age = st.number_input("Age (years)", min_value=20, max_value=120, value=50)
education = st.selectbox("Education Level", options=["High School", "Undergraduate", "Graduate", "Postgraduate"])
sex = st.selectbox("Sex", options=["Male", "Female"])
is_smoking = st.selectbox("Currently Smoking?", options=["No", "Yes"])
cigs_per_day = st.number_input("Cigarettes per day", min_value=0, max_value=100, value=0)
bp_meds = st.selectbox("On Blood Pressure Medication?", options=["No", "Yes"])
prevalent_stroke = st.selectbox("History of Stroke?", options=["No", "Yes"])
prevalent_hyp = st.selectbox("Hypertension?", options=["No", "Yes"])
diabetes = st.selectbox("Diabetes?", options=["No", "Yes"])
tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=300, value=120)
dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=200, value=80)
bmi = st.number_input("Body Mass Index (kg/m²)", min_value=10.0, max_value=60.0, value=25.0)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100)

# Map categorical inputs to numerical values
education_mapping = {"High School": 1, "Undergraduate": 2, "Graduate": 3, "Postgraduate": 4}
education = education_mapping[education]

sex = 1 if sex == "Male" else 0
is_smoking = 1 if is_smoking == "Yes" else 0
bp_meds = 1 if bp_meds == "Yes" else 0
prevalent_stroke = 1 if prevalent_stroke == "Yes" else 0
prevalent_hyp = 1 if prevalent_hyp == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'education': [education],
    'sex': [sex],
    'is_smoking': [is_smoking],
    'cigsPerDay': [cigs_per_day],
    'BPMeds': [bp_meds],
    'prevalentStroke': [prevalent_stroke],
    'prevalentHyp': [prevalent_hyp],
    'diabetes': [diabetes],
    'totChol': [tot_chol],
    'sysBP': [sys_bp],
    'diaBP': [dia_bp],
    'BMI': [bmi],
    'heartRate': [heart_rate],
    'glucose': [glucose]
})

# Predict and display results
if st.button("Predict"):
    prediction = xgb_model.predict(input_data)
    probability = xgb_model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"The model predicts a high risk of heart disease. (Probability: {probability:.2f})")
    else:
        st.success(f"The model predicts a low risk of heart disease. (Probability: {probability:.2f})")

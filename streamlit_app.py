import streamlit as st
import numpy as np
import joblib

models = joblib.load("models.pkl")

FEATURES = [
    'age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
    'diaBP', 'BMI', 'heartRate', 'glucose'
]

def select_best_model(input_data):
    best_model_name = None
    best_risk_score = -1
    best_model = None
    for model_name, model in models.items():
        scaled_data = scaler.transform([input_data])
        risk_score = model.predict_proba(scaled_data)[0][1]
        if risk_score > best_risk_score:
            best_risk_score = risk_score
            best_model_name = model_name
            best_model = model
    return best_model_name, best_model, best_risk_score

def generate_recommendations(risk_score):
    if risk_score < 0.3:
        return "Maintain a healthy lifestyle with regular exercise and a balanced diet."
    elif risk_score < 0.7:
        return "Consider regular check-ups and monitor your cholesterol and blood pressure."
    else:
        return "High risk detected! Consult a healthcare provider immediately for a detailed assessment."

st.title("Cardiovascular Disease Risk Predictor")
st.write("Input your health details to assess your risk and receive personalized recommendations.")

age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
education = st.selectbox("Education Level (1=Less, 4=Graduate)", options=[1, 2, 3, 4])
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
is_smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, value=0, step=1)
BPMeds = st.selectbox("Blood Pressure Medication", options=[0, 1])
prevalentStroke = st.selectbox("Prevalent Stroke", options=[0, 1])
prevalentHyp = st.selectbox("Prevalent Hypertension", options=[0, 1])
diabetes = st.selectbox("Diabetes", options=[0, 1])
totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=0, value=200, step=1)
sysBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0, value=120, step=1)
diaBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, value=80, step=1)
BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=25.0, step=0.1)
heartRate = st.number_input("Heart Rate (beats per minute)", min_value=0, value=70, step=1)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, value=100, step=1)

if st.button("Calculate Risk"):
    input_data = [
        age, education, sex, is_smoking, cigsPerDay, BPMeds, prevalentStroke,
        prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]
    best_model_name, best_model, risk_score = select_best_model(input_data)
    recommendation = generate_recommendations(risk_score)
    st.subheader("Best Model:")
    st.write(best_model_name)
    st.subheader("Risk Score:")
    st.write(f"{risk_score:.2f}")
    st.subheader("Personalized Recommendation:")
    st.write(recommendation)

st.write("Note: This prediction is not a substitute for professional medical advice.")

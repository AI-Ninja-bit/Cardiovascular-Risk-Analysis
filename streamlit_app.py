import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

xgb_model = joblib.load("xgb_model.pkl")
scaler = StandardScaler()

FEATURES = [
    'age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
    'diaBP', 'BMI', 'heartRate', 'glucose'
]

def calculate_risk_score(input_data):
    scaled_data = scaler.fit_transform([input_data])
    risk_score = xgb_model.predict_proba(scaled_data)[0][1]
    return risk_score

def generate_recommendations(risk_score):
    if risk_score < 0.3:
        return "Maintain a healthy lifestyle with regular exercise and a balanced diet."
    elif risk_score < 0.7:
        return "Consider regular check-ups and monitor your cholesterol and blood pressure."
    else:
        return "High risk detected! Consult a healthcare provider immediately for a detailed assessment."

st.set_page_config(page_title="CVD Risk Predictor", page_icon="❤️", layout="wide")
st.title("Cardiovascular Disease Risk Predictor")
st.subheader("Input your health details to assess your risk and receive personalized recommendations.")

st.image("https://www.example.com/logo.png", width=150)

col1, col2 = st.columns([2, 3])

with col1:
    age = st.slider("Age", min_value=0, max_value=120, value=30, step=1)
    education = st.selectbox("Education Level (1=Less, 4=Graduate)", options=[1, 2, 3, 4])
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    is_smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
    cigsPerDay = st.slider("Cigarettes Per Day", min_value=0, max_value=100, value=0, step=1)
    BPMeds = st.selectbox("Blood Pressure Medication", options=[0, 1])
    prevalentStroke = st.selectbox("Prevalent Stroke", options=[0, 1])
    prevalentHyp = st.selectbox("Prevalent Hypertension", options=[0, 1])

with col2:
    diabetes = st.selectbox("Diabetes", options=[0, 1])
    totChol = st.slider("Total Cholesterol (mg/dL)", min_value=0, max_value=400, value=200, step=1)
    sysBP = st.slider("Systolic Blood Pressure (mmHg)", min_value=50, max_value=200, value=120, step=1)
    diaBP = st.slider("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=150, value=80, step=1)
    BMI = st.slider("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    heartRate = st.slider("Heart Rate (beats per minute)", min_value=30, max_value=200, value=70, step=1)
    glucose = st.slider("Glucose Level (mg/dL)", min_value=50, max_value=250, value=100, step=1)

if st.button("Calculate Risk", key="calculate"):
    input_data = [
        age, education, sex, is_smoking, cigsPerDay, BPMeds, prevalentStroke,
        prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]
    risk_score = calculate_risk_score(input_data)
    recommendation = generate_recommendations(risk_score)
    
    st.subheader("Risk Score:")
    st.markdown(f"<h3 style='color:{'red' if risk_score > 0.7 else 'green' if risk_score < 0.3 else 'orange'};'>{risk_score:.2f}</h3>", unsafe_allow_html=True)
    
    st.progress(risk_score * 100)

    st.subheader("Personalized Recommendation:")
    st.write(recommendation)

    fig, ax = plt.subplots()
    sns.barplot(x=['Risk'], y=[risk_score], palette='coolwarm', ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

st.write("Note: This prediction is not a substitute for professional medical advice.")

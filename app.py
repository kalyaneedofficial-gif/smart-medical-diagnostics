"""
Module 3: Analytics & Reporting Dashboard
Streamlit interface for interacting with the ML pipeline.
"""
import streamlit as st
from data_processor import DataProcessor
from model_engine import DiagnosticEngine

st.set_page_config(page_title="Smart Medical Diagnostics", layout="wide")

st.title("Smart Medical Diagnostics & Disease Prediction System")
st.markdown("Enter patient clinical parameters below to generate diagnostic risk analysis.")

processor = DataProcessor()
engine = DiagnosticEngine()

# User Inputs
st.sidebar.header("Patient Clinical Parameters")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
glucose = st.sidebar.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=110)
bp = st.sidebar.number_input("Blood Pressure (mmHg)", min_value=40, max_value=200, value=80)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
insulin = st.sidebar.number_input("Insulin (mu U/ml)", min_value=0, max_value=800, value=85)

input_data = {
    'age': age,
    'glucose': glucose,
    'blood_pressure': bp,
    'bmi': bmi,
    'insulin': insulin
}

if st.button("Run Diagnostic Assessment"):
    try:
        scaled_features = processor.preprocess(input_data)
        probability, level = engine.predict_risk(scaled_features)

        st.subheader("Diagnostic Results")
        st.metric(label="Risk Assessment Level", value=level)
        st.write(f"**Predicted Disease Probability:** {probability * 100:.2f}%")
        
        st.info("Assessment generated successfully based on current patient metrics.")
    except Exception as e:
        st.error(f"Error executing diagnostic prediction: {str(e)}")

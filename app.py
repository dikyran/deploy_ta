
import streamlit as st
import pandas as pd
import pickle

# Load model dan scaler
with open('models/best_models.pkl', 'rb') as f:
    best_models = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Judul aplikasi
st.title("Prediksi Kelayakan Air Minum (Groundwater Quality)")

# Input user
ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
tds = st.number_input("TDS (Total Dissolved Solids)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, step=0.1)

model_name = st.selectbox("Pilih Model", list(best_models.keys()))

if st.button("Prediksi"):
    input_data = pd.DataFrame([[ph, tds, temperature, turbidity]], columns=['ph', 'tds', 'temperature', 'turbidity'])

    if model_name in ['SVM', 'LogisticRegression', 'ANN', 'KNN']:
        input_scaled = scaler.transform(input_data)
        prediction = best_models[model_name].predict(input_scaled)
    else:
        prediction = best_models[model_name].predict(input_data)

    st.subheader(f"Hasil Prediksi: {prediction[0]}")

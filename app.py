import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="ReAdmitShield", page_icon="🏥", layout="centered")

st.title("🏥 ReAdmitShield: 30-Day Readmission Prevention Dashboard")
st.markdown("This dashboard helps visualize patient readmission risks using simulated EHR data and a trained AI model.")

# --- Check required files ---
data_path = "data/simulated_ehr.csv"
model_path = "models/readmit_model.pkl"
encoder_path = "models/label_encoders.pkl"

if not os.path.exists(data_path) or not os.path.exists(model_path):
    st.error("❌ Missing files! Please make sure 'models/readmit_model.pkl' and 'data/simulated_ehr.csv' exist.")
    st.stop()

# --- Load data and model ---
df = pd.read_csv(data_path)
model = joblib.load(model_path)

# Some versions might not have encoders if you didn’t save them
encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

st.success("✅ Model and data loaded successfully!")

# --- Sidebar for patient input ---
st.sidebar.header("Enter Patient Details")

# Automatically use column names from dataset except 'readmitted'
input_data = {}
for col in df.columns:
    if col == 'readmitted':
        continue
    if df[col].dtype == 'object':
        options = list(df[col].unique())
        input_data[col] = st.sidebar.selectbox(col, options)
    else:
        val = float(df[col].mean())
        input_data[col] = st.sidebar.number_input(col, value=val)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Encode if needed
if encoders:
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

# --- Predict risk ---
if st.sidebar.button("Predict Readmission Risk"):
    prediction = model.predict_proba(input_df)[0][1]
    risk_percent = round(prediction * 100, 2)

    if risk_percent > 70:
        st.error(f"⚠️ High Readmission Risk: {risk_percent}%")
    elif risk_percent > 40:
        st.warning(f"🟠 Moderate Readmission Risk: {risk_percent}%")
    else:
        st.success(f"🟢 Low Readmission Risk: {risk_percent}%")

    st.markdown("---")
    st.write("### Patient Summary")
    st.write(input_df)
else:
    st.info("👈 Enter patient details and click **Predict Readmission Risk** to begin.")

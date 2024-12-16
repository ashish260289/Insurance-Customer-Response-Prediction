import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the trained Logistic Regression model
model = load('logistic_regression_model.joblib')

# Create a Streamlit app
st.title("Insurance Customer Response Prediction")

# Sidebar inputs
st.sidebar.header("Input Features")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100)
driving_license = st.sidebar.selectbox("Driving License", [0, 1])
region_code = st.sidebar.slider("Region Code", 0.0, 100.0, step=0.1)
previously_insured = st.sidebar.selectbox("Previously Insured", [0, 1])
vehicle_age = st.sidebar.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
vehicle_damage = st.sidebar.selectbox("Vehicle Damage", ["Yes", "No"])
annual_premium = st.sidebar.number_input("Annual Premium", 0.0, 100000.0, step=500.0)
policy_sales_channel = st.sidebar.slider("Policy Sales Channel", 0.0, 200.0, step=1.0)
vintage = st.sidebar.slider("Vintage", 0, 300)

# Encode inputs as required by the model
vehicle_age_map = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
vehicle_damage_map = {"No": 0, "Yes": 1}
gender_map = {"Male": 1, "Female": 0}

input_features = np.array([
    gender_map[gender],
    age,
    driving_license,
    region_code,
    previously_insured,
    vehicle_age_map[vehicle_age],
    vehicle_damage_map[vehicle_damage],
    annual_premium,
    policy_sales_channel,
    vintage
]).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        st.success("The customer is likely to respond positively!")
    else:
        st.error("The customer is unlikely to respond.")


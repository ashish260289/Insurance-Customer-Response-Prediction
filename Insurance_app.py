import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load the trained Logistic Regression model
model = load('logistic_regression_model.joblib')
# Assuming you saved the preprocessor as well
preprocessor = load('preprocessor.joblib')

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

# Prepare input features
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

# Column names as used during training
input_columns = [
    'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 
    'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
]

# Convert input_features to a DataFrame with the correct column names
input_df = pd.DataFrame(input_features, columns=input_columns)

# Debugging step: Print the input dataframe to check the structure
st.write("Input DataFrame for Prediction:")
st.write(input_df)

# Apply the preprocessor to transform the input features (ensure preprocessing consistency)
input_features_processed = preprocessor.transform(input_df)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_features_processed)
    if prediction[0] == 1:
        st.error("The customer is unlikely to respond.")
    else:
        st.success("The customer is likely to respond.")

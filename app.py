import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Title
st.title("📉 Customer Churn Prediction App")
st.markdown("Fill in the customer details below to predict churn.")

# Input form
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Convert inputs to model format (7 features)
input_data = []

# 1. tenure, 2. monthly_charges, 3. total_charges
input_data.extend([tenure, monthly_charges, total_charges])

# 4. InternetService_Fiber optic
input_data.append(1 if internet_service == "Fiber optic" else 0)

# 5. Contract_One year
input_data.append(1 if contract == "One year" else 0)

# 6. Contract_Two year
input_data.append(1 if contract == "Two year" else 0)

# 7. PaperlessBilling_Yes
input_data.append(1 if paperless_billing == "Yes" else 0)

# Convert to numpy and scale
input_array = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1] * 100  # churn % prob

    if prediction[0] == 1:
        st.error(f"❌ The customer is **likely to churn**.\n📊 Churn Probability: **{probability:.2f}%**")
    else:
        st.success(f"✅ The customer is **likely to stay**.\n📊 Churn Probability: **{probability:.2f}%**")

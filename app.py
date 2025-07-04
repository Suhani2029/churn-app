import streamlit as st
import numpy as np
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.markdown("Fill in the customer details below to predict churn.")

tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

input_data = []

input_data.extend([tenure, monthly_charges, total_charges])

input_data.append(1 if internet_service == "Fiber optic" else 0)

input_data.append(1 if contract == "One year" else 0)

input_data.append(1 if contract == "Two year" else 0)

input_data.append(1 if paperless_billing == "Yes" else 0)

input_array = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1] * 100  # churn % prob

    if prediction[0] == 1:
        st.error(f"❌ The customer is **likely to churn**.\n📊 Churn Probability: **{probability:.2f}%**")
    else:
        st.success(f"✅ The customer is **likely to stay**.\n📊 Churn Probability: **{probability:.2f}%**")


st.markdown("---")
st.header("📊 Model Comparison: Logistic Regression vs Random Forest")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

y_pred_log = model.predict(X_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }

log_metrics = get_metrics(y_test, y_pred_log)
rf_metrics = get_metrics(y_test, y_pred_rf)

import pandas as pd
comparison_df = pd.DataFrame([log_metrics, rf_metrics], index=['Logistic Regression', 'Random Forest'])
st.subheader("📋 Metrics Table")
st.dataframe(comparison_df.style.format("{:.2%}"))

st.subheader("📈 Visual Comparison")
st.bar_chart(comparison_df.T)

import pandas as pd
import matplotlib.pyplot as plt

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

logistic_preds = model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_preds)

rf_model = joblib.load("rf_model.pkl")
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

st.markdown("## 🔍 Model Comparison")

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [round(logistic_accuracy * 100, 2), round(rf_accuracy * 100, 2)]
})

st.table(comparison_df)

fig, ax = plt.subplots()
ax.bar(comparison_df['Model'], comparison_df['Accuracy'], color=['skyblue', 'orange'])
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
st.pyplot(fig)

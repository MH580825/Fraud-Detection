import streamlit as st
import numpy as np
import pickle

# Load trained model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to check if it's fraudulent.")

# User input for 29 features (V1 to V28 + Amount)
features = []
for i in range(1, 29):  # Features V1 to V28
    features.append(st.number_input(f"Feature V{i}", value=0.0))

# Amount input
amount = st.number_input("Transaction Amount ($)", value=0.0)
features.append(amount)  # Append Amount as the 29th feature

# Predict Button
if st.button("Predict Fraud"):
    # Convert input to NumPy array
    data = np.array(features).reshape(1, -1)

    # Apply the updated StandardScaler
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)[0]

    # Display result
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe.")

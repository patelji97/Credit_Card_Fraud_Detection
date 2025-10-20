import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("This app predicts whether a credit card transaction is **Fraudulent (1)** or **Legitimate (0)** using a trained Random Forest model.")

# Sidebar for user input
st.sidebar.header("Enter Transaction Details")

# The dataset has many features (V1–V28, Amount, Time)
# For simplicity, we'll take a few key features as input manually
v1 = st.sidebar.number_input("V1", value=0.0)
v2 = st.sidebar.number_input("V2", value=0.0)
v3 = st.sidebar.number_input("V3", value=0.0)
v4 = st.sidebar.number_input("V4", value=0.0)
v5 = st.sidebar.number_input("V5", value=0.0)
amount = st.sidebar.number_input("Transaction Amount", value=100.0)

# Collect inputs into a DataFrame (assuming other features as zero)
def create_input():
    data = np.zeros((1, 30))  # 30 columns: Time, V1–V28, Amount
    data[0, 1] = v1
    data[0, 2] = v2
    data[0, 3] = v3
    data[0, 4] = v4
    data[0, 5] = v5
    data[0, 29] = amount
    return pd.DataFrame(data, columns=[f"V{i}" if i != 0 and i != 29 else ("Time" if i == 0 else "Amount") for i in range(30)])

input_df = create_input()

st.subheader("🔍 Transaction Input Data:")
st.dataframe(input_df)

# Predict button
if st.button("Predict Fraud Status"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ This transaction is **Fraudulent!**")
    else:
        st.success("✅ This transaction is **Legitimate.**")

st.write("---")
st.caption("Developed with ❤️ using Streamlit and Machine Learning")




        
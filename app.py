import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("📊 Investment Profit Prediction App")

digital = st.number_input("Digital Marketing Spend")
promotion = st.number_input("Promotion Spend")
research = st.number_input("Research Spend")
state = st.selectbox("State", ["New York", "California", "Florida"])

input_data = pd.DataFrame({
    "DigitalMarketing": [digital],
    "Promotion": [promotion],
    "Research": [research],
    "State": [state]
})

input_data = pd.get_dummies(input_data, drop_first=True)

prediction = model.predict(input_data)

if st.button("Predict Profit"):
    st.success(f"Estimated Profit: {prediction[0]:.2f}")

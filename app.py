import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Investment Profit Prediction")

st.title("📊 Investment Profit Prediction App")

# Load or Train Model
if not os.path.exists("model.pkl"):
    dataset = pd.read_csv("data/Investment.csv")
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X = pd.get_dummies(X, drop_first=True)

    model = LinearRegression()
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))
else:
    model = pickle.load(open("model.pkl", "rb"))

# ----------------------------
# User Inputs
# ----------------------------

digital = st.number_input("Digital Marketing Spend")
promotion = st.number_input("Promotion Spend")
research = st.number_input("Research Spend")
state = st.selectbox("State", ["New York", "California", "Florida"])

# Prepare input dataframe
input_data = pd.DataFrame({
    "DigitalMarketing": [digital],
    "Promotion": [promotion],
    "Research": [research],
    "State": [state]
})

input_data = pd.get_dummies(input_data, drop_first=True)

# Make sure all columns match training data
training_data = pd.read_csv("data/Investment.csv")
X_train_cols = pd.get_dummies(training_data.iloc[:, :-1], drop_first=True).columns

for col in X_train_cols:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X_train_cols]

# Prediction
if st.button("Predict Profit"):
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Profit: {prediction[0]:,.2f}")

import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression

# Check if model exists
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

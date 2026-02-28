import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Investment Profit Prediction", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Investment Profit Prediction Dashboard")
st.markdown("### 🚀 Multiple Linear Regression | ML Deployment Project")

# -----------------------------
# Load or Train Model
# -----------------------------
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

# Reload dataset for metrics
dataset = pd.read_csv("data/Investment.csv")
X_full = pd.get_dummies(dataset.iloc[:, :-1], drop_first=True)
y_full = dataset.iloc[:, -1]

r2 = r2_score(y_full, model.predict(X_full))

# -----------------------------
# Metrics Section
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📈 R² Score", f"{r2:.4f}")
col2.metric("📊 Dataset Size", len(dataset))
col3.metric("🔢 Features Used", X_full.shape[1])

st.divider()

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("🔎 Enter Investment Details")

col1, col2 = st.columns(2)

with col1:
    digital = st.number_input("Digital Marketing Spend", min_value=0.0)
    promotion = st.number_input("Promotion Spend", min_value=0.0)

with col2:
    research = st.number_input("Research Spend", min_value=0.0)
    state = st.selectbox("State", ["New York", "California", "Florida"])

# Prepare input
input_data = pd.DataFrame({
    "DigitalMarketing": [digital],
    "Promotion": [promotion],
    "Research": [research],
    "State": [state]
})

input_data = pd.get_dummies(input_data, drop_first=True)

# Match training columns
for col in X_full.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X_full.columns]

# -----------------------------
# Prediction & History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("💰 Predict Profit"):
    prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Profit: ₹ {prediction:,.2f}")

    # Save to history
    st.session_state.history.append({
        "DigitalMarketing": digital,
        "Promotion": promotion,
        "Research": research,
        "State": state,
        "Predicted Profit": round(prediction, 2)
    })

# -----------------------------
# Prediction History Table
# -----------------------------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

st.divider()

st.markdown("""
### 🛠 Tech Stack Used:
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Multiple Linear Regression
""")

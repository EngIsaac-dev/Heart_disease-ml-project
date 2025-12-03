import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# -------------------
# Load models and scaler
# -------------------
log_reg_model = joblib.load("log_reg_model.pkl")
rf_model = joblib.load("rf_clf.pkl")
scaler = joblib.load("scaler.pkl")
nn_model = tf.keras.models.load_model("nn_model.h5")

# -------------------
# App Title
# -------------------
st.title("💓 Heart Disease Prediction App")

st.write("""
### Enter patient information below:
Choose the model and fill in the patient details to predict heart disease.
""")

# -------------------
# Input Form
# -------------------
age = st.number_input("Age", min_value=10, max_value=120, value=50)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 130)
chol = st.number_input("Cholesterol Level", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1–3)", [1, 2, 3])

# -------------------
# Model Selection
# -------------------
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Random Forest", "Neural Network"]
)

# -------------------
# Prepare input
# -------------------
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale input for models that need it
scaled_data = scaler.transform(input_data)

# -------------------
# Prediction
# -------------------
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = log_reg_model.predict(scaled_data)[0]

    elif model_choice == "Random Forest":
        prediction = rf_model.predict(scaled_data)[0]

    else:  # Neural Network
        nn_pred = nn_model.predict(scaled_data)
        prediction = 1 if nn_pred[0][0] >= 0.5 else 0

    # -------------------
    # Show Result
    # -------------------
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("💔 The patient is **likely to have heart disease**.")
    else:
        st.success("💚 The patient is **unlikely to have heart disease**.")

"""
streamlit_app.py  (OPTIONAL)
-----------------------------
Run with:   streamlit run streamlit_app.py
Install:    pip install streamlit scikit-learn pandas numpy
"""

import streamlit as st
import numpy as np
import pickle, os

# ── Page config ──
st.set_page_config(page_title="DiabetesIQ", page_icon="◎", layout="centered")

st.title("◎ DiabetesIQ — Diabetes Risk Predictor")
st.caption("PIMA Indians Diabetes Dataset · Logistic Regression & Random Forest")

# ── Load model ──
@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
scaler, lr_model, rf_model = bundle["scaler"], bundle["lr_model"], bundle["rf_model"]

st.markdown(
    f"Model accuracies — **LR: {bundle['lr_accuracy']}%** | **RF: {bundle['rf_accuracy']}%**"
)
st.divider()

# ── Inputs ──
st.subheader("Enter Health Data")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies",              0, 20,  2)
    glucose     = st.number_input("Glucose (mg/dL)",          0, 300, 120)
    bp          = st.number_input("Blood Pressure (mmHg)",    0, 200, 72)
    skin        = st.number_input("Skin Thickness (mm)",       0, 100, 23)

with col2:
    insulin     = st.number_input("Insulin (μU/mL)",          0, 900, 85)
    bmi         = st.number_input("BMI (kg/m²)",              0.0, 70.0, 28.5)
    dpf         = st.number_input("Diabetes Pedigree Function",0.0, 3.0, 0.5, format="%.3f")
    age         = st.number_input("Age (years)",               1, 120, 35)

model_choice = st.radio("Model", ["Logistic Regression", "Random Forest"], horizontal=True)

# ── Predict ──
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    if model_choice == "Random Forest":
        pred = rf_model.predict(features)[0]
        prob = rf_model.predict_proba(features)[0][1]
    else:
        pred = lr_model.predict(scaler.transform(features))[0]
        prob = lr_model.predict_proba(scaler.transform(features))[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ **High Risk Detected** — Probability: {prob*100:.1f}%")
        st.progress(prob)
        st.markdown("**Health Suggestions:**")
        for tip in [
            "🥗 Adopt a low-GI, high-fibre diet.",
            "🏃 Exercise 30 min/day, 5 days/week.",
            "⚖️ Lose 5–7% body weight if overweight.",
            "🩺 Consult your doctor soon.",
            "💧 Stay hydrated; cut sugary drinks.",
        ]:
            st.markdown(f"- {tip}")
    else:
        st.success(f"✅ **Low Risk Detected** — Probability: {prob*100:.1f}%")
        st.progress(prob)
        st.markdown("**Keep it up!**")
        for tip in [
            "✅ Maintain your healthy habits.",
            "🥦 Eat balanced, colourful meals.",
            "📅 Get annual blood glucose checks.",
            "😴 Prioritise quality sleep.",
        ]:
            st.markdown(f"- {tip}")

st.divider()
st.caption("⚕️ For educational purposes only. Not a substitute for medical advice.")

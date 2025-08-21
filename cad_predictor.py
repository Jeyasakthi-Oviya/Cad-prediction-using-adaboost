import streamlit as st
import numpy as np
import pickle

# Load saved models & scalers
with open('quantile_transformer.pkl', 'rb') as f:
    QT = pickle.load(f)

with open('adaboost.pkl', 'rb') as f:
    ada_clf = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Coronary Artery Disease Prediction")
st.markdown("### Enter Patient Details Below:")

# Risk Level Function
def risk_level(value, normal, moderate, high):
    if value is None:
        return "ℹ️ **Risk Levels:** Normal (≤{}) | Moderate ({}-{}) | High (>{})".format(normal, normal+1, moderate, high)
    elif value <= normal:
        return "✅ **Normal**"
    elif value <= moderate:
        return "⚠️ **Moderate Risk**"
    else:
        return "🚨 **High Risk**"

# User Inputs
st.markdown("### **Basic Information**")
typical_chest_pain = st.selectbox("Typical Chest Pain", [None, 0, 1], index=0)  
age = st.number_input("Age", min_value=20, max_value=100, step=1, value=None, format="%d")
st.markdown(f"**{risk_level(age, 40, 60, 75)}**")

htn = st.selectbox("Hypertension (HTN)", [None, 0, 1], index=0)  
dm = st.selectbox("Diabetes Mellitus (DM)", [None, 0, 1], index=0)  

st.markdown("### **Medical Measurements**")
bp = st.number_input("Blood Pressure (BP)", min_value=50, max_value=200, step=1, value=None, format="%d")
st.markdown(f"**{risk_level(bp, 120, 140, 180)}**")

tinversion = st.selectbox("T Inversion", [None, 0, 1], index=0)  

fbs = st.number_input("Fasting Blood Sugar (FBS)", min_value=50, max_value=300, step=5, value=None, format="%d")
st.markdown(f"**{risk_level(fbs, 100, 126, 200)}**")

tg = st.number_input("Triglycerides (TG)", min_value=50, max_value=1000, step=10, value=None, format="%d")
st.markdown(f"**{risk_level(tg, 150, 200, 500)}**")

atypical = st.selectbox("Atypical Angina", [None, 0, 1], index=0)  
nonanginal = st.selectbox("Nonanginal Pain", [None, 0, 1], index=0)  

ef_tte = st.number_input("Ejection Fraction (EF-TTE)", min_value=10, max_value=80, step=1, value=None, format="%d")
st.markdown(f"**{risk_level(ef_tte, 50, 40, 30)}**")

# Prediction Button
if st.button("Predict"):
    input_values = [typical_chest_pain, age, htn, dm, bp, tinversion, fbs, tg, atypical, nonanginal, ef_tte]
    if None in input_values:
        st.markdown('<p style="color:red; font-size:20px;">⚠️ Please enter all details before predicting.</p>', unsafe_allow_html=True)
    else:
        tg_transformed = QT.transform(np.array([[tg]]))[0, 0]
        input_data = np.array([[typical_chest_pain, age, htn, dm, bp, tinversion, fbs, tg_transformed, atypical, nonanginal, ef_tte]])
        input_scaled = scaler.transform(input_data)
        
        prediction_prob = ada_clf.predict_proba(input_scaled)[0][1] * 100  # Get probability of CAD
        prediction = ada_clf.predict(input_scaled)[0]
        
        if prediction == 1:
            st.markdown(f'<p style="color:red; font-size:20px;">⚠️ Coronary Artery Disease (CAD) Detected with {prediction_prob:.2f}% confidence.</p>', unsafe_allow_html=True)
            st.markdown("### Suggested Tests for Further Diagnosis:")
            st.markdown("- 🩺 **Electrocardiogram (ECG)**")
            st.markdown("- 🏃 **Stress Test**")
            st.markdown("- 🩸 **Blood Test**")
            st.markdown("- 🏥 **Echocardiogram**")
        else:
            st.markdown(f'<p style="color:green; font-size:20px;">✅ No CAD - Patient is Normal ({100 - prediction_prob:.2f}% confidence).</p>', unsafe_allow_html=True)
            st.markdown("### Health Tips for Better Heart Health:")
            st.markdown("- 🍎 **Eat a Heart-Healthy Diet** (low in saturated fats, high in fiber)")
            st.markdown("- 🏃 **Exercise Regularly** (30 minutes of moderate activity daily)")
            st.markdown("- 😌 **Manage Stress** (yoga, meditation, or breathing exercises)")

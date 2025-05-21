import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open('cluster_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Customer Segmentation Predictor")
st.markdown("Input behavioral metrics to predict customer segment.")

# Input sliders
rec = st.slider("Recency (days since last purchase)", 0, 365, 90)
freq = st.slider("Frequency (total transactions)", 0, 10, 2)
mon = st.slider("Monetary (total spend, log-scaled)", 0, 10, 5)
ipt = st.slider("Interpurchase Time (avg. gap days)", 0.0, 30.0, 5.0)
imp = st.slider("Impulse Score (0 = deliberate, 1 = impulsive)", 0.0, 1.0, 0.5)

# Prediction section
if st.button("Predict Segment"):
    input_df = pd.DataFrame([[rec, freq, mon, ipt, imp]],
                            columns=['Recency', 'Frequency', 'Monetary', 'InterpurchaseTime', 'ImpulseScore'])

    # Show raw input for debugging
    st.markdown("### 🔍 Raw Input")
    st.dataframe(input_df)

    input_scaled = scaler.transform(input_df)

    # Show scaled input for debugging
    st.markdown("### ⚙️ Scaled Input")
    st.dataframe(pd.DataFrame(input_scaled, columns=input_df.columns))

    pred = model.predict(input_scaled)[0]

    # UX suggestion map
    ux_map = {
        0: "Flash deals, one-click buying",
        1: "Smart nudges, reminders",
        2: "Loyalty perks, personalization",
        3: "Low-key reactivation",
        4: "Comeback offers, FOMO banners"
    }

    st.markdown("### 🧠 Predicted Segment")
    st.success(f"**Cluster {pred} — {ux_map[pred]}**")

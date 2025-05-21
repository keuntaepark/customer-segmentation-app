import streamlit as st
import pandas as pd
import numpy as np
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

if st.button("Predict Segment"):
    # Assemble input
    input_df = pd.DataFrame([[rec, freq, mon, ipt, imp]],
                            columns=['Recency', 'Frequency', 'Monetary', 'InterpurchaseTime', 'ImpulseScore'])

    st.markdown("### 🔍 Raw Input")
    st.dataframe(input_df)

    # Apply log1p to the 3 relevant features (used during training)
    log_cols = ['Frequency', 'Monetary', 'InterpurchaseTime']
    input_df[log_cols] = np.log1p(input_df[log_cols])

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    st.markdown("### ⚙️ Scaled Input")
    st.dataframe(pd.DataFrame(input_scaled, columns=input_df.columns))

    # Prediction
    pred = model.predict(input_scaled)[0]

    ux_map = {
        0: "Flash deals, one-click buying",
        1: "Smart nudges, reminders",
        2: "Loyalty perks, personalization",
        3: "Low-key reactivation",
        4: "Comeback offers, FOMO banners"
    }

    st.markdown("### 🧠 Predicted Segment")
    st.success(f"**Cluster {pred} — {ux_map[pred]}**")

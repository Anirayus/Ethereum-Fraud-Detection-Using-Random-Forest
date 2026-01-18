
import streamlit as st
import pandas as pd
import joblib


# Load model & scaler

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()


# Get feature list

if not hasattr(scaler, "feature_names_in_"):
    st.error("⚠️ scaler.pkl has no 'feature_names_in_'. Please retrain the scaler.")
    st.stop()

feature_cols = list(scaler.feature_names_in_)

def is_flag(col):
    return str(col).strip().lower() == "flag"

# All features except Flag
input_features = [c for c in feature_cols if not is_flag(c)]
if not input_features:
    st.error("No input features found.")
    st.stop()


# Interface

st.title("⟠ Cryptocurrency Fraud Detection (Random Forest)")
st.subheader("Key in ALL Ethereum transaction features")



# Input form (text input → float)

with st.form("all_features_form"):
    user_inputs = {}

    cols = st.columns(3)
    for i, feat in enumerate(input_features):
        col = cols[i % 3]

        raw_value = col.text_input(
            label=str(feat),
            value="0" 
        )

        try:
            user_inputs[feat] = float(raw_value)
        except ValueError:
            st.error(f"❌ Invalid numeric value for: {feat}")
            st.stop()

    submitted = st.form_submit_button("🔍 Check")


# Prediction

if submitted:
    X = pd.DataFrame([user_inputs])

    # Ensure correct order
    X = X[input_features]

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error("Scaling failed: feature mismatch with trained scaler.")
        st.exception(e)
        st.stop()

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0]
        p_legit = float(prob[0])
        p_fraud = float(prob[1])
    else:
        pred = int(model.predict(X_scaled)[0])
        p_fraud = 1.0 if pred == 1 else 0.0
        p_legit = 1.0 - p_fraud

    label = "FRAUD" if p_fraud >= 0.5 else "LEGIT"

    st.markdown("---")
    st.subheader("Prediction Result")

    if label == "FRAUD":
        st.error(f"🔴 Prediction: **{label}**")
    else:
        st.success(f"🟢 Prediction: **{label}**")

    st.write(f"Probability → LEGIT: **{p_legit:.3f}**, FRAUD: **{p_fraud:.3f}**")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ✅ Load trained models and scaler
best_rf_model = joblib.load("rf_model.pkl")
best_xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

# --- Preprocessing Function ---
def preprocess_input(df, scaler):
    df = df.copy()

    if 'Class' in df.columns:
        df.drop(columns='Class', inplace=True)

    expected_base_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing_cols = [col for col in expected_base_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']].values)

    time_unscaled = df['Time'] * scaler.scale_[0] + scaler.mean_[0]
    df['Hour'] = (time_unscaled // 3600).astype(int)

    df = df[model_features]
    return df

# --- Sequential Prediction Function ---
def sequential_predict(input_df, rf_model, xgb_model, scaler):
    try:
        processed = preprocess_input(input_df, scaler)

        rf_scores = rf_model.predict_proba(processed)[:, 1]
        xgb_scores = xgb_model.predict_proba(processed)[:, 1]

        decisions = []
        for rf_score, xgb_score in zip(rf_scores, xgb_scores):
            if rf_score >= 0.83:
                if xgb_score >= 0.99:
                    decisions.append("Auto-Block")
                else:
                    decisions.append("Flag for Review")
            else:
                decisions.append("Allow")

        result = processed.copy()
        result["RF_Fraud_Prob"] = rf_scores
        result["XGB_Fraud_Prob"] = xgb_scores
        result["Decision"] = decisions
        return result

    except Exception as e:
        st.error(f"🚨 Internal error: {e}")
        return pd.DataFrame()

# --- Streamlit App ---
st.set_page_config(page_title="Shika Fraud Detection App", layout="centered")
st.title("🚨 Shika Fraud Detection System")

st.markdown("Upload a transaction CSV with the following columns: **Time, V1–V28, Amount**")
st.markdown("💡 **Need a sample file to test?** [Click here to download one]()")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)

        required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        if not all(col in input_df.columns for col in required_cols):
            st.error("❌ The uploaded CSV is missing required columns.")
        else:
            result_df = sequential_predict(input_df, best_rf_model, best_xgb_model, scaler)

            if not result_df.empty:
                st.success("✅ Predictions complete.")

                # Show full results
                st.markdown("### 🔍 All Transactions")
                st.dataframe(result_df[["RF_Fraud_Prob", "XGB_Fraud_Prob", "Decision"]], use_container_width=True)

                # Show only suspicious transactions
                suspicious_df = result_df[result_df["Decision"].isin(["Auto-Block", "Flag for Review"])]
                if not suspicious_df.empty:
                    st.markdown("### 🚨 Suspected Fraud Transactions")
                    st.dataframe(suspicious_df[["RF_Fraud_Prob", "XGB_Fraud_Prob", "Decision"]], use_container_width=True)
                else:
                    st.info("✅ No suspicious transactions found in this file.")

                # Download button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download full predictions as CSV",
                    data=csv,
                    file_name='fraud_predictions.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

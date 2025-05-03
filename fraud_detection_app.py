import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ✅ Load trained models and scaler
best_rf_model = joblib.load("rf_model.pkl")
best_xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")  # ✅ Load column order

# --- Preprocessing Function ---
def preprocess_input(df, scaler):
    df = df.copy()

    # Drop target column if present
    if 'Class' in df.columns:
        df.drop(columns='Class', inplace=True)

    # Step 1: Check column presence
    expected_base_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing_cols = [col for col in expected_base_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Step 2: Apply scaler and print intermediate state
    try:
        df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    except Exception as e:
        st.error(f"🚨 Scaling error: {e}")
        st.write("Columns present during scaling:", df.columns.tolist())
        raise

    # Step 3: Create 'Hour' from unscaled Time
    try:
        time_unscaled = df['Time'] * scaler.scale_[0] + scaler.mean_[0]
        df['Hour'] = (time_unscaled // 3600).astype(int)
    except Exception as e:
        st.error(f"🚨 Hour generation error: {e}")
        raise

    # Step 4: Enforce exact column order
    try:
        df = df[model_features]
    except Exception as e:
        st.error(f"🚨 Column ordering error: {e}")
        st.write("Available columns:", df.columns.tolist())
        st.write("Expected columns:", model_features)
        raise
    return df

# --- Sequential Prediction Function ---
def sequential_predict(input_df, rf_model, xgb_model, scaler):
    try:
        processed = preprocess_input(input_df, scaler)

        # 🔍 Always print the actual columns the model is receiving
        st.write("📊 Processed DataFrame columns (before prediction):")
        st.write(processed.columns.tolist())

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
        # 🔥 Show what went wrong during prediction
        st.error(f"🚨 Internal prediction error: {e}")
        st.write("✅ Final processed input shape:")
        st.write(processed.shape)
        st.write("📊 Final column names:")
        st.write(processed.columns.tolist())
        return pd.DataFrame()


# --- Streamlit App ---
st.title("🚨 Fraud Detection System")
st.markdown("Upload a transaction CSV with the following columns: **Time, V1-V28, Amount**")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)

        required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        if not all(col in input_df.columns for col in required_cols):
            st.error("❌ The uploaded CSV is missing required columns.")
        else:
            # 🔍 Debug: Show expected column structure
            st.write("✅ Model expects features in this order:")
            st.write(model_features)

            result_df = sequential_predict(input_df, best_rf_model, best_xgb_model, scaler)

            st.success("✅ Predictions complete.")
            st.dataframe(result_df[["RF_Fraud_Prob", "XGB_Fraud_Prob", "Decision"]])

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

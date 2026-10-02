"""
Customer Churn Predictor + SHAP + AI Explanation
-------------------------------------------------
Run with:
    streamlit run app.py

Architecture:

Customer Input
      ↓
XGBoost Model
      ↓
SHAP
      ↓
Explanation Engine
      ↓
Gemini LLM (only called when user clicks "Generate AI Explanation")
      ↓
Business-friendly explanation
"""

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from explanation_engine import explain_customer
from llm_explainer import generate_llm_explanation


# ======================================================================
# PAGE CONFIGURATION
# ======================================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------------------------
# Minimal, calm visual styling. No new colors/components — just tighter
# spacing, quieter borders, and consistent type so the page reads as one
# coherent surface instead of a stack of default Streamlit widgets.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }
        h1 { font-weight: 700; letter-spacing: -0.02em; }
        h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
        [data-testid="stMetric"] {
            background: rgba(127,127,127,0.06);
            border: 1px solid rgba(127,127,127,0.15);
            border-radius: 10px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.7; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
        }
        .factor-line {
            padding: 4px 0;
            font-size: 0.95rem;
        }
        hr { margin: 1.6rem 0; opacity: 0.15; }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================================
# LOAD ARTIFACTS
# ======================================================================

@st.cache_resource
def load_artifacts():

    preprocessor = joblib.load("preprocessor.pkl")

    xgb_model = XGBClassifier()
    xgb_model.load_model("xgb_model.json")

    feature_names = joblib.load("shap_feature_names.pkl")
    threshold = joblib.load("business_threshold.pkl")
    categorical_features = joblib.load("categorical_features.pkl")
    numerical_features = joblib.load("numerical_features.pkl")

    explainer = shap.TreeExplainer(xgb_model)

    return (
        preprocessor,
        xgb_model,
        explainer,
        feature_names,
        threshold,
        categorical_features,
        numerical_features
    )


(
    preprocessor,
    xgb_model,
    explainer,
    feature_names,
    threshold,
    categorical_features,
    numerical_features
) = load_artifacts()


# ======================================================================
# CATEGORICAL OPTIONS
# ======================================================================

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
}


# ======================================================================
# SHAP HELPER
# ======================================================================

def aggregate_shap_to_original_features(
    shap_row,
    feature_names,
    categorical_features,
    numerical_features
):
    mapping = {}

    for col in numerical_features:
        mapping[f"numerical__{col}"] = col

    for col in categorical_features:
        prefix = f"categorical__{col}_"
        for fname in feature_names:
            if fname.startswith(prefix):
                mapping[fname] = col

    contrib = pd.Series(shap_row, index=feature_names)

    agg = {}
    for original in set(mapping.values()):
        cols = [f for f, o in mapping.items() if o == original]
        agg[original] = contrib[cols].sum()

    result = pd.Series(agg).sort_values(key=abs, ascending=False)
    return result


# ======================================================================
# UI — HEADER
# ======================================================================

st.title("📊 Customer Churn Predictor")
st.caption(
    "Predict customer churn risk and understand the prediction using SHAP."
)


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:

    st.header("Customer Details")

    input_data = {}

    with st.expander("Account", expanded=True):
        input_data["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
        input_data["Contract"] = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
        input_data["PaperlessBilling"] = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"])
        input_data["PaymentMethod"] = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])
        input_data["MonthlyCharges"] = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
        input_data["TotalCharges"] = st.number_input(
            "Total Charges ($)",
            0.0,
            10000.0,
            float(input_data["MonthlyCharges"]) * max(input_data["tenure"], 1),
            step=10.0
        )

    with st.expander("Demographics", expanded=False):
        input_data["gender"] = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
        input_data["SeniorCitizen"] = 1 if st.checkbox("Senior Citizen") else 0
        input_data["Partner"] = st.selectbox("Has Partner", CATEGORICAL_OPTIONS["Partner"])
        input_data["Dependents"] = st.selectbox("Has Dependents", CATEGORICAL_OPTIONS["Dependents"])

    with st.expander("Services", expanded=False):
        input_data["PhoneService"] = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"])
        input_data["MultipleLines"] = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])
        input_data["InternetService"] = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
        input_data["OnlineSecurity"] = st.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])
        input_data["OnlineBackup"] = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])
        input_data["DeviceProtection"] = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])
        input_data["TechSupport"] = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
        input_data["StreamingTV"] = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
        input_data["StreamingMovies"] = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])

    predict_clicked = st.button("Predict Churn", type="primary", use_container_width=True)


# ======================================================================
# SESSION STATE
# ======================================================================
# Streamlit reruns the whole script on every widget interaction (including
# clicking the AI explanation button below). Without storing results in
# session_state, a fresh "Predict Churn" click would be required just to
# re-show the SHAP output — and the AI explanation would silently regenerate
# (and re-call the LLM) on every unrelated rerun. Storing the computed
# result means a rerun only re-renders what's already there.

if "result" not in st.session_state:
    st.session_state.result = None
if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None


# ======================================================================
# RUN PREDICTION (only on button click)
# ======================================================================

if predict_clicked:

    ordered_cols = categorical_features + numerical_features
    row = {col: input_data[col] for col in ordered_cols}
    customer_df = pd.DataFrame([row])

    customer_transformed = preprocessor.transform(customer_df)
    proba = xgb_model.predict_proba(customer_transformed)[0, 1]
    prediction = int(proba >= threshold)

    shap_values = explainer.shap_values(customer_transformed)
    shap_row = shap_values[0]

    explanation = explain_customer(input_data)

    st.session_state.result = {
        "proba": proba,
        "prediction": prediction,
        "customer_transformed": customer_transformed,
        "shap_row": shap_row,
        "explanation": explanation,
    }
    # New prediction invalidates any previously generated AI explanation.
    st.session_state.ai_explanation = None


if st.session_state.result is None:
    st.info("Fill in the customer details in the sidebar and click **Predict Churn**.")
    st.stop()

result = st.session_state.result
proba = result["proba"]
prediction = result["prediction"]
customer_transformed = result["customer_transformed"]
shap_row = result["shap_row"]
explanation = result["explanation"]


# ======================================================================
# RESULTS
# ======================================================================

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Churn Probability", f"{proba:.1%}")
    st.metric("Decision Threshold", f"{threshold:.0%}")

    if prediction == 1:
        st.error("⚠️ Predicted: **Likely to Churn**")
    else:
        st.success("✅ Predicted: **Likely to Stay**")

with col2:
    st.subheader("Why? — SHAP Waterfall")

    shap_explanation = shap.Explanation(
        values=shap_row,
        base_values=explainer.expected_value,
        data=customer_transformed[0],
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

st.divider()

# ======================================================================
# GROUPED SHAP FACTORS
# ======================================================================

st.subheader("Top Factors")

agg = aggregate_shap_to_original_features(
    shap_row, feature_names, categorical_features, numerical_features
)

top = agg.head(8).reset_index()
top.columns = ["Feature", "SHAP Contribution"]
top["Effect"] = np.where(
    top["SHAP Contribution"] > 0, "↑ Increases churn risk", "↓ Decreases churn risk"
)

st.dataframe(
    top,
    use_container_width=True,
    hide_index=True,
    column_config={
        "SHAP Contribution": st.column_config.NumberColumn(format="%.3f")
    }
)

st.caption(
    "SHAP values show how much each feature pushed this customer's "
    "prediction above or below the average (base) prediction. "
    "Positive = pushes toward churn."
)

st.divider()

# ======================================================================
# EXPLANATION ENGINE RESULTS
# ======================================================================

st.subheader("🔎 Model Explanation")

factor_col1, factor_col2 = st.columns(2)

with factor_col1:
    st.markdown("**🔴 Factors Increasing Churn**")
    for factor in explanation["top_churn_factors"]:
        st.markdown(f"<div class='factor-line'>• {factor}</div>", unsafe_allow_html=True)

with factor_col2:
    st.markdown("**🟢 Factors Reducing Churn**")
    for factor in explanation["protective_factors"]:
        st.markdown(f"<div class='factor-line'>• {factor}</div>", unsafe_allow_html=True)

st.divider()

# ======================================================================
# AI BUSINESS EXPLANATION — gated behind a button
# ======================================================================
# The LLM is only called when the user explicitly asks for it, and the
# result is cached in session_state, so switching tabs/expanders or any
# other rerun does not trigger another API call.

st.subheader("🤖 AI Business Explanation")

generate_clicked = st.button("Generate AI Explanation", use_container_width=False)

if generate_clicked:
    llm_input = {
        "prediction": "CHURN" if explanation["prediction"] == 1 else "STAY",
        "probability": explanation["probability"],
        "threshold": explanation["business_threshold"],
        "top_factors_increasing": explanation["top_churn_factors"],
        "top_factors_reducing": explanation["protective_factors"]
    }
    with st.spinner("Generating AI explanation..."):
        st.session_state.ai_explanation = generate_llm_explanation(llm_input)

if st.session_state.ai_explanation:
    st.markdown(st.session_state.ai_explanation)
else:
    st.caption("Click the button above to generate a business-friendly write-up of this prediction.")
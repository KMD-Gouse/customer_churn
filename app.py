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

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from churn_utils import add_features, apply_calibration, build_explanation
from app_support import (
    ArtifactLoadError,
    INTERNET_SERVICE_COLUMNS,
    load_model_bundle,
    validate_customer_input,
)
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
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    pipeline, calibrator, config = load_model_bundle(artifact_dir)
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]
    try:
        explainer = shap.TreeExplainer(xgb_model)
    except Exception as exc:
        raise ArtifactLoadError(
            "The SHAP explainer could not be created from the saved model. "
            "Regenerate the artifacts with the installed package versions."
        ) from exc
    return pipeline, calibrator, preprocessor, explainer, config


try:
    (
        pipeline,
        calibrator,
        preprocessor,
        explainer,
        config,
    ) = load_artifacts()
except ArtifactLoadError as exc:
    st.error("The churn model is currently unavailable.")
    st.info(str(exc))
    st.caption(
        "Recovery: run churn_final_corrected.ipynb from beginning to end, "
        "then restart this application."
    )
    st.stop()

feature_names = config["shap_feature_names"]
feature_mapping = config["feature_mapping"]
threshold = float(config["threshold"])


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

def aggregate_shap_to_original_features(shap_row, names, mapping):
    contrib = pd.Series(shap_row, index=names)
    return contrib.groupby(lambda name: mapping[name]).sum().sort_values(
        key=abs, ascending=False
    )


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
            float(input_data["MonthlyCharges"]) * input_data["tenure"],
            step=10.0
        )

    with st.expander("Demographics", expanded=False):
        input_data["gender"] = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
        input_data["SeniorCitizen"] = 1 if st.checkbox("Senior Citizen") else 0
        input_data["Partner"] = st.selectbox("Has Partner", CATEGORICAL_OPTIONS["Partner"])
        input_data["Dependents"] = st.selectbox("Has Dependents", CATEGORICAL_OPTIONS["Dependents"])

    with st.expander("Services", expanded=False):
        input_data["PhoneService"] = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"])
        phone_disabled = input_data["PhoneService"] == "No"
        multiple_line_options = (
            ["No phone service"] if phone_disabled else ["No", "Yes"]
        )
        input_data["MultipleLines"] = st.selectbox(
            "Multiple Lines", multiple_line_options, disabled=phone_disabled
        )
        input_data["InternetService"] = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
        internet_disabled = input_data["InternetService"] == "No"
        for field in INTERNET_SERVICE_COLUMNS:
            options = ["No internet service"] if internet_disabled else ["No", "Yes"]
            label = {
                "OnlineSecurity": "Online Security",
                "OnlineBackup": "Online Backup",
                "DeviceProtection": "Device Protection",
                "TechSupport": "Tech Support",
                "StreamingTV": "Streaming TV",
                "StreamingMovies": "Streaming Movies",
            }[field]
            input_data[field] = st.selectbox(
                label, options, disabled=internet_disabled
            )

    predict_clicked = st.button("Predict Churn", type="primary", width="stretch")


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
    validation_errors, validation_warnings = validate_customer_input(
        input_data, CATEGORICAL_OPTIONS
    )
    st.session_state.result = None
    st.session_state.ai_explanation = None

    for warning in validation_warnings:
        st.warning(warning)

    if validation_errors:
        st.error("Please correct the customer details before predicting:")
        for error in validation_errors:
            st.markdown(f"- {error}")
    else:
        try:
            customer_df = add_features(pd.DataFrame([input_data]))[
                config["input_columns"]
            ]
            customer_transformed = preprocessor.transform(customer_df)
            raw_score = pipeline.predict_proba(customer_df)[:, 1]
            risk_score = float(apply_calibration(calibrator, raw_score)[0])
            prediction = int(risk_score >= threshold)

            shap_values = np.asarray(explainer.shap_values(customer_transformed))
            shap_row = shap_values[0]

            explanation = build_explanation(
                raw_row=customer_df.iloc[0],
                score=risk_score,
                threshold=threshold,
                shap_row=shap_row,
                names=feature_names,
                mapping=feature_mapping,
            )

            st.session_state.result = {
                "risk_score": risk_score,
                "prediction": prediction,
                "customer_transformed": customer_transformed,
                "shap_row": shap_row,
                "explanation": explanation,
            }
        except Exception:
            st.error(
                "The prediction could not be completed. Verify the customer "
                "values and regenerate the model artifacts if the problem persists."
            )


if st.session_state.result is None:
    st.info("Fill in the customer details in the sidebar and click **Predict Churn**.")
    st.stop()

result = st.session_state.result
risk_score = result["risk_score"]
prediction = result["prediction"]
customer_transformed = result["customer_transformed"]
shap_row = result["shap_row"]
explanation = result["explanation"]


# ======================================================================
# RESULTS
# ======================================================================

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Calibrated Churn Risk", f"{risk_score:.1%}")
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
    shap_row, feature_names, feature_mapping
)

top = agg.head(8).reset_index()
top.columns = ["Feature", "SHAP Contribution"]
top["Effect"] = np.where(
    top["SHAP Contribution"] > 0, "↑ Increases churn risk", "↓ Decreases churn risk"
)

st.dataframe(
    top,
    width="stretch",
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
    for factor in explanation["top_factors_increasing"]:
        st.markdown(
            f"<div class='factor-line'>• {factor['factor']} ({factor['strength']})</div>",
            unsafe_allow_html=True,
        )

with factor_col2:
    st.markdown("**🟢 Factors Reducing Churn**")
    for factor in explanation["top_factors_reducing"]:
        st.markdown(
            f"<div class='factor-line'>• {factor['factor']} ({factor['strength']})</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ======================================================================
# AI BUSINESS EXPLANATION — gated behind a button
# ======================================================================
# The LLM is only called when the user explicitly asks for it, and the
# result is cached in session_state, so switching tabs/expanders or any
# other rerun does not trigger another API call.

st.subheader("🤖 AI Business Explanation")

generate_clicked = st.button("Generate AI Explanation", width="content")

if generate_clicked:
    with st.spinner("Generating AI explanation..."):
        st.session_state.ai_explanation = generate_llm_explanation(explanation)

if st.session_state.ai_explanation:
    st.markdown(st.session_state.ai_explanation)
else:
    st.caption("Click the button above to generate a business-friendly write-up of this prediction.")

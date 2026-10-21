"""churn_utils.py - shared helpers (written by the notebook).
Training and the Streamlit app import THE SAME code, so features / calibration /
explanations can never drift apart."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df):
    """Row-wise features only (no statistics learned from data -> no leakage)."""
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    avg = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = avg.fillna(df["MonthlyCharges"])
    return df


# --- Platt calibration (smooth; avoids the 0/1 steps isotonic gives on small data) ---
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(raw_scores, y_true):
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_scores), y_true)


def apply_calibration(calibrator, raw_scores):
    return calibrator.predict_proba(_logit(raw_scores))[:, 1]


# --- LLM-ready explanation payload (consumed by llm_explainer.py) ---
def _clean_value(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    return v


def build_explanation(raw_row, score, threshold, shap_row, names, mapping, k=4):
    """SHAP is summed per ORIGINAL feature (signed), then described with the
    customer's actual value, e.g. 'Contract = Month-to-month' (strength: high)."""
    s = pd.Series(shap_row, index=list(names))
    by_feature = s.groupby(lambda n: mapping[n]).sum()
    top = float(by_feature.abs().max()) or 1.0

    def strength(v):
        a = abs(v) / top
        return "high" if a > 0.6 else "medium" if a > 0.25 else "low"

    def fmt(series):
        return [
            {"factor": f"{feat} = {_clean_value(raw_row[feat])}", "strength": strength(val)}
            for feat, val in series.items()
        ]

    return {
        "prediction": "Likely to churn" if score >= threshold else "Not likely to churn",
        "probability": float(score),   # calibrated RISK SCORE (not a guaranteed probability)
        "threshold": float(threshold),
        "top_factors_increasing": fmt(by_feature[by_feature > 0].sort_values(ascending=False).head(k)),
        "top_factors_reducing": fmt(by_feature[by_feature < 0].sort_values().head(k)),
    }

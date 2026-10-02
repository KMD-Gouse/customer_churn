import joblib
import pandas as pd
import numpy as np
import shap
import xgboost as xgb


# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================

preprocessor = joblib.load("preprocessor.pkl")

feature_names = joblib.load(
    "shap_feature_names.pkl"
)

business_threshold = joblib.load(
    "business_threshold.pkl"
)

categorical_features = joblib.load(
    "categorical_features.pkl"
)

numerical_features = joblib.load(
    "numerical_features.pkl"
)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

xgb_model = xgb.XGBClassifier()

xgb_model.load_model(
    "xgb_model.json"
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    xgb_model
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_original_feature(transformed_feature):
    """
    Convert transformed feature name into original feature name.

    Example:
        categorical__Contract_Month-to-month
        -> Contract

        numerical__tenure
        -> tenure
    """

    if transformed_feature.startswith("numerical__"):
        return transformed_feature.replace(
            "numerical__",
            ""
        )

    if transformed_feature.startswith("categorical__"):

        feature_without_prefix = transformed_feature.replace(
            "categorical__",
            "",
            1
        )

        for feature in categorical_features:

            if feature_without_prefix.startswith(
                feature + "_"
            ):
                return feature

    return transformed_feature


# ============================================================
# HUMAN-READABLE FEATURE DESCRIPTION
# ============================================================

def describe_feature(
    original_feature,
    transformed_feature,
    customer
):
    """
    Convert a technical SHAP feature into
    a human-readable customer explanation.
    """

    value = customer[original_feature]

    # Numerical features
    if original_feature in numerical_features:

        if original_feature == "tenure":

            return f"Short tenure ({value} months)"

        elif original_feature == "MonthlyCharges":

            return f"Monthly charges (${value:.2f})"

        elif original_feature == "TotalCharges":

            return f"Total charges (${value:.2f})"

        elif original_feature == "SeniorCitizen":

            if value == 1:
                return "Senior citizen"

            return "Non-senior citizen"

        return f"{original_feature}: {value}"

    # Categorical features
    return f"{original_feature}: {value}"


# ============================================================
# MAIN EXPLANATION ENGINE
# ============================================================

def explain_customer(customer):
    """
    Generate churn prediction and SHAP explanation
    for a single customer.

    Parameters
    ----------
    customer : dict
        Customer information using the original
        dataset feature names.

    Returns
    -------
    dict
        Probability, prediction, and top
        churn/protective factors.
    """

    # --------------------------------------------------------
    # Convert customer dictionary to DataFrame
    # --------------------------------------------------------

    customer_df = pd.DataFrame(
        [customer]
    )

    # --------------------------------------------------------
    # Churn probability
    # --------------------------------------------------------

    probability = xgb_model.predict_proba(
        preprocessor.transform(customer_df)
    )[0, 1]

    # --------------------------------------------------------
    # Business decision
    # --------------------------------------------------------

    prediction = int(
        probability >= business_threshold
    )

    # --------------------------------------------------------
    # Transform customer for SHAP
    # --------------------------------------------------------

    customer_transformed = (
        preprocessor.transform(customer_df)
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        customer_transformed
    )

    shap_values = np.asarray(
        shap_values
    ).flatten()

    # --------------------------------------------------------
    # Create contribution dataframe
    # --------------------------------------------------------

    contributions = pd.DataFrame({
        "transformed_feature": feature_names,
        "shap_value": shap_values
    })

    contributions["original_feature"] = (
        contributions["transformed_feature"]
        .apply(get_original_feature)
    )

    contributions["abs_shap"] = (
        contributions["shap_value"]
        .abs()
    )

    # --------------------------------------------------------
    # Keep the most relevant contribution for each
    # original feature
    # --------------------------------------------------------

    feature_contributions = []

    for original_feature in contributions[
        "original_feature"
    ].unique():

        rows = contributions[
            contributions["original_feature"]
            == original_feature
        ]

        # For categorical features, use the
        # active one-hot encoded category.
        if original_feature in categorical_features:

            value = customer[original_feature]

            matching_rows = rows[
                rows["transformed_feature"].str.endswith(
                    "_" + str(value)
                )
            ]

            if len(matching_rows) > 0:

                row = matching_rows.iloc[0]

            else:

                row = rows.loc[
                    rows["abs_shap"].idxmax()
                ]

        else:

            row = rows.loc[
                rows["abs_shap"].idxmax()
            ]

        feature_contributions.append({
            "feature": original_feature,
            "shap_value": float(
                row["shap_value"]
            ),
            "description": describe_feature(
                original_feature,
                row["transformed_feature"],
                customer
            )
        })

    feature_contributions = pd.DataFrame(
        feature_contributions
    )

    # --------------------------------------------------------
    # Separate positive and negative contributions
    # --------------------------------------------------------

    increasing = (
        feature_contributions[
            feature_contributions["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
    )

    decreasing = (
        feature_contributions[
            feature_contributions["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
    )

    # --------------------------------------------------------
    # Top factors
    # --------------------------------------------------------

    top_churn_factors = (
        increasing["description"]
        .head(5)
        .tolist()
    )

    protective_factors = (
        decreasing["description"]
        .head(5)
        .tolist()
    )

    # --------------------------------------------------------
    # Final explanation
    # --------------------------------------------------------

    return {
        "probability": float(probability),

        "prediction": prediction,

        "business_threshold": float(
            business_threshold
        ),

        "top_churn_factors": top_churn_factors,

        "protective_factors": protective_factors
    }
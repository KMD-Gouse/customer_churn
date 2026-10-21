"""Validation and artifact-loading support for the Streamlit application."""

import json
from pathlib import Path

import joblib
import numpy as np


ARTIFACT_FILENAMES = {
    "pipeline": "churn_pipeline.pkl",
    "calibrator": "calibrator.pkl",
    "config": "config.json",
}

REQUIRED_CONFIG_KEYS = {
    "threshold",
    "input_columns",
    "categorical_features",
    "numerical_features",
    "shap_feature_names",
    "feature_mapping",
}

INTERNET_SERVICE_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

NUMERIC_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (0.0, 200.0),
    "TotalCharges": (0.0, 10000.0),
}


class ArtifactLoadError(RuntimeError):
    """Raised when the saved model bundle is missing, corrupt, or incompatible."""


def _validate_config(config):
    if not isinstance(config, dict):
        raise ArtifactLoadError("config.json must contain a JSON object.")

    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ArtifactLoadError(
            "config.json is missing required keys: " + ", ".join(missing)
        )

    try:
        threshold = float(config["threshold"])
    except (TypeError, ValueError) as exc:
        raise ArtifactLoadError("The configured threshold must be numeric.") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ArtifactLoadError("The configured threshold must be between 0 and 1.")

    input_columns = config["input_columns"]
    feature_names = config["shap_feature_names"]
    mapping = config["feature_mapping"]
    if not isinstance(input_columns, list) or not input_columns:
        raise ArtifactLoadError("config.json input_columns must be a non-empty list.")
    if not isinstance(feature_names, list) or not feature_names:
        raise ArtifactLoadError(
            "config.json shap_feature_names must be a non-empty list."
        )
    if not isinstance(mapping, dict):
        raise ArtifactLoadError("config.json feature_mapping must be an object.")

    unmapped = sorted(set(feature_names).difference(mapping))
    if unmapped:
        raise ArtifactLoadError(
            "SHAP mappings are missing for "
            f"{len(unmapped)} transformed feature(s). Regenerate the artifacts."
        )


def load_model_bundle(artifact_dir):
    """Load and validate the complete deployment bundle."""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        raise ArtifactLoadError(
            f"Artifact directory not found: {artifact_dir}. "
            "Run the corrected notebook to generate it."
        )

    missing_files = [
        filename
        for filename in ARTIFACT_FILENAMES.values()
        if not (artifact_dir / filename).is_file()
    ]
    if missing_files:
        raise ArtifactLoadError(
            "Missing model artifacts: "
            + ", ".join(missing_files)
            + ". Run the corrected notebook to regenerate the deployment bundle."
        )

    try:
        with (artifact_dir / ARTIFACT_FILENAMES["config"]).open(
            encoding="utf-8"
        ) as config_file:
            config = json.load(config_file)
        _validate_config(config)
        pipeline = joblib.load(artifact_dir / ARTIFACT_FILENAMES["pipeline"])
        calibrator = joblib.load(artifact_dir / ARTIFACT_FILENAMES["calibrator"])
    except ArtifactLoadError:
        raise
    except json.JSONDecodeError as exc:
        raise ArtifactLoadError(
            "config.json is not valid JSON. Regenerate the artifacts."
        ) from exc
    except Exception as exc:
        raise ArtifactLoadError(
            "The saved model files could not be loaded. They may be corrupt or "
            "incompatible with the installed package versions. Regenerate the "
            "artifacts in this environment."
        ) from exc

    named_steps = getattr(pipeline, "named_steps", {})
    if "preprocessor" not in named_steps or "model" not in named_steps:
        raise ArtifactLoadError(
            "The saved pipeline must contain 'preprocessor' and 'model' steps."
        )
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ArtifactLoadError("The saved pipeline does not support predict_proba().")
    if not callable(getattr(calibrator, "predict_proba", None)):
        raise ArtifactLoadError("The saved calibrator does not support predict_proba().")

    return pipeline, calibrator, config


def validate_customer_input(customer, categorical_options):
    """Return blocking errors and non-blocking warnings for one customer."""
    errors = []
    warnings = []

    required_fields = set(categorical_options).union(NUMERIC_RANGES, {"SeniorCitizen"})
    missing = sorted(field for field in required_fields if field not in customer)
    if missing:
        errors.append("Missing required fields: " + ", ".join(missing) + ".")
        return errors, warnings

    for field, allowed in categorical_options.items():
        if customer[field] not in allowed:
            errors.append(
                f"{field} has an unsupported value: {customer[field]!r}."
            )

    for field, (minimum, maximum) in NUMERIC_RANGES.items():
        value = customer[field]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            errors.append(f"{field} must be numeric.")
            continue
        if not np.isfinite(value):
            errors.append(f"{field} must be finite.")
        elif not minimum <= value <= maximum:
            errors.append(f"{field} must be between {minimum} and {maximum}.")

    if customer["SeniorCitizen"] not in (0, 1):
        errors.append("SeniorCitizen must be either 0 or 1.")

    if customer["PhoneService"] == "No":
        if customer["MultipleLines"] != "No phone service":
            errors.append(
                "MultipleLines must be 'No phone service' when PhoneService is 'No'."
            )
    elif customer["MultipleLines"] == "No phone service":
        errors.append(
            "MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'."
        )

    if customer["InternetService"] == "No":
        inconsistent = [
            field
            for field in INTERNET_SERVICE_COLUMNS
            if customer[field] != "No internet service"
        ]
        if inconsistent:
            errors.append(
                "These fields must be 'No internet service' when InternetService "
                "is 'No': " + ", ".join(inconsistent) + "."
            )
    else:
        inconsistent = [
            field
            for field in INTERNET_SERVICE_COLUMNS
            if customer[field] == "No internet service"
        ]
        if inconsistent:
            errors.append(
                "These fields cannot be 'No internet service' when internet is "
                "active: " + ", ".join(inconsistent) + "."
            )

    tenure = customer["tenure"]
    monthly = customer["MonthlyCharges"]
    total = customer["TotalCharges"]
    if isinstance(tenure, (int, float, np.number)) and isinstance(
        total, (int, float, np.number)
    ):
        if np.isfinite(tenure) and np.isfinite(total):
            if tenure == 0 and total != 0:
                errors.append("TotalCharges must be 0 when tenure is 0 months.")
            elif tenure > 0 and total == 0:
                warnings.append(
                    "TotalCharges is 0 for an existing customer; confirm the value."
                )
            elif tenure > 0 and monthly > 0:
                average = total / tenure
                if average < monthly * 0.2 or average > monthly * 2:
                    warnings.append(
                        "TotalCharges looks unusual relative to tenure and monthly "
                        "charges; verify it before using the prediction."
                    )

    return errors, warnings

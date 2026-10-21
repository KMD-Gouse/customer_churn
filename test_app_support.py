import json
import tempfile
import unittest
from pathlib import Path

from app_support import (
    ARTIFACT_FILENAMES,
    ArtifactLoadError,
    load_model_bundle,
    validate_customer_input,
)


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
        "Mailed check",
    ],
}


def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }


class CustomerInputValidationTests(unittest.TestCase):
    def test_valid_customer_has_no_messages(self):
        errors, warnings = validate_customer_input(
            valid_customer(), CATEGORICAL_OPTIONS
        )
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_phone_service_dependency_is_enforced(self):
        customer = valid_customer()
        customer["PhoneService"] = "No"
        errors, _ = validate_customer_input(customer, CATEGORICAL_OPTIONS)
        self.assertTrue(any("MultipleLines" in error for error in errors))

    def test_internet_service_dependencies_are_enforced(self):
        customer = valid_customer()
        customer["InternetService"] = "No"
        errors, _ = validate_customer_input(customer, CATEGORICAL_OPTIONS)
        self.assertTrue(any("No internet service" in error for error in errors))

    def test_zero_tenure_requires_zero_total_charges(self):
        customer = valid_customer()
        customer["tenure"] = 0
        errors, _ = validate_customer_input(customer, CATEGORICAL_OPTIONS)
        self.assertTrue(any("TotalCharges must be 0" in error for error in errors))

    def test_unusual_total_charges_produces_warning(self):
        customer = valid_customer()
        customer["TotalCharges"] = 100.0
        errors, warnings = validate_customer_input(customer, CATEGORICAL_OPTIONS)
        self.assertEqual(errors, [])
        self.assertTrue(any("looks unusual" in warning for warning in warnings))


class ArtifactLoadingTests(unittest.TestCase):
    def test_missing_artifact_directory_has_recovery_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "not-created"
            with self.assertRaisesRegex(ArtifactLoadError, "Run the corrected notebook"):
                load_model_bundle(missing)

    def test_missing_artifact_files_are_named(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ArtifactLoadError, "churn_pipeline.pkl"):
                load_model_bundle(temp_dir)

    def test_invalid_config_is_reported_without_raw_json_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            for filename in ARTIFACT_FILENAMES.values():
                (artifact_dir / filename).write_bytes(b"")
            (artifact_dir / "config.json").write_text("{invalid", encoding="utf-8")

            with self.assertRaisesRegex(ArtifactLoadError, "not valid JSON"):
                load_model_bundle(artifact_dir)

    def test_current_artifact_bundle_is_valid(self):
        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        pipeline, calibrator, config = load_model_bundle(artifact_dir)
        self.assertTrue(callable(pipeline.predict_proba))
        self.assertTrue(callable(calibrator.predict_proba))
        self.assertEqual(config["threshold"], 0.52)


if __name__ == "__main__":
    unittest.main()

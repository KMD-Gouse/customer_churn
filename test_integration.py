import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from app_support import load_model_bundle
from churn_utils import add_features, apply_calibration


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")


class DeploymentScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.calibrator, cls.config = load_model_bundle(
            PROJECT_ROOT / "artifacts"
        )

    def test_saved_app_path_matches_notebook_round_trip_scores(self):
        raw = pd.read_csv(PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        raw["TotalCharges"] = pd.to_numeric(
            raw["TotalCharges"], errors="coerce"
        ).fillna(0)

        modeled = add_features(raw).drop(columns=["customerID", "gender"])
        x = modeled.drop(columns=["Churn"])
        y = modeled["Churn"].map({"No": 0, "Yes": 1})

        _, x_temp, _, y_temp = train_test_split(
            x, y, test_size=0.40, stratify=y, random_state=42
        )
        _, x_test, _, _ = train_test_split(
            x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

        app_rows = add_features(raw.loc[x_test.index[:5]])[
            self.config["input_columns"]
        ]
        raw_scores = self.pipeline.predict_proba(app_rows)[:, 1]
        risk_scores = apply_calibration(self.calibrator, raw_scores)

        np.testing.assert_allclose(
            risk_scores,
            np.array([0.057, 0.194, 0.206, 0.568, 0.024]),
            atol=0.0005,
        )

    def test_configured_threshold_and_model_dimensions_are_consistent(self):
        self.assertEqual(self.config["threshold"], 0.52)
        self.assertEqual(
            len(self.config["shap_feature_names"]),
            len(self.config["feature_mapping"]),
        )
        self.assertEqual(
            set(self.config["input_columns"]),
            set(
                self.config["categorical_features"]
                + self.config["numerical_features"]
            ),
        )


class StreamlitApplicationTests(unittest.TestCase):
    def test_app_starts_and_default_prediction_completes(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)

        app.button[0].click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 2)
        self.assertEqual(app.metric[1].value, "52%")

    def test_invalid_billing_input_is_blocked(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        app.slider[0].set_value(0)
        app.run(timeout=30)

        total_charges = next(
            widget
            for widget in app.number_input
            if widget.label == "Total Charges ($)"
        )
        total_charges.set_value(100.0)
        app.run(timeout=30)
        app.button[0].click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(
            any("correct" in message.value.lower() for message in app.error)
        )


if __name__ == "__main__":
    unittest.main()

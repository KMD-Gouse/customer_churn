import unittest

import numpy as np
import pandas as pd

from churn_utils import (
    add_features,
    apply_calibration,
    build_explanation,
    fit_calibrator,
)


class ChurnUtilsTests(unittest.TestCase):
    def test_add_features_handles_zero_tenure_and_does_not_mutate_input(self):
        raw = pd.DataFrame(
            [
                {
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "Yes",
                    "StreamingTV": "No",
                    "StreamingMovies": "Yes",
                    "PaymentMethod": "Bank transfer (automatic)",
                    "TotalCharges": 0.0,
                    "MonthlyCharges": 70.0,
                    "tenure": 0,
                }
            ]
        )

        result = add_features(raw)

        self.assertNotIn("num_services", raw.columns)
        self.assertEqual(result.loc[0, "num_services"], 4)
        self.assertEqual(result.loc[0, "is_autopay"], 1)
        self.assertEqual(result.loc[0, "avg_monthly_spend"], 70.0)

    def test_platt_calibration_returns_bounded_scores(self):
        raw_scores = np.array([0.05, 0.15, 0.3, 0.7, 0.85, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])

        calibrator = fit_calibrator(raw_scores, labels)
        calibrated = apply_calibration(calibrator, raw_scores)

        self.assertTrue(np.all((calibrated >= 0) & (calibrated <= 1)))
        self.assertTrue(np.all(np.diff(calibrated) >= 0))

    def test_build_explanation_groups_transformed_columns(self):
        raw_row = pd.Series({"Contract": "Month-to-month", "tenure": 3})
        payload = build_explanation(
            raw_row=raw_row,
            score=0.8,
            threshold=0.52,
            shap_row=np.array([0.7, -0.2, -0.1]),
            names=["Contract_month", "tenure_scaled", "tenure_raw"],
            mapping={
                "Contract_month": "Contract",
                "tenure_scaled": "tenure",
                "tenure_raw": "tenure",
            },
        )

        self.assertEqual(payload["prediction"], "Likely to churn")
        self.assertEqual(payload["top_factors_increasing"][0]["strength"], "high")
        self.assertEqual(payload["top_factors_reducing"][0]["factor"], "tenure = 3")


if __name__ == "__main__":
    unittest.main()

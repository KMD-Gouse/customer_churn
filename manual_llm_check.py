"""Manually verify the optional Gemini explanation integration.

This file is intentionally excluded from the automated test suite because it
may make a billable external API request when ``GEMINI_API_KEY`` is configured.
Run it explicitly with ``python manual_llm_check.py``.
"""

from llm_explainer import generate_llm_explanation


EXAMPLE_EXPLANATION = {
    "prediction": "Likely to churn",
    "probability": 0.7159,
    "threshold": 0.52,
    "top_factors_increasing": [
        {"factor": "Contract = Month-to-month", "strength": "high"},
        {"factor": "tenure = 3", "strength": "medium"},
        {"factor": "OnlineSecurity = No", "strength": "medium"},
    ],
    "top_factors_reducing": [
        {"factor": "MultipleLines = No", "strength": "low"},
    ],
}


def main():
    """Request and display one explanation using the configured API key."""
    response = generate_llm_explanation(EXAMPLE_EXPLANATION)
    print("\n========== AI EXPLANATION ==========\n")
    print(response)


if __name__ == "__main__":
    main()

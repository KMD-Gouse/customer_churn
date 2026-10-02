from llm_explainer import generate_llm_explanation


example_explanation = {
    "prediction": "CHURN",
    "probability": 0.7159,
    "threshold": 0.06,

    "top_factors_increasing": [
        "Contract: Month-to-month",
        "Short tenure (3 months)",
        "OnlineSecurity: No",
        "InternetService: Fiber optic",
        "TechSupport: No"
    ],

    "top_factors_reducing": [
        "MultipleLines: No",
        "gender: Female"
    ]
}


response = generate_llm_explanation(example_explanation)

print("\n========== AI EXPLANATION ==========\n")
print(response)
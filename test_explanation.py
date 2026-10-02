from explanation_engine import explain_customer

customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 256.5
}

result = explain_customer(customer)

print("\n==============================")
print("CUSTOMER CHURN EXPLANATION")
print("==============================")

print(
    f"\nChurn Probability: "
    f"{result['probability']:.2%}"
)

print(
    f"Business Threshold: "
    f"{result['business_threshold']:.2f}"
)

print(
    f"Prediction: "
    f"{'CHURN' if result['prediction'] == 1 else 'NO CHURN'}"
)

print("\nTop factors increasing churn:")

for i, factor in enumerate(
    result["top_churn_factors"],
    1
):
    print(f"{i}. {factor}")


print("\nFactors reducing churn:")

for i, factor in enumerate(
    result["protective_factors"],
    1
):
    print(f"{i}. {factor}")
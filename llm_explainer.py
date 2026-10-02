from google import genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Create Gemini client
client = genai.Client(api_key=api_key)


def generate_llm_explanation(explanation):
    """
    Converts the structured ML explanation into
    a business-friendly natural language explanation.
    """

    prediction = explanation["prediction"]
    probability = explanation["probability"]
    threshold = explanation["threshold"]

    increasing_factors = explanation.get(
        "top_factors_increasing", []
    )

    reducing_factors = explanation.get(
        "top_factors_reducing", []
    )

    prompt = f"""
You are an AI assistant explaining a customer churn prediction
to a business user.

IMPORTANT RULES:
- Do NOT make your own prediction.
- Do NOT change the probability or threshold.
- Do NOT invent customer information.
- Do NOT claim that a feature causes churn.
- Do NOT make generalizations about demographic groups.
- Describe SHAP factors only as factors influencing this individual prediction.
- Use ONLY the information provided.

ML PREDICTION
-------------
Prediction: {prediction}
Churn Probability: {probability:.2%}
Business Threshold: {threshold:.2%}

FACTORS INCREASING CHURN RISK
-----------------------------
{increasing_factors}

FACTORS REDUCING CHURN RISK
---------------------------
{reducing_factors}

Provide the response in the following format:

Prediction:
Clearly state whether the customer is predicted to churn
or not.

Risk Summary:
Explain what the probability and business threshold mean.

Main Reasons:
Explain the most important factors contributing to the prediction.

Recommended Actions:
Give 2-4 practical actions the business could consider
based only on the identified factors.

Keep the explanation professional and easy for a
non-technical business user to understand.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            "Unable to generate AI explanation at the moment.\n\n"
            f"Technical error: {str(e)}"
        )
import json
import pandas as pd
from tqdm import tqdm
from ollama import chat

# -----------------------------
# Configuration
# -----------------------------

MODEL = "qwen2.5:7b"

INPUT_FILE = "data/bnpl.csv"

OUTPUT_FILE = "output/bnpl_training.csv"

# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# -----------------------------
# New Columns
# -----------------------------

new_columns = [
    "Context",
    "Prompt",
    "Completion",
    "Risk",
    "Confidence",
    "Missing_Information",
    "Tags"
]

for column in new_columns:
    if column not in df.columns:
        df[column] = ""

# -----------------------------
# Process Rows
# -----------------------------

for index, row in tqdm(df.iterrows(), total=len(df)):

    prompt = f"""
You are a senior Buy Now Pay Later (BNPL) fraud analyst.

Your task is to convert one structured transaction into AI training data.

Rules

- ONLY use the provided information.
- NEVER invent facts.
- If information is missing, explicitly say so.
- Be objective.
- Return ONLY valid JSON.
- Do not use markdown.

Transaction

Transaction ID: {row['Transaction_ID']}
Customer Age: {row['Customer_Age']}
Gender: {row['Gender']}
Annual Income: {row['Annual_Income']}
Credit Score: {row['Credit_Score']}
Purchase Category: {row['Purchase_Category']}
BNPL Provider: {row['BNPL_Provider']}
Purchase Amount: {row['Purchase_Amount']}
Device Type: {row['Device_Type']}
Connection Type: {row['Connection_Type']}
Checkout Time Seconds: {row['Checkout_Time_Seconds']}
Browser: {row['Browser']}
Repayment Status: {row['Repayment_Status']}

Return EXACTLY this JSON:

{{
    "context":"",
    "prompt":"",
    "completion":"",
    "risk":"",
    "confidence":"",
    "missing_information":[
        ""
    ],
    "tags":[
        ""
    ]
}}
"""

    try:

        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response["message"]["content"].strip()

        # Remove accidental markdown fences
        content = content.replace("```json", "")
        content = content.replace("```", "").strip()

        result = json.loads(content)

        df.at[index, "Context"] = result.get("context", "")
        df.at[index, "Prompt"] = result.get("prompt", "")
        df.at[index, "Completion"] = result.get("completion", "")
        df.at[index, "Risk"] = result.get("risk", "")
        df.at[index, "Confidence"] = result.get("confidence", "")
        df.at[index, "Missing_Information"] = ", ".join(
            result.get("missing_information", [])
        )
        df.at[index, "Tags"] = ", ".join(
            result.get("tags", [])
        )

    except Exception as e:

        print(f"\nError processing row {index}")
        print(e)

# -----------------------------
# Save
# -----------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nFinished!")
print(f"Saved to {OUTPUT_FILE}")
import json
import os
import pandas as pd
from tqdm import tqdm
from ollama import chat

# =====================================================
# CONFIGURATION
# =====================================================

MODEL = "qwen2.5:7b"

INPUT_FILE = "data/bnpl.csv"
OUTPUT_FILE = "output/bnpl_training_full.csv"

SAVE_EVERY = 10

# =====================================================
# JSON SCHEMA
# =====================================================

SCHEMA = {
    "type": "object",
    "properties": {
        "context": {"type": "string"},
        "prompt": {"type": "string"},
        "completion": {"type": "string"},
        "risk": {"type": "string"},
        "confidence": {"type": "string"},
        "missing_information": {
            "type": "array",
            "items": {"type": "string"}
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "context",
        "prompt",
        "completion",
        "risk",
        "confidence",
        "missing_information",
        "tags"
    ]
}

# =====================================================
# LOAD DATA
# =====================================================

if os.path.exists(OUTPUT_FILE):
    print("Loading existing output file...")
    df = pd.read_csv(OUTPUT_FILE)
else:
    print("Loading original dataset...")
    df = pd.read_csv(INPUT_FILE)

new_columns = [
    "Context",
    "Prompt",
    "Completion",
    "Risk",
    "Confidence",
    "Missing_Information",
    "Tags"
]

for c in new_columns:
    if c not in df.columns:
        df[c] = ""

# =====================================================
# PROCESS ROWS
# =====================================================

for index, row in tqdm(df.iterrows(), total=len(df)):
# for index, row in tqdm(df.head(100).iterrows(), total=100):
    
    # Skip completed rows
    if pd.notna(row["Context"]) and str(row["Context"]).strip() != "":
        continue

    prompt = f"""
You are a senior Buy Now Pay Later (BNPL) fraud and credit risk analyst.

Analyze ONE transaction.

Rules:

- ONLY use provided information.
- NEVER invent facts.
- State when information is insufficient.
- Be concise.
- Do not speculate.

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
"""

    try:

        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=SCHEMA
        )

        result = json.loads(response["message"]["content"])

        df.at[index, "Context"] = result["context"]
        df.at[index, "Prompt"] = result["prompt"]
        df.at[index, "Completion"] = result["completion"]
        df.at[index, "Risk"] = result["risk"]
        df.at[index, "Confidence"] = result["confidence"]
        df.at[index, "Missing_Information"] = "; ".join(
            result["missing_information"]
        )
        df.at[index, "Tags"] = "; ".join(
            result["tags"]
        )

    except Exception as e:

        print(f"\nRow {index} failed")

        print(e)

        with open("errors.log", "a", encoding="utf-8") as f:
            f.write(f"\nRow {index}\n")
            f.write(str(e))
            f.write("\n")

        continue

    # Save periodically
    if index % SAVE_EVERY == 0:
        df.to_csv(OUTPUT_FILE, index=False)

# =====================================================
# FINAL SAVE
# =====================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nFinished!")
print(f"Saved to {OUTPUT_FILE}")
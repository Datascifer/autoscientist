# import json
# import pandas as pd
# from ollama import chat
# from tqdm import tqdm

# # -------------------------
# # Load CSV
# # -------------------------

# df = pd.read_csv("data/bnpl.csv")

# # -------------------------
# # Create new columns
# # -------------------------

# df["Context"] = ""
# df["Prompt"] = ""
# df["Completion"] = ""
# df["Risk"] = ""
# df["Confidence"] = ""
# df["Missing_information"] = ""
# df["Tags"] = ""

# # -------------------------
# # Process each row
# # -------------------------

# for index, row in tqdm(df.iterrows(), total=len(df)):

#     prompt = f"""
# You are a senior Buy Now Pay Later (BNPL) fraud and credit risk analyst.

# Your job is to convert one structured transaction into AI training data.

# Rules:

# - ONLY use the information provided.
# - Do NOT invent facts.
# - If information is missing, say so.
# - Return VALID JSON ONLY.
# - No markdown.
# - No explanation.

# Transaction

# Transaction ID: {row['Transaction_ID']}
# Customer Age: {row['Customer_Age']}
# Gender: {row['Gender']}
# Annual Income: {row['Annual_Income']}
# Credit Score: {row['Credit_Score']}
# Purchase Category: {row['Purchase_Category']}
# BNPL Provider: {row['BNPL_Provider']}
# Purchase Amount: {row['Purchase_Amount']}
# Device Type: {row['Device_Type']}
# Connection Type: {row['Connection_Type']}
# Checkout Time Seconds: {row['Checkout_Time_Seconds']}
# Browser: {row['Browser']}
# Repayment Status: {row['Repayment_Status']}

# Return EXACTLY this JSON format:

# {{
#     "context":"",
#     "prompt":"",
#     "completion":"",
#     "risk":"",
#     "confidence":""
# }}
# """

#     try:

#         response = chat(
#             model="qwen2.5:7b",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )

#         result = json.loads(response["message"]["content"])

#         df.at[index, "Context"] = result["context"]
#         df.at[index, "Prompt"] = result["prompt"]
#         df.at[index, "Completion"] = result["completion"]
#         df.at[index, "Risk"] = result["risk"]
#         df.at[index, "Confidence"] = result["confidence"]
#         df.at[index, "Missing_information"] = result.get("missing_information", "")
#         df.at[index, "Tags"] = result.get("tags", "")

#     except Exception as e:

#         print(f"Row {index} failed")

#         print(e)

# # -------------------------
# # Save CSV
# # -------------------------

# df.to_csv(
#     "output/bnpl_training.csv",
#     index=False
# )

# print("Finished!")
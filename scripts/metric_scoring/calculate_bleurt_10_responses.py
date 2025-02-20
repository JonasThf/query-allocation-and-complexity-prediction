import json
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Initialize BLEURT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bleurt_model_name = "Elron/bleurt-large-512"
bleurt_model = AutoModelForSequenceClassification.from_pretrained(bleurt_model_name).to(device)
bleurt_tokenizer = AutoTokenizer.from_pretrained(bleurt_model_name)

# Function to calculate BLEURT score
def calculate_bleurt(candidate, reference):
    inputs = bleurt_tokenizer(candidate, reference, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        scores = bleurt_model(**inputs)
    return float(scores.logits.squeeze().cpu().numpy())

# Function to check if a response is empty or contains only whitespaces
def is_invalid_response(response):
    return not response or response.isspace()

# Set the model variable (e.g., "t5_base", "t5_large", or "t5_xl")
model = "t5_base"

# File paths
input_file = f'../../datasets/generated/step_7/train_data_with_50k_times_10_{model}_answers_with_avgtime_max5000.jsonl'
output_file = f'../../datasets/generated/step_8/bleurt_scores_{model}.jsonl'

# Load dataset
df = pd.read_json(input_file, orient='records', lines=True)

# Process each row for BLEURT scores for the specified model
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Calculating BLEURT Scores for {model}"):
    output = row.get('output', "")

    for i in range(1, 11):  # Iterate through 10 responses
        response_key = f'r_{model}_{i}'
        score_key = f'scores_{model}_{i}'
        candidate = row.get(response_key, "")

        if not is_invalid_response(candidate) and output:
            try:
                bleurt_score = calculate_bleurt(candidate, output)
                df.at[index, score_key]['bleurt'] = bleurt_score
            except Exception as e:
                print(f"Error calculating BLEURT score at index {index} for {response_key}: {e}")
                df.at[index, score_key]['bleurt'] = None
        else:
            df.at[index, score_key]['bleurt'] = None

# Save the results
with open(output_file, "w") as out_file:
    for entry in df.to_dict(orient="records"):
        json.dump(entry, out_file, separators=(",", ":"))
        out_file.write("\n")

print(f"Process completed. Results saved to {output_file}.")

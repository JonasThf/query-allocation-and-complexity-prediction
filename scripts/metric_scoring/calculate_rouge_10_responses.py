import json
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

# Function to calculate ROUGE scores
def calculate_rouge(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    return scorer.score(reference, candidate)

# Function to check if a response is empty or contains only whitespaces
def is_invalid_response(response):
    return not response or response.isspace()

# Set the model variable (e.g., "t5_base", "t5_large", "t5_xl")
model = "t5_base"

# File paths
input_file = f'../../datasets/generated/step_7/train_data_with_50k_times_10_{model}_answers_with_avgtime_max5000.jsonl'
output_file = f'../../datasets/generated/step_8/rouge_scores_{model}.jsonl'

# Load dataset
df = pd.read_json(input_file, orient='records', lines=True)

# Process each row for ROUGE scores
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Calculating ROUGE Scores for {model}"):
    output = row.get('output', "")

    for i in range(1, 11):  # Iterate through 10 responses
        response_key = f'r_{model}_{i}'
        score_key = f'scores_{model}_{i}'
        candidate = row.get(response_key, "")

        if not is_invalid_response(candidate) and output:
            try:
                rouge_scores = calculate_rouge(candidate, output)
                df.at[index, score_key]['rouge1'] = rouge_scores['rouge1'].fmeasure
                df.at[index, score_key]['rouge2'] = rouge_scores['rouge2'].fmeasure
                df.at[index, score_key]['rougeL'] = rouge_scores['rougeL'].fmeasure
                df.at[index, score_key]['rougeLsum'] = rouge_scores['rougeLsum'].fmeasure
            except Exception as e:
                print(f"Error calculating ROUGE at index {index} for {response_key}: {e}")
                df.at[index, score_key]['rouge1'] = None
                df.at[index, score_key]['rouge2'] = None
                df.at[index, score_key]['rougeL'] = None
                df.at[index, score_key]['rougeLsum'] = None
        else:
            df.at[index, score_key]['rouge1'] = None
            df.at[index, score_key]['rouge2'] = None
            df.at[index, score_key]['rougeL'] = None
            df.at[index, score_key]['rougeLsum'] = None

# Save the results
with open(output_file, "w") as out_file:
    for entry in df.to_dict(orient="records"):
        json.dump(entry, out_file, separators=(",", ":"))
        out_file.write("\n")

print(f"Process completed. Results saved to {output_file}.")

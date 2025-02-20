import json
import pandas as pd
from sacrebleu import corpus_bleu
from tqdm import tqdm

# Specify the model name (t5_base, t5_large, or t5_xl)
model_name = "t5_base"

# Paths to input and output files
input_file = f"../../datasets/generated/step_7/train_data_with_50k_times_10_{model_name}_answers_with_avgtime_max5000.jsonl"
output_file = f"../../datasets/generated/step_8/bleu_scores_{model_name}.jsonl"

# Function to calculate BLEU score
def calculate_bleu(candidate, reference):
    return corpus_bleu([candidate], [[reference]]).score

# Function to check if a response is empty or contains only whitespaces
def is_invalid_response(response):
    return not response or response.isspace()

# Load dataset
df = pd.read_json(input_file, orient='records', lines=True)

# Process each row for BLEU score with a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Calculating BLEU Scores for {model_name}"):
    output = row.get('output', "")

    for i in range(1, 11):  # Iterate through 10 responses
        response_key = f'r_{model_name}_{i}'
        score_key = f'scores_{model_name}_{i}'
        candidate = row.get(response_key, "")

        if not is_invalid_response(candidate) and output:
            try:
                bleu_score = calculate_bleu(candidate, output)
                df.at[index, score_key]['bleu'] = bleu_score  # Update the BLEU score in the nested dictionary
            except Exception as e:
                print(f"Error calculating BLEU score at index {index} for {response_key}: {e}")
                df.at[index, score_key]['bleu'] = None
        else:
            df.at[index, score_key]['bleu'] = None  # Handle invalid responses or missing output

# Save the results
with open(output_file, "w") as output_file:
    for entry in df.to_dict(orient="records"):
        json.dump(entry, output_file, separators=(",", ":"))
        output_file.write("\n")

print(f"BLEU scores saved to {output_file}")

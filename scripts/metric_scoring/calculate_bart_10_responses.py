import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize BART model and tokenizer
bart_model_name = "facebook/bart-large-cnn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name).to(device)
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)

# Set max_length for truncation to avoid large tensor issues
MAX_LENGTH = 1024

# Specify the model name (t5_base, t5_large, or t5_xl)
model_name = "t5_base"

# Paths to input and output files
input_file = f"../../datasets/generated/step_7/train_data_with_50k_times_10_{model_name}_answers_with_avgtime_max5000.jsonl"
output_file = f"../../datasets/generated/step_8/bart_scores_{model_name}.jsonl"

# Function to calculate BART score using log probabilities
def calculate_bart_log_prob(candidate, reference):
    """
    Calculates the BART score using log probabilities.

    Args:
        candidate (str): The generated text to be evaluated.
        reference (str): The reference text for comparison.

    Returns:
        float: The negative log-likelihood score.

    """
    candidate_inputs = bart_tokenizer(candidate, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
    reference_inputs = bart_tokenizer(reference, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)

    with torch.no_grad():
        outputs = bart_model(input_ids=reference_inputs.input_ids, attention_mask=reference_inputs.attention_mask, labels=candidate_inputs.input_ids)

    # Negative log-likelihood
    return float(-outputs.loss.item())

def is_invalid_response(response):
    """
    Checks if a response is empty or contains only whitespace.

    Args:
        response (str): The response text to be evaluated.

    Returns:
        bool: True if the response is empty or consists only of whitespace, False otherwise.
    """
    return not response or response.isspace()

# Load dataset
df = pd.read_json(input_file, orient='records', lines=True)

# Process each row for BART score with a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Calculating BART Scores for {model_name}"):
    output = row.get('output', "")

    for i in range(1, 11):  # Iterate through 10 responses
        response_key = f'r_{model_name}_{i}'
        score_key = f'scores_{model_name}_{i}'
        candidate = row.get(response_key, "")

        if is_invalid_response(candidate) or '<mask>' in candidate or '<mask>' in output:
            df.at[index, score_key]['bart'] = None  # Update the 'bart' key inside the corresponding scores dictionary
        else:
            if candidate and output:
                try:
                    bart_score = calculate_bart_log_prob(candidate, output)
                    df.at[index, score_key]['bart'] = bart_score  # Update the 'bart' score in the nested dictionary
                except Exception as e:
                    print(f"Error calculating BART score at index {index} for {response_key}: {e}")
                    df.at[index, score_key]['bart'] = None
            else:
                df.at[index, score_key]['bart'] = None  # Handle case when candidate or output is missing

# Save the results
with open(output_file, "w") as output_file:
    for entry in df.to_dict(orient="records"):
        json.dump(entry, output_file, separators=(",", ":"))
        output_file.write("\n")

print(f"BART scores saved to {output_file}")

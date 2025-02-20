import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize GPT-2 model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Max input length for GPT-2
MAX_LENGTH = 1024

# Function to calculate GPT-2 log probability
def calculate_logprobs_gpt2(candidate, reference):
    # Truncate sequences longer than 1024 tokens
    candidate_inputs = gpt2_tokenizer(candidate, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LENGTH).to(device)
    reference_inputs = gpt2_tokenizer(reference, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LENGTH).to(device)

    with torch.no_grad():
        outputs = gpt2_model(input_ids=reference_inputs.input_ids, attention_mask=reference_inputs.attention_mask, labels=candidate_inputs.input_ids)
    return float(-outputs.loss.item())

# Function to check if a response is empty or contains only whitespaces
def is_invalid_response(response):
    return not response or response.isspace()

# Set the model variable (e.g., "t5_base", "t5_large", "t5_xl")
model = "t5_base"

# File paths
input_file = f'../../datasets/generated/step_7/train_data_with_50k_times_10_{model}_answers_with_avgtime_max5000.jsonl'
output_file = f'../../datasets/generated/step_8/logprobs_scores_{model}.jsonl'

# Load dataset
df = pd.read_json(input_file, orient='records', lines=True)

# Process each row for GPT-2 logprobs score for 10 responses
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Calculating Logprobs Scores for {model}"):
    output = row.get('output', "")

    for i in range(1, 11):  # Iterate through 10 responses
        response_key = f'r_{model}_{i}'
        score_key = f'scores_{model}_{i}'
        candidate = row.get(response_key, "")

        if not is_invalid_response(candidate) and output:
            try:
                logprobs_score = calculate_logprobs_gpt2(candidate, output)
                df.at[index, score_key]['logprobs'] = logprobs_score
            except Exception as e:
                print(f"Error calculating logprobs at index {index} for {response_key}: {e}")
                df.at[index, score_key]['logprobs'] = None
        else:
            df.at[index, score_key]['logprobs'] = None

# Save the results
with open(output_file, "w") as out_file:
    for entry in df.to_dict(orient="records"):
        json.dump(entry, out_file, separators=(",", ":"))
        out_file.write("\n")

print(f"Process completed. Results saved to {output_file}.")

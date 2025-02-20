import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import json

# Load the pre-trained BERT model for sentence similarity
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# Specify the model name (t5_base, t5_large, or t5_xl)
model_name = "t5_base"

# Paths to input and output files
input_file = f"../../datasets/generated/step_7/train_data_with_50k_times_10_{model_name}_answers_with_avgtime_max5000.jsonl"
output_file = f"../../datasets/generated/step_8/bert_scores_{model_name}.jsonl"

# Function to calculate BERT similarity score
def calculate_bert_score(candidate, reference):
    # Encode the sentences
    candidate_embedding = model.encode(candidate, convert_to_tensor=True)
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(candidate_embedding, reference_embedding).item()
    
    return similarity_score

# Function to check if a response is empty or contains only whitespaces
def is_invalid_response(response):
    return not response or response.isspace()

# Load dataset
df = pd.read_json(input_file, orient='records', lines=True)

# Process each row for BERT score for all 10 responses
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Calculating BERT Scores for {model_name}"):
    output = row.get('output', "")
    
    for i in range(1, 11):  # Process responses 1 to 10
        response_key = f"r_{model_name}_{i}"
        scores_key = f"scores_{model_name}_{i}"
        
        candidate = row.get(response_key, "")
        if not is_invalid_response(candidate) and output:
            try:
                bert_score = calculate_bert_score(candidate, output)
                row[scores_key]['bert'] = bert_score  # Update the 'bert' score
            except Exception as e:
                print(f"Error calculating BERT score for {response_key} at index {index}: {e}")
                row[scores_key]['bert'] = None
        else:
            row[scores_key]['bert'] = None

# Save the results
with open(output_file, "w") as f:
    for record in df.to_dict(orient="records"):
        json.dump(record, f)
        f.write("\n")

print(f"BERT scores saved to {output_file}")

import gc
import time
import torch
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

# Function to load a dataset split and add new columns
def load_and_prepare_dataset(split, file_path):
    dataset = load_dataset('json', data_files={split: file_path})[split]
    df = pd.DataFrame(dataset)
    
    # Keep only the necessary columns
    df = df[['id', 'instruction', 'input', 'output']]
    
    # Define the metrics structure
    metrics_template = {
        'bart': 0.0,
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'rougeLsum': 0.0,
        'bleu': 0.0,
        'bert': 0.0,
        'bleurt': 0.0,
        'logprobs': 0.0
    }
    
    # Add response columns for each model
    df['r_t5_large'] = [""] * len(df)
    df['r_t5_xl'] = [""] * len(df)
    df['r_t5_xxl'] = [""] * len(df)
    
    # Add time columns for each model's response time in miliseconds
    df['rt_t5_large'] = [0] * len(df)
    df['rt_t5_xl'] = [0] * len(df)
    df['rt_t5_xxl'] = [0] * len(df)
    
    
    # Add score columns for each model's response
    df['scores_t5_large'] = [metrics_template.copy() for _ in range(len(df))]
    df['scores_t5_xl'] = [metrics_template.copy() for _ in range(len(df))]
    df['scores_t5_xxl'] = [metrics_template.copy() for _ in range(len(df))]
    
    # Add discrepancy columns between two LLMs, whereas if the value is negative, the first LLM performed better.
    df['discrepancy_large_vs_xl'] = [0] * len(df)
    df['discrepancy_xl_vs_xxl'] = [0] * len(df)
    df['discrepancy_large_vs_xxl'] = [0] * len(df)
    
    # Add columns for query features
    df['query_chars_count'] = [0] * len(df)
    df['query_words_count'] = [0] * len(df)
    df['query_unique_word_count'] = [0] * len(df)
    df['query_readability_score'] = [0.0] * len(df)
    df['query_special_tokens_count'] = [0] * len(df)
    # Check if keywords like "explain, analyze, compare, simulate, matrices" are present
    df['query_keywords_count'] = [0] * len(df)
    
    return df

# Load and prepare the dataset
train_df = load_and_prepare_dataset('train', 'mix-instruct/train_data_prepared.jsonl')

# Function to get model response in single queries and measure response time
def get_response_with_timing(model, tokenizer, query, max_new_tokens, device):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    inputs.pop('token_type_ids', None)  # Remove token_type_ids if present

    # Set generation parameters
    generation_params = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "num_return_sequences": 1,               # Generate only one sequence per input
        "no_repeat_ngram_size": 0,               # Default: 0. If 3, ensure no repetition of 3-grams in the generated text
        "eos_token_id": tokenizer.eos_token_id,  # End-of-sequence token to stop generation, prevent the model from generating excessively long sequences that continue beyond the desired point.
    }

    # Measure the start time
    start_time = time.time()

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_params)

    # Measure the end time and calculate response time
    end_time = time.time()
    response_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response, response_time

# Function to process single queries and measure response time
def process_single_queries_with_timing(model, tokenizer, df, start_index, end_index, output_column, time_column, max_new_tokens, device):
    for i in tqdm(range(start_index, end_index), desc=f"Processing Queries with {model.name_or_path}"):
        # Load query from instruction and input column
        instruction = df.loc[i, 'instruction']
        input_text = df.loc[i, 'input']
        query = f"Instruction: {instruction} Input: {input_text}".strip() if instruction else f"Input: {input_text}".strip()

        # Get response from the model and measure the response time
        response, response_time = get_response_with_timing(model, tokenizer, query, max_new_tokens, device)

        # Update rows in the dataframe
        df.at[i, output_column] = response
        df.at[i, time_column] = response_time

    return df

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize answers_df with the same structure as train_df
empty_t5_large_answers_df = train_df.copy()

print("Loading Flan-T5 model...")
model_path_t5_large = "flan-t5-large-model"
t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_path_t5_large).to(device)
t5_tokenizer = AutoTokenizer.from_pretrained(model_path_t5_large)
t5_tokenizer.pad_token = t5_tokenizer.eos_token

t5_output_column = 'r_t5_large'
t5_time_column = 'rt_t5_large'
max_new_tokens = 5000
start_index = 30000
end_index = 40000

print("Running on ", device)
filled_t5_large_answers_df = process_single_queries_with_timing(t5_model, t5_tokenizer, empty_t5_large_answers_df, start_index, end_index, t5_output_column, t5_time_column, max_new_tokens, device)

# Output file path
output_file_path = 'mix-instruct/train_data_with_30kto40k_t5_large_answers_with_time_max5000.json'

# Write the updated DataFrame to the output file
print("Writing the final output to file...")
filled_t5_large_answers_df.to_json(output_file_path, orient='records', lines=True)

print("Process completed.")

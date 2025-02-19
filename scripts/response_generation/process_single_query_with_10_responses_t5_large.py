import gc
import os
import time
import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def construct_query(instruction, input_text):
    """
    Constructs a formatted query string by combining instruction and input text.

    Args:
        instruction (str): Instruction string describing the task.
        input_text (str): Additional input related to the instruction.

    Returns:
        str: A formatted query string with appropriate labels.
    """
    if instruction and input_text:
        return f"Instruction: {instruction} Input: {input_text}"
    elif instruction:
        return f"Instruction: {instruction}"
    else:
        return f"Input: {input_text}"

def load_and_prepare_dataset(file_path):
    """
    Loads a dataset from a JSONL file and initializes additional columns for model responses and evaluation metrics.

    Args:
        file_path (str): Path to the JSONL dataset file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the dataset with additional response and score columns.
    """
    df = pd.read_json(file_path, orient='records', lines=True)
    
    # Keep only the necessary columns and limit to the first 50k queries
    df = df[['id', 'instruction', 'input', 'output']].head(50000)
    
    # Add columns for 10 responses and their respective response times
    for i in range(1, 11):  # Responses start from r_t5_xl_1 to r_t5_xl_10
        df[f'r_t5_large_{i}'] = [""] * len(df)
        df[f'scores_t5_large_{i}'] = [{"bart": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
                                    "rougeLsum": 0.0, "bleu": 0.0, "bert": 0.0, "bleurt": 0.0,
                                    "logprobs": 0.0} for _ in range(len(df))]

    # Add a single column for the average response time across the 10 responses, since model does not return time per response.
    df['rt_t5_large_avg'] = [0.0] * len(df)
    
    return df

# Function to generate independent responses and measure response time
def generate_independent_responses_with_timing(model, tokenizer, query, device, num_responses, max_new_tokens):
    """
    Generates multiple independent responses from an LLM and measures response time.

    Args:
        model (torch.nn.Module): The language model used for generation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text encoding and decoding.
        query (str): The input query for which responses are generated.
        device (torch.device): The device (CPU/GPU) on which inference is performed.
        num_responses (int): Number of responses to generate.
        max_new_tokens (int): Maximum length of generated responses in tokens.

    Returns:
        tuple: A list of generated responses and the average response time (ms).
    """
    inputs = tokenizer(query, return_tensors="pt").to(device)
    inputs.pop('token_type_ids', None)  # Remove token_type_ids if present

    # Set generation parameters
    generation_params = {
        "max_new_tokens": max_new_tokens,        # Ensures output length does not exceed set token limit.
        "do_sample": True,                       # Introduces randomness, allows model to produce diverse/creative responses.
        "temperature": 0.7,                      # Balances creativity with coherence
        "top_k": 50,                             # Encourages coherent responses
        "top_p": 0.9,                            # Adds diversity without losing coherence
        "num_return_sequences": num_responses,   # Generates multiple (here 10) diverse responses
        "repetition_penalty": 1.5,               # Reduces repetitive text
        "no_repeat_ngram_size": 3,               # Prevents repetitive phrases
        "eos_token_id": tokenizer.eos_token_id,  # Ensures proper stopping
        "pad_token_id": tokenizer.pad_token_id,  # Handles varied lengths
    }

    # Measure start time
    start_time = time.time()

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_params)

    # Measure end time and calculate total response time in milliseconds
    end_time = time.time()
    total_response_time = (end_time - start_time) * 1000

    # Decode the responses
    responses = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False) for output in outputs]
    avg_response_time = total_response_time / num_responses

    return responses, avg_response_time

# Function to process queries and store responses in the DataFrame
def process_queries_with_multiple_responses(model, tokenizer, df, start_index, end_index, device, num_responses, max_new_tokens):
    """
    Processes queries from a DataFrame, generates multiple responses using an LLM, and stores them along with response times.

    Args:
        model (torch.nn.Module): The language model used for generating responses.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding and decoding queries.
        df (pandas.DataFrame): DataFrame containing queries with 'instruction' and 'input' columns.
        start_index (int): Starting index in the DataFrame for processing.
        end_index (int): Ending index in the DataFrame for processing.
        device (torch.device): The device (CPU/GPU) for model inference.
        num_responses (int): Number of independent responses to generate for each query.
        max_new_tokens (int): Maximum number of tokens for each generated response.

    Returns:
        pandas.DataFrame: Updated DataFrame with generated responses and average response time.
    """
    for i in tqdm(range(start_index, end_index), desc=f"Processing Queries with {model.name_or_path}"):
        try:
            # Construct the query from instruction and input
            instruction = df.loc[i, 'instruction']
            input_text = df.loc[i, 'input']
            query = construct_query(instruction, input_text)

            # Get responses and the average response time
            responses, avg_response_time = generate_independent_responses_with_timing(
                model, tokenizer, query, device, num_responses, max_new_tokens
            )

            # Store each response in their respective columns
            for j, response in enumerate(responses, start=1):
                df.at[i, f'r_t5_large_{j}'] = response

            # Store the average response time
            df.at[i, 'rt_t5_large_avg'] = float(avg_response_time)
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    return df

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on ", device)

# Load the model and tokenizer
model_path_t5_large = "google/flan-t5-large"
print("Loading Flan-T5-Large model...")
t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_path_t5_large).to(device)
t5_tokenizer = AutoTokenizer.from_pretrained(model_path_t5_large)
t5_tokenizer.pad_token = t5_tokenizer.eos_token
print("Loading complete.")

# Log GPU status
print("Device status:")
log_file = "output_t5_large_10k.log"
os.system(f"nvidia-smi >> {log_file}")

# Load and prepare the dataset
df = load_and_prepare_dataset('../../datasets/generated/step_3/train_data_reduced_50k.jsonl')

# Define parameters for generation
max_new_tokens = 5000
num_responses = 10

# Define range of queries
start_index = 0
end_index = 50000

# Process queries
filled_df = process_queries_with_multiple_responses(t5_model, t5_tokenizer, df, start_index, end_index, device, num_responses, max_new_tokens)

# Define output file path
output_file_path = '../../datasets/generated/step_7/train_data_with_50k_times_10_t5_large_answers_with_avgtime_max5000.jsonl'

# Write the updated DataFrame to a JSONL file
data_list = filled_df.to_dict(orient='records')
print(f"Writing the final output to file {output_file_path}.")
with open(output_file_path, 'w') as output_file:
    for entry in data_list:
        json.dump(entry, output_file)
        output_file.write('\n')

print("Process completed.")
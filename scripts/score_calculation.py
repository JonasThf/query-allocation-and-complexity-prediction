import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import bert_score
import bleurt
from bleurt import score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Function to calculate normalized BART score using log probabilities
def calculate_normalized_bart_log_prob_score(candidate, reference):
    candidate_inputs = bart_tokenizer(candidate, return_tensors="pt", truncation=True, padding=True).to('cuda')
    reference_inputs = bart_tokenizer(reference, return_tensors="pt", truncation=True, padding=True).to('cuda')
    
    with torch.no_grad():
        outputs = bart_model(input_ids=reference_inputs.input_ids, attention_mask=reference_inputs.attention_mask, labels=candidate_inputs.input_ids)
    
    # Negative log-likelihood
    log_prob = outputs.loss.item()
    
    # Normalize by length of the candidate
    normalized_log_prob = log_prob / candidate_inputs.input_ids.size(1)
    
    return float(-normalized_log_prob)

# Function to calculate BART score using log probabilities
def calculate_bart_log_prob_score(candidate, reference):
    candidate_inputs = bart_tokenizer(candidate, return_tensors="pt", truncation=True, padding=True).to('cuda')
    reference_inputs = bart_tokenizer(reference, return_tensors="pt", truncation=True, padding=True).to('cuda')
    
    with torch.no_grad():
        outputs = bart_model(input_ids=reference_inputs.input_ids, attention_mask=reference_inputs.attention_mask, labels=candidate_inputs.input_ids)
    
    # Negative log-likelihood
    log_prob = outputs.loss.item()
    
    return float(-log_prob)

# Define the metrics structure globally
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

# Function to calculate log probability score using GPT-2
def calculate_logprobs_gpt2(candidate, reference):
    # Tokenize and pad the inputs
    candidate_inputs = gpt2_tokenizer(candidate, return_tensors="pt", truncation=True, padding=True).to('cuda')
    reference_inputs = gpt2_tokenizer(reference, return_tensors="pt", truncation=True, padding=True).to('cuda')

    # Ensure both tensors have the same length by padding to the maximum length
    max_length = max(candidate_inputs.input_ids.size(1), reference_inputs.input_ids.size(1))
    candidate_inputs = gpt2_tokenizer(candidate, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length).to('cuda')
    reference_inputs = gpt2_tokenizer(reference, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length).to('cuda')

    with torch.no_grad():
        outputs = gpt2_model(input_ids=reference_inputs.input_ids, attention_mask=reference_inputs.attention_mask, labels=candidate_inputs.input_ids)

    # Negative log-likelihood
    log_prob = outputs.loss.item()

    return float(-log_prob)

# Function to calculate ROUGE score
def calculate_rouge(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Function to calculate BLEU score
def calculate_bleu(candidate, reference):
    bleu = corpus_bleu([candidate], [[reference]])
    return bleu.score

def calculate_bleurt(candidate, reference):
    inputs = bleurt_tokenizer(candidate, reference, return_tensors="pt", truncation=True, padding=True).to('cuda')
    with torch.no_grad():
        scores = bleurt_model(**inputs)
    return float(scores.logits.squeeze().cpu().numpy())

# Ensure all tensors and models are on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BLEURT model
bleurt_model_name = "Elron/bleurt-large-512"
bleurt_model = AutoModelForSequenceClassification.from_pretrained(bleurt_model_name).to(device)
bleurt_tokenizer = AutoTokenizer.from_pretrained(bleurt_model_name)

# Initialize model and metrics
# ATTENTION: This model is not able to process tokens containing "<mask>".
#bart_model_name = "facebook/bart-large-cnn"
#bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name).to('cuda')
#bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)

# Initialize GPT-2 model for logprobs calculation
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Set the padding token
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

if bleurt_tokenizer.pad_token is None:
    bleurt_tokenizer.pad_token = bleurt_tokenizer.eos_token

# Load the DataFrame
df = pd.read_json('mix-instruct/train_data_with_0-50k_merged_t5_large_xl_xxl_answers_without_xxl_time_max5000_with_xxl_bart_score_and_features_model_label.json', orient='records', lines=True)

start_index = 0
end_index = 50000

# Iterate over each entry
for index in tqdm(range(start_index, end_index), desc=f"Calculating scores and other metrics"):
    row = df.iloc[index]
    output = row['output']
    r_t5_large = row.get('r_t5_large', "")
    r_t5_xl = row.get('r_t5_xl', "")
    r_t5_xxl = row.get('r_t5_xxl', "")

    
    # Process for r_t5_large
    if not r_t5_large.strip():  # If candidate is empty
        print(f"Empty candidate at index {index} for r_t5_large")
        df.at[index, 'scores_t5_large'] = {
            k: None for k in metrics_template.keys()}
    else:
        if '<mask>' in output or '<mask>' in r_t5_large:  # If candidate or input contains <mask>
            print(
                f"<mask> detected at index {index} for r_t5_large; skipping BART score")
            scores = metrics_template.copy()
            scores['bart'] = None
            df.at[index, 'scores_t5_large'] = scores
            try:
                # Logprobs Score
                logprobs_t5_score = calculate_logprobs_gpt2(r_t5_large, output)
                df.loc[index, 'scores_t5_large']['logprobs'] = logprobs_t5_score

                # ROUGE Score
                #rouge_scores = calculate_rouge(r_t5_large, output)
                #df.loc[index, 'scores_t5_large']['rouge1'] = rouge_scores['rouge1'].fmeasure
                #df.loc[index, 'scores_t5_large']['rouge2'] = rouge_scores['rouge2'].fmeasure
                #df.loc[index, 'scores_t5_large']['rougeL'] = rouge_scores['rougeL'].fmeasure
                #df.loc[index, 'scores_t5_large']['rougeLsum'] = rouge_scores['rougeLsum'].fmeasure

                # BLEU Score
                #bleu_score = calculate_bleu(r_t5_large, output)
                #df.loc[index, 'scores_t5_large']['bleu'] = bleu_score

                # BERT Score (running on GPU)
                #bert_scores = bert_score.score([r_t5_large], [output], lang="en", device='cuda')
                #df.loc[index, 'scores_t5_large']['bert'] = bert_scores[2][0].item()

                # BLEURT Score
                #bleurt_score = calculate_bleurt(r_t5_large, output)
                #df.loc[index, 'scores_t5_large']['bleurt'] = bleurt_score
            except Exception as e:
                print(f"Error processing row {index} for r_t5_large: {e}")
                df.at[index, 'scores_t5_large'] = {
                    k: None for k in metrics_template.keys()}
        else:
            try:
                # Calculate BART Score
                #bart_t5_large_score = calculate_bart_log_prob_score(
                #    r_t5_large, output)
                #scores = metrics_template.copy()
                #scores['bart'] = bart_t5_large_score
                #df.at[index, 'scores_t5_large'] = scores

                # Logprobs Score
                logprobs_t5_score = calculate_logprobs_gpt2(r_t5_large, output)
                df.loc[index, 'scores_t5_large']['logprobs'] = logprobs_t5_score

                # ROUGE Score
                #rouge_scores = calculate_rouge(r_t5_large, output)
                #df.loc[index, 'scores_t5_large']['rouge1'] = rouge_scores['rouge1'].fmeasure
                #df.loc[index, 'scores_t5_large']['rouge2'] = rouge_scores['rouge2'].fmeasure
                #df.loc[index, 'scores_t5_large']['rougeL'] = rouge_scores['rougeL'].fmeasure
                #df.loc[index, 'scores_t5_large']['rougeLsum'] = rouge_scores['rougeLsum'].fmeasure

                # BLEU Score
                #bleu_score = calculate_bleu(r_t5_large, output)
                #df.loc[index, 'scores_t5_large']['bleu'] = bleu_score

                # BERT Score (running on GPU)
                #bert_scores = bert_score.score([r_t5_large], [output], lang="en", device='cuda')
                #df.loc[index, 'scores_t5_large']['bert'] = bert_scores[2][0].item()

                # BLEURT Score
                #bleurt_score = calculate_bleurt(r_t5_large, output)
                #df.loc[index, 'scores_t5_large']['bleurt'] = bleurt_score
            except Exception as e:
                print(f"Error processing row {index} for r_t5_large: {e}")
                df.at[index, 'scores_t5_large'] = {
                    k: None for k in metrics_template.keys()}

    # Process for r_t5_xl
    if not r_t5_xl.strip():  # If candidate is empty
        print(f"Empty candidate at index {index} for r_t5_xl")
        df.at[index, 'scores_t5_xl'] = {
            k: None for k in metrics_template.keys()}
    else:
        if '<mask>' in output or '<mask>' in r_t5_xl:  # If candidate or input contains <mask>
            print(
                f"<mask> detected at index {index} for r_t5_xl; skipping BART score")
            scores = metrics_template.copy()
            scores['bart'] = None
            df.at[index, 'scores_t5_xl'] = scores
            try:
                # Logprobs Score
                logprobs_t5_score = calculate_logprobs_gpt2(r_t5_xl, output)
                df.loc[index, 'scores_t5_xl']['logprobs'] = logprobs_t5_score

                # ROUGE Score
                #rouge_scores = calculate_rouge(r_t5_xl, output)
                #df.loc[index, 'scores_t5_xl']['rouge1'] = rouge_scores['rouge1'].fmeasure
                #df.loc[index, 'scores_t5_xl']['rouge2'] = rouge_scores['rouge2'].fmeasure
                #df.loc[index, 'scores_t5_xl']['rougeL'] = rouge_scores['rougeL'].fmeasure
                #df.loc[index, 'scores_t5_xl']['rougeLsum'] = rouge_scores['rougeLsum'].fmeasure

                # BLEU Score
                #bleu_score = calculate_bleu(r_t5_xl, output)
                #df.loc[index, 'scores_t5_xl']['bleu'] = bleu_score

                # BERT Score (running on GPU)
                #bert_scores = bert_score.score([r_t5_xl], [output], lang="en", device='cuda')
                #df.loc[index, 'scores_t5_xl']['bert'] = bert_scores[2][0].item()

                # BLEURT Score
                #bleurt_score = calculate_bleurt(r_t5_xl, output)
                #df.loc[index, 'scores_t5_xl']['bleurt'] = bleurt_score
            except Exception as e:
                print(f"Error processing row {index} for r_t5_xl: {e}")
                df.at[index, 'scores_t5_xl'] = {
                    k: None for k in metrics_template.keys()}
        else:
            try:
                # Calculate BART Score
                #bart_t5_xl_score = calculate_bart_log_prob_score(r_t5_xl, output)
                #scores = metrics_template.copy()
                #scores['bart'] = bart_t5_xl_score
                #df.at[index, 'scores_t5_xl'] = scores

                # Logprobs Score
                logprobs_t5_score = calculate_logprobs_gpt2(r_t5_xl, output)
                df.loc[index, 'scores_t5_xl']['logprobs'] = logprobs_t5_score

                # ROUGE Score
                rouge_scores = calculate_rouge(r_t5_xl, output)
                #df.loc[index, 'scores_t5_xl']['rouge1'] = rouge_scores['rouge1'].fmeasure
                #df.loc[index, 'scores_t5_xl']['rouge2'] = rouge_scores['rouge2'].fmeasure
                #df.loc[index, 'scores_t5_xl']['rougeL'] = rouge_scores['rougeL'].fmeasure
                #df.loc[index, 'scores_t5_xl']['rougeLsum'] = rouge_scores['rougeLsum'].fmeasure

                # BLEU Score
                #bleu_score = calculate_bleu(r_t5_xl, output)
                #df.loc[index, 'scores_t5_xl']['bleu'] = bleu_score

                # BERT Score (running on GPU)
                #bert_scores = bert_score.score([r_t5_xl], [output], lang="en", device='cuda')
                #df.loc[index, 'scores_t5_xl']['bert'] = bert_scores[2][0].item()

                # BLEURT Score
                #bleurt_score = calculate_bleurt(r_t5_xl, output)
                #df.loc[index, 'scores_t5_xl']['bleurt'] = bleurt_score
            except Exception as e:
                print(f"Error processing row {index} for r_t5_xl: {e}")
                df.at[index, 'scores_t5_xl'] = {
                    k: None for k in metrics_template.keys()}
    
    
    # Process for r_t5_xxl
    if not r_t5_xxl.strip():  # If candidate is empty
        print(f"Empty candidate at index {index} for r_t5_xxl")
        df.at[index, 'scores_t5_xxl'] = {
            k: None for k in metrics_template.keys()}
    else:
        if '<mask>' in output or '<mask>' in r_t5_xxl:  # If candidate or input contains <mask>
            print(
                f"<mask> detected at index {index} for r_t5_xxl; skipping BART score")
            scores = metrics_template.copy()
            scores['bart'] = None
            df.at[index, 'scores_t5_xxl'] = scores
            try:
                # Logprobs Score
                logprobs_t5_score = calculate_logprobs_gpt2(r_t5_xxl, output)
                df.loc[index, 'scores_t5_xxl']['logprobs'] = logprobs_t5_score

                # ROUGE Score
                #rouge_scores = calculate_rouge(r_t5_xxl, output)
                #df.loc[index, 'scores_t5_xxl']['rouge1'] = rouge_scores['rouge1'].fmeasure
                #df.loc[index, 'scores_t5_xxl']['rouge2'] = rouge_scores['rouge2'].fmeasure
                #df.loc[index, 'scores_t5_xxl']['rougeL'] = rouge_scores['rougeL'].fmeasure
                #df.loc[index, 'scores_t5_xxl']['rougeLsum'] = rouge_scores['rougeLsum'].fmeasure

                # BLEU Score
                #bleu_score = calculate_bleu(r_t5_xxl, output)
                #df.loc[index, 'scores_t5_xl']['bleu'] = bleu_score

                # BERT Score (running on GPU)
                #bert_scores = bert_score.score([r_t5_xxl], [output], lang="en", device='cuda')
                #df.loc[index, 'scores_t5_xxl']['bert'] = bert_scores[2][0].item()

                # BLEURT Score
                #bleurt_score = calculate_bleurt(r_t5_xxl, output)
                #df.loc[index, 'scores_t5_xxl']['bleurt'] = bleurt_score
            except Exception as e:
                print(f"Error processing row {index} for r_t5_xxl: {e}")
                df.at[index, 'scores_t5_xxl'] = {
                    k: None for k in metrics_template.keys()}
        else:
            try:
                # Calculate BART Score
                #bart_t5_xxl_score = calculate_bart_log_prob_score(r_t5_xxl, output)
                #scores = metrics_template.copy()
                #scores['bart'] = bart_t5_xxl_score
                #df.at[index, 'scores_t5_xxl'] = scores

                # Logprobs Score
                logprobs_t5_score = calculate_logprobs_gpt2(r_t5_xxl, output)
                df.loc[index, 'scores_t5_xxl']['logprobs'] = logprobs_t5_score

                # ROUGE Score
                #rouge_scores = calculate_rouge(r_t5_xxl, output)
                #df.loc[index, 'scores_t5_xxl']['rouge1'] = rouge_scores['rouge1'].fmeasure
                #df.loc[index, 'scores_t5_xxl']['rouge2'] = rouge_scores['rouge2'].fmeasure
                #df.loc[index, 'scores_t5_xxl']['rougeL'] = rouge_scores['rougeL'].fmeasure
                #df.loc[index, 'scores_t5_xxl']['rougeLsum'] = rouge_scores['rougeLsum'].fmeasure

                # BLEU Score
                #bleu_score = calculate_bleu(r_t5_xxl, output)
                #df.loc[index, 'scores_t5_xxl']['bleu'] = bleu_score

                # BERT Score (running on GPU)
                #bert_scores = bert_score.score([r_t5_xxl], [output], lang="en", device='cuda')
                #df.loc[index, 'scores_t5_xxl']['bert'] = bert_scores[2][0].item()

                # BLEURT Score
                #bleurt_score = calculate_bleurt(r_t5_xxl, output)
                #df.loc[index, 'scores_t5_xxl']['bleurt'] = bleurt_score
            except Exception as e:
                print(f"Error processing row {index} for r_t5_xxl: {e}")
                df.at[index, 'scores_t5_xxl'] = {
                    k: None for k in metrics_template.keys()}

# Output file path
output_file_path = 'mix-instruct/train_data_with_0-50k_merged_t5_large_xl_xxl_answers_without_xxl_time_max5000_with_xxl_bart_score_and_features_model_label_additional_scores.json'

# Write the updated DataFrame to the output file
print("Writing the final output to file...")
df.to_json(output_file_path, orient='records', lines=True)

print("Process completed.")

# Query Allocation and Complexity Prediction

This repository contains all code, datasets, and scripts for the query allocation and complexity prediction experiments conducted in the master's thesis. The goal of this work is to analyze query complexity, predict optimal LLM allocation, and evaluate various routing strategies to balance efficiency and response quality.

## Repository Structure

### datasets/ 
- MixInstruct dataset (https://huggingface.co/datasets/llm-blender/mix-instruct)
- Generated files structured in 13 preprocessing outputs

### plots/  
- Plots used in the thesis

### preprocessing/  
- Jupyter notebook containing 13 preprocessing scripts (one per step)

### routers/ 
- Four router implementations (baseline, query-feature-based, LLM-performance-based, historical runtime-informed)
- Each as a jupyter notebook

### scripts/
- LLM response generation scripts to retrieve responses from deployed Language Models (Flan-t5-Base, -Large, -XL)
- Metric scoring scripts to assess response quality of captured responses

## Setup Instructions

1. Create and Activate a Virtual Environment
   python3 -m venv env
   source env/bin/activate  (Windows: env\Scripts\activate)

2. Install Dependencies
   pip install -r requirements.txt

3. Install Jupyter Kernel
   Before running any Jupyter notebook, install the kernel for the virtual environment:
   python -m ipykernel install --user --name=env --display-name "Python (env)"

## Experimental Procedure

1. Preprocessing Pipeline (13 Steps)
   The preprocessing workflow consists of 13 steps that must be executed sequentially to prepare a structured dataset for routing experiments.
   All of them were already run once, just unzip all files in /datasets/generated/step_x. To run them again for validation, follow these steps:

   Step 1: Create an Empty Feature Dataset
   Initializes an empty dataset structure for query feature extraction.

   Step 2: Create an Empty Response Dataset
   Initializes an empty dataset structure for storing model responses.

   Step 3: Reduce MixInstruct Dataset to 50,000 Queries
   Extracts 50,000 queries from the MixInstruct dataset.

   Step 4: Categorize Queries
   Uses a categorization model to classify each query into one of 14 categories. Manual formatting may be required.

   Step 5: Calculate Query Features
   Extracts various features from queries and populates the feature dataset.

   Step 6: Calculate Complexity Score
   Computes a complexity score for each query based on its extracted features.

   Step 7: Generate Model Responses
   Uses scripts from scripts/response_generation/ to generate responses from Flan-T5-Base, Flan-T5-Large, and Flan-T5-XL.
   Run each script in a GPU environment with sufficient memory (40GB for XL).

   Step 8: Compute Metric Scores
   Uses scripts/metric_scoring/ to calculate BLEU, ROUGE, BARTScore, BERTScore, and other evaluation metrics.

   Step 9: Merge Responses and Metric Scores
   Combines model responses and their metric scores into training datasets.

   Step 10: Merge Full Dataset
   Merges the 50k query file with 10 responses and scores per query into the response dataset.

   Step 11: Compute Average Scores and Additional Features
   Aggregates average metric scores and computes additional response-based features.

   Step 12: Compute Discrepancies
   Calculates discrepancies between different model outputs, used for model selection.

   Step 13: Compute Ground Truth Model Label
   Determines the optimal model label for each query based on discrepancies and quality scores.

2. Router Training + Testing + Evaluation  
   The routers/ folder contains four different routing implementations. Run each jupyter notebook and make sure all necessary output files from the preprocessing pipeline
   are accessable.

   Have fun!

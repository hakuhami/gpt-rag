# PromiseEval baseline method

This repository is the baseline method, RAG through GPT-4o.  
Since the JSON structure differs for each language's dataset, branches are separated for each language.  
("evaluator.py" is a validation evaluation script for checking and will not be used in production.)

## Program architecture

 ```plaintext
 gpt-rag/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py (Handles loading and saving data)
│   ├── data_preprocessor.py (Preprocesses and transforms data)
│   ├── rag_model.py (Implements the RAG model for analysis)
│   └── evaluator.py (Evaluates model performance) (checking the sample output data)
│
├── scripts/
│   └── run_analysis.py (Runs the entire analysis process)
│
├── data/ (Prepared in each language's branch)
│   ├── raw/
│   │   └── (Json data) (Prepared in each language's branch)
│   ├── processed/
│   │   ├── search_data.json (Processed data for search/retrieval)
│   │   └── test_data.json (Processed data for testing)
│   └── output/
│       ├── predictions.json (Generated data by the LLM)
│       └── average_results.json (The evaluation results of the generated JSON data) (checking the sample output data)
│
├── config/
│   └── config.yml
│
├── main.py (Entry point for running the analysis)
├── .gitignore (Specifies files to ignore in version control)
└── README.md
```

## Experimental procedure

1. Clone the remote branch corresponding to each language (designing prompts tailored to the structure of each language's dataset).

2. Place the JSON files to be analyzed in the "data/raw".

3. Create a config.yml file in the "config" with the following parameters.

   ```plaintext
   openai_api_key: ""
   model_name: ""
   sample_raw_data_path: ""
   search_data_path: ""
   test_data_path: ""
   generated_data_path: ""
   average_results_path: ""
   test_size:

4. Run the "main.py".


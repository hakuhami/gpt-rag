# PromiseEval

This repository the baseline method, RAG through GPT model.

## Program architecture

 ```plaintext
 gpt-rag/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py # Handles loading and saving data
│   ├── data_preprocessor.py # Preprocesses and transforms data
│   ├── rag_model.py # Implements the RAG model for analysis
│   └── evaluator.py # Evaluates model performance
│
├── scripts/
│   └── run_analysis.py # Runs the entire analysis process
│
├── data/
│   ├── raw/
│   │   └── # Json data
│   ├── processed/
│   │   ├── search_data.json # Processed data for search/retrieval
│   │   └── test_data.json # Processed data for testing
│   └── output/
│       └── .gitkeep # Placeholder for output directory
│
├── config/
│   └── config.yml
│
├── main.py # Entry point for running the analysis
├── .gitignore # Specifies files to ignore in version control
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

## Experimental results using sample data (200 samples) for each language

Model : "gpt-4o"
F1 scores : 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality'  
ROUGE scores : 'promise_string', 'evidence_string'  

 ```plaintext
 Chinese
```

 ```plaintext
 English
```

 ```plaintext
 French
```

 ```plaintext
 Japanese
```

 ```plaintext
 Korean
```

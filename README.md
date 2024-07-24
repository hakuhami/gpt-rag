# PromiseEval

This repository the baseline method, RAG through GPT-4o.

## Program architecture

 ```plaintext
 gpt-rag/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py # Handles loading and saving data
│   ├── data_preprocessor.py # Preprocesses and transforms data
│   ├── rag_model.py # Implements the RAG model for analysis
│   ├── evaluator.py # Evaluates model performance
│   └── utils.py # Contains utility functions
│
├── scripts/
│   └── run_analysis.py # Orchestrates the entire analysis process
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
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py # Tests for data loading functions
│   ├── test_data_preprocessor.py # Tests for data preprocessing functions
│   ├── test_rag_model.py # Tests for RAG model implementation
│   └── test_evaluator.py # Tests for evaluation functions
│
├── main.py # Entry point for running the analysis
├── requirements.txt # Lists project dependencies
├── .gitignore # Specifies files to ignore in version control
└── README.md
```

## Experimental procedure

1.
2.

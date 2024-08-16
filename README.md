# PromiseEval baseline method

This repository is the baseline method, RAG through GPT-4o.    
> [!NOTE]
> Since the JSON structure differs for each language's dataset, branches are separated for each language.  
> The main branch is tailored to the Japanese dataset labels, but the parts that change according to the language are only "rag_model.py" (prompt part), "converter.py" (part specifying labels to be removed), and "evaluator.py".  
> Therefore, the structure of the program can be understood by looking at the main branch (all branches are almost the same).

("evaluator.py" is a validation evaluation script for checking and will not be used in production.)

## Program architecture

 ```plaintext
 gpt-rag/
│
├── src/
│   ├── __init__.py
│   ├── converter.py (Arrange the labels in the dataset for experimental purposes)
│   ├── text_extractor.py (Extract the text from each page based on the URLs of the company reports provided in the given JSON file.)
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

3. Run text_extractor.py to extract the text from each page. And run converter.py to arrange the dataset labels for the experiment.

4. Create a config.yml file in the "config" with the following parameters. Copy and paste, then fill in the respective parameters.

   ```plaintext
   openai_api_key: "your_openai_api_key"
   model_name: "gpt-4o"
   sample_raw_data_path: "data/raw/[filename]"
   search_data_path: "data/processed/[filename]"
   test_data_path: "data/processed/[filename]"
   generated_data_path: "data/output/[filename]"
   average_results_path: "data/output/[filename]"
   test_size: 0.2
   ```

5. Run the "main.py".

## JSON format

 ```plaintext
(Japanese)
  {
      "data": str,
      "promise_status": str,
      "promise_string": str or null,
      "verification_timeline": str,
      "evidence_status": str,
      "evidence_string": str or null,
      "evidence_quality": str
  }:
```

 ```plaintext
(Other language)
  {
      "data": str,
      "promise_status": str,
      "verification_timeline": str,
      "evidence_status": str,
      "evidence_quality": str
  }:
```

## Experimental results using sample data (200 samples) for each language

Model : "gpt-4o"  
F1 scores : 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality'  
ROUGE scores : 'promise_string', 'evidence_string'  

 ```plaintext
 Chinese
```

 ```plaintext
 English
 {
  "promise_status": {
    "f": 0.85
  },
  "verification_timeline": {
    "f": 0.5701219512195121
  },
  "evidence_status": {
    "f": 0.7072463768115943
  },
  "evidence_quality": {
    "f": 0.5886363636363637
  }
}
```

 ```plaintext
 French
 {
  "promise_status": {
    "f": 0.855625717566016
  },
  "verification_timeline": {
    "f": 0.48910200523103753
  },
  "evidence_status": {
    "f": 0.8879239040529362
  },
  "evidence_quality": {
    "f": 0.42922824302134654
  }
}
```

 ```plaintext
 Japanese
 {
  "promise_string": {
    "r": 0.475,
    "p": 0.5,
    "f": 0.4799999975599999
  },
  "evidence_string": {
    "r": 0.3,
    "p": 0.3,
    "f": 0.2999999985
  },
  "promise_status": {
    "f": 0.7450980392156862
  },
  "verification_timeline": {
    "f": 0.5242424242424242
  },
  "evidence_status": {
    "f": 0.8946648426812586
  },
  "evidence_quality": {
    "f": 0.5121212121212121
  }
}
```

 ```plaintext
 Korean
```
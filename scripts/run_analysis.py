import sys
import os
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import save_json_data, load_json_data
from src.data_preprocessor import split_data
from src.rag_model import RAGModel
from src.evaluator import evaluate_results
import yaml

def run_analysis(config_path: str) -> None:
    """
    Execute the analysis.

    Args:
        config_path (str): The path to the configuration file
    """
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the pre-prepared JSON data
    json_data = load_json_data(config['sample_raw_data_path'])

    # Split the data into search and test sets
    search_data, test_data = split_data(json_data, test_size=config['test_size'])

    # Save the search and test data
    save_json_data(search_data, config['search_data_path'])
    save_json_data(test_data, config['test_data_path'])

    # Prepare the RAG model with the search data
    rag_model = RAGModel(api_key=config['openai_api_key'], model_name=config['model_name'])
    rag_model.prepare_documents(search_data)

    # Analyze the test data
    predictions = []
    for item in test_data:
        result = rag_model.analyze_paragraph(item['data'])
        predictions.append(result)

    # Save the prediction results
    save_json_data(predictions, config['output_path'])

    # Evaluate the prediction results
    evaluate_scores = evaluate_results(test_data, predictions)

    print("F1 Scores and ROUGE Scores:")
    for element, score in evaluate_scores.items():
        print(f"{element}: {score:.4f}")

if __name__ == "__main__":
    config_path = 'config/config.yml'
    run_analysis(config_path)
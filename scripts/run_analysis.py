import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import save_json_data, load_json_data
from src.data_preprocessor import split_data
from src.rag_model import RAGModel
from src.evaluator import evaluate_results, save_average_results_to_file
import yaml
import json

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
    print("Search and test data is saved.")

    # Prepare the RAG model with the search data
    rag_model = RAGModel(api_key=config['openai_api_key'], model_name=config['model_name'])
    rag_model.prepare_documents(search_data)
    print("Documents are prepared.")

    # Analyze the test data
    predictions = []
    # skipped_items = []
    for item in test_data:
        result = rag_model.analyze_paragraph(item['data'])
        print(f"{result},")
        result_dict = json.loads(result)
        predictions.append(result_dict)
        
    # for item in test_data:
    #     result = rag_model.analyze_paragraph(item['data'])
    #     if result is None:
    #         print("")
    #         print(f"Skipping item due to None result")
    #         print("")
    #         skipped_items.append(result)
    #         continue
        
    #     try:
    #         result_dict = json.loads(result)
    #         print("")
    #         print(f"{result_dict}")
    #         print("")
    #         predictions.append(result_dict)
            
    #     except json.JSONDecodeError:
    #         print("")
    #         print(f"JSONDecodeError occurred. Skipping item.")
    #         print("")
    #         skipped_items.append(result)
    # print("Analysis is completed.")
    # print(f"Skipped items: {skipped_items}")

    # Save the prediction results
    save_json_data(predictions, config['generated_data_path'])
    print("Predictions are saved.")

    # Evaluate the prediction results
    evaluate_scores = evaluate_results(config['test_data_path'], config['generated_data_path'])
    print("Evaluation is completed.")
    
    save_average_results_to_file(evaluate_scores, config['average_results_path'])
    print(f"F1 Scores and ROUGE Scores:{evaluate_scores}")

from typing import List, Dict
from sklearn.metrics import f1_score
import json

def load_json_data(file_path: str) -> List[Dict]:
    """
    Load JSON data from a file.

    Args:
        file_path (str): The path to the JSON file

    Returns:
        List[Dict]: The loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        return json.load(file)

def calculate_f1_scores(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate the F1 score for the prediction results (for categorical elements),
    including 'N/A' for 'promise_status', but excluding 'N/A' for other labels.

    Args:
        true_data (List[Dict]): The ground truth data
        pred_data (List[Dict]): The predicted data

    Returns:
        Dict[str, float]: The F1 score for each element
    """    
    f1_scores = {}
    
    categorical_elements = ['promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']
    
    for element in categorical_elements:
        true_values = []
        pred_values = []
        
        # Filtering: Exclude pairs with N/A labels.
        for true_item, pred_item in zip(true_data, pred_data):
            true_label = true_item.get(element)
            pred_label = pred_item.get(element)
            
            # Only for the 'promise_status' label, pairs containing 'N/A' are included in the calculation.
            if element == 'promise_status':
                true_values.append(true_label)
                pred_values.append(pred_label)
            else:
                if true_label != 'N/A' and pred_label != 'N/A':
                    true_values.append(true_label)
                    pred_values.append(pred_label)
        
        if true_values and pred_values:
            f1 = f1_score(true_values, pred_values, average='weighted')
            f1_scores[element] = f1
    
    return f1_scores

def evaluate_results(true_data_path: str, pred_data_path: str) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the overall performance of the prediction results.(rouge score and f1 score)

    Args:
        true_data_path (str): The file path to the ground truth data
        pred_data_path (str): The file path to the predicted data

    Returns:
        Dict[str, Dict[str, float]]: The evaluation scores for each element
    """
    true_data = load_json_data(true_data_path)
    pred_data = load_json_data(pred_data_path)

    f1_scores = calculate_f1_scores(true_data, pred_data)
    
    evaluation = {}
    evaluation.update({k: {'f': v} for k, v in f1_scores.items()})
    
    return evaluation

def save_average_results_to_file(results: Dict[str, Dict[str, float]], filename: str):
    with open(filename, 'w') as file:
        json.dump(results, file, indent=2)
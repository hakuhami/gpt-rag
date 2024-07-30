from typing import List, Dict
from rouge import Rouge
from sklearn.metrics import f1_score
import json

# Evaluation logics are to be changed according to the language since the JSON structure differs for each language's dataset.

def calculate_f1_scores(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate the F1 score for the prediction results (for categorical elements).

    Args:
        true_data (List[Dict]): The ground truth data
        pred_data (List[Dict]): The predicted data

    Returns:
        Dict[str, float]: The F1 score for each element
    """    
    f1_scores = {}
    
    categorical_elements = ['promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']
    
    for element in categorical_elements:
        true_values = [item.get(element) for item in true_data if isinstance(item, dict) and item.get(element) is not None]
        pred_values = [item.get(element) for item in pred_data if isinstance(item, dict) and item.get(element) is not None]
        
        if true_values and pred_values:
            f1 = f1_score(true_values, pred_values, average='weighted')
            f1_scores[element] = f1
    
    print(f1_scores)
    return f1_scores

def evaluate_results(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the overall performance of the prediction results.(rouge score and f1 score)

    Args:
        true_data (List[Dict]): The ground truth data
        pred_data (List[Dict]): The predicted data

    Returns:
        Dict[str, Dict[str, float]]: The evaluation scores for each element
    """
    f1_scores = calculate_f1_scores(true_data, pred_data)
    
    evaluation = {}
    evaluation.update({k: {'f': v} for k, v in f1_scores.items()})
    
    print(evaluation)
    return evaluation

def average_results(scores_list: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    avg_scores = {}
    if not scores_list:
        return avg_scores
    elements = scores_list[0].keys()
    for element in elements:
        f_scores = [scores[element]['f'] for scores in scores_list if 'f' in scores[element]]
        p_scores = [scores[element]['p'] for scores in scores_list if 'p' in scores[element]]
        r_scores = [scores[element]['r'] for scores in scores_list if 'r' in scores[element]]
        avg_scores[element] = {
            'f': sum(f_scores) / len(f_scores) if f_scores else 0.0,
            'p': sum(p_scores) / len(p_scores) if p_scores else 0.0,
            'r': sum(r_scores) / len(r_scores) if r_scores else 0.0,
        }
    return avg_scores

def save_average_results_to_file(results: Dict[str, Dict[str, float]], filename: str):
    with open(filename, 'w') as file:
        json.dump(results, file, indent=2)
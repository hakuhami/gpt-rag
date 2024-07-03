from typing import List, Dict
from sklearn.metrics import f1_score

def calculate_f1_scores(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, float]:
    """
    予測結果のF1スコアを計算する

    Args:
        true_data (List[Dict]): 正解データ
        pred_data (List[Dict]): 予測データ

    Returns:
        Dict[str, float]: 各要素のF1スコア
    """
    f1_scores = {}
    
    elements = ['commitment_present', 'commitment_timing', 'evidence_present', 'relation_quality']
    
    for element in elements:
        true_values = [item[element] for item in true_data if element in item and item[element] is not None]
        pred_values = [item[element] for item in pred_data if element in item and item[element] is not None]
        
        if len(true_values) > 0 and len(pred_values) > 0:
            f1 = f1_score(true_values, pred_values, average='weighted')
            f1_scores[element] = f1
    
    return f1_scores
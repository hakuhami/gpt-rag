# from typing import List, Dict
# from sklearn.metrics import f1_score

# def calculate_f1_scores(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, float]:
#     """
#     予測結果のF1スコアを計算する

#     Args:
#         true_data (List[Dict]): 正解データ
#         pred_data (List[Dict]): 予測データ

#     Returns:
#         Dict[str, float]: 各要素のF1スコア
#     """
#     f1_scores = {}
    
#     elements = ['commitment_present', 'commitment_text', 'commitment_timing', 'evidence_present','evidence_text', 'relation_quality']
    
#     for element in elements:
#         true_values = [item[element] for item in true_data if element in item and item[element] is not None]
#         pred_values = [item[element] for item in pred_data if element in item and item[element] is not None]
        
#         if len(true_values) > 0 and len(pred_values) > 0:
#             f1 = f1_score(true_values, pred_values, average='weighted')
#             f1_scores[element] = f1
    
#     return f1_scores

from typing import List, Dict
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

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
    
    elements = ['commitment_present', 'commitment_text', 'commitment_timing', 'evidence_present', 'evidence_text', 'relation_quality']
    
    for element in elements:
        true_values = [str(item.get(element)) for item in true_data]
        pred_values = [str(item.get(element)) for item in pred_data]
        
        # テキスト要素を数値にエンコード
        if element in ['commitment_text', 'evidence_text']:
            le = LabelEncoder()
            true_values = le.fit_transform(true_values)
            pred_values = le.transform(pred_values)
        else:
            # 数値要素の場合、Noneを特別な値（例：-1）に置き換え
            true_values = [-1 if v == 'None' else int(v) for v in true_values]
            pred_values = [-1 if v == 'None' else int(v) for v in pred_values]
        
        f1 = f1_score(true_values, pred_values, average='weighted', zero_division=0)
        f1_scores[element] = f1
    
    return f1_scores
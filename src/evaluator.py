# from typing import List, Dict
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import LabelEncoder

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
    
#     elements = ['commitment_present', 'commitment_text', 'commitment_timing', 'evidence_present', 'evidence_text', 'relation_quality']
    
#     for element in elements:
#         true_values = [str(item.get(element)) for item in true_data]
#         pred_values = [str(item.get(element)) for item in pred_data]
        
#         # テキスト要素を数値にエンコード
#         if element in ['commitment_text', 'evidence_text']:
#             le = LabelEncoder()
#             true_values = le.fit_transform(true_values)
#             pred_values = le.transform(pred_values)
#         else:
#             # 数値要素の場合、Noneを特別な値（例：-1）に置き換え
#             true_values = [-1 if v == 'None' else int(v) for v in true_values]
#             pred_values = [-1 if v == 'None' else int(v) for v in pred_values]
        
#         f1 = f1_score(true_values, pred_values, average='weighted', zero_division=0)
#         f1_scores[element] = f1
    
#     return f1_scores

from typing import List, Dict
from rouge import Rouge

def calculate_rouge_scores(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    予測結果のROUGEスコアを計算する

    Args:
        true_data (List[Dict]): 正解データ
        pred_data (List[Dict]): 予測データ

    Returns:
        Dict[str, Dict[str, float]]: 各要素のROUGEスコア
    """
    rouge = Rouge()
    rouge_scores = {}
    
    text_elements = ['commitment_text', 'evidence_text']
    
    for element in text_elements:
        true_texts = [item.get(element, "") for item in true_data]
        pred_texts = [item.get(element, "") for item in pred_data]
        
        # 空の文字列を除外
        valid_pairs = [(t, p) for t, p in zip(true_texts, pred_texts) if t and p]
        
        if valid_pairs:
            true_valid, pred_valid = zip(*valid_pairs)
            scores = rouge.get_scores(pred_valid, true_valid, avg=True)
            rouge_scores[element] = scores['rouge-l']
        else:
            rouge_scores[element] = {'f': 0.0, 'p': 0.0, 'r': 0.0}
    
    return rouge_scores

def calculate_f1_scores(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, float]:
    """
    予測結果のF1スコアを計算する（カテゴリカルな要素用）

    Args:
        true_data (List[Dict]): 正解データ
        pred_data (List[Dict]): 予測データ

    Returns:
        Dict[str, float]: 各要素のF1スコア
    """
    from sklearn.metrics import f1_score
    
    f1_scores = {}
    
    categorical_elements = ['commitment_present', 'commitment_timing', 'evidence_present', 'relation_quality']
    
    for element in categorical_elements:
        true_values = [item.get(element) for item in true_data if item.get(element) is not None]
        pred_values = [item.get(element) for item in pred_data if item.get(element) is not None]
        
        if true_values and pred_values:
            f1 = f1_score(true_values, pred_values, average='weighted')
            f1_scores[element] = f1
        else:
            f1_scores[element] = 0.0
    
    return f1_scores

def evaluate_results(true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    予測結果の総合評価を行う

    Args:
        true_data (List[Dict]): 正解データ
        pred_data (List[Dict]): 予測データ

    Returns:
        Dict[str, Dict[str, float]]: 各要素の評価スコア
    """
    rouge_scores = calculate_rouge_scores(true_data, pred_data)
    f1_scores = calculate_f1_scores(true_data, pred_data)
    
    evaluation = {}
    evaluation.update(rouge_scores)
    evaluation.update({k: {'f': v} for k, v in f1_scores.items()})
    
    return evaluation
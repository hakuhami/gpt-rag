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
    
    text_elements = ['promise_string', 'evidence_string']
    
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
    
    categorical_elements = ['promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']
    
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
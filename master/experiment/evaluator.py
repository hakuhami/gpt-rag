# HICSS向けの、業種ごと、ESGごとの分析用スクリプト
# 入力：どの業種か（どのESGか）、どのディレクトリ内か、"evidence_quality"の内のどの値か（2つのディレクトリ群を比較し、該当の業種ごとにURLで判定しデータを全て見る）
# 出力：それぞれのディレクトリにて、指定したラベルの数（エクセルの表にまとめられるように）


# HICSS向けの、業種ごと、ESGごとの分析用スクリプト
# 実行するだけで、以下で設定した条件に基づいて分析結果が表示される

import os
import json
from typing import List, Dict
from collections import Counter

# 業種ごとの企業URLリスト（実際のURLは適宜置き換えてください）
INDUSTRY_DOMAINS = {
    "automobile": [
        "global.honda",
        "www.mazda.com",
        "www.unipres.co.jp"
    ],
    "energy": [
        "www.sustainability-report.inpex.co.jp",
        "www.tohoku-epco.co.jp",
        "hd.saibugas.co.jp"
    ],
    "trade": [
        "www.itochu.co.jp",
        "marubeni.disclosure.site",
        "www.mitsuuroko.com"
    ]
}

def load_all_data(file_path: str) -> List[Dict]:
    """
    単一のJSONファイルから正解データを読み込む
    
    Args:
        file_path (str): 正解データのファイルパス
        
    Returns:
        List[Dict]: 読み込まれた正解データ
    """
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
        print(f"Successfully loaded {len(data)} items from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: JSON decode error in file: {file_path}")
        return []

def load_all_predictions(base_dir: str) -> List[Dict]:
    """
    予測結果データを全て読み込む
    
    Args:
        base_dir (str): 予測結果のベースディレクトリ
        
    Returns:
        List[Dict]: 読み込まれた全予測結果
    """
    all_predictions = []
    for i in range(1, 6):  # 1から5までの各セット
        file_path = os.path.join(base_dir, f"{i}_set_predictions.json")
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                predictions = json.load(file)
                all_predictions.extend(predictions)
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Warning: JSON decode error in file: {file_path}")
    
    return all_predictions

def filter_data_by_industry(data: List[Dict], industry: str) -> List[Dict]:
    """
    業種に基づいてデータをフィルタリング（ドメイン部分一致）
    
    Args:
        data (List[Dict]): フィルタリングするデータ
        industry (str): 業種名
        
    Returns:
        List[Dict]: フィルタリングされたデータ
    """
    if industry not in INDUSTRY_DOMAINS:
        print(f"Warning: Unknown industry: {industry}")
        return []
    
    domains = INDUSTRY_DOMAINS[industry]
    filtered_data = []
    
    for item in data:
        url = item.get('URL', '')
        # いずれかのドメインがURLに含まれているかチェック
        if any(domain in url for domain in domains):
            filtered_data.append(item)
    
    return filtered_data

def filter_data_by_esg_type(data: List[Dict], esg_type: str) -> List[Dict]:
    """
    ESGタイプに基づいてデータをフィルタリング
    
    Args:
        data (List[Dict]): フィルタリングするデータ
        esg_type (str): ESGタイプ（E, S, G）
        
    Returns:
        List[Dict]: フィルタリングされたデータ
    """
    return [item for item in data if item.get('ESG_type', '') == esg_type]

def analyze_evidence_quality(ground_truth: List[Dict], predictions: List[Dict], 
                           true_value: str, pred_value: str) -> Dict:
    """
    evidence_qualityの値の分析
    
    Args:
        ground_truth (List[Dict]): 正解データ
        predictions (List[Dict]): 予測データ
        true_value (str): 正解データの条件
        pred_value (str): 予測データの条件
        
    Returns:
        Dict: 分析結果
    """
    # IDでマッチングするためのディクショナリ作成
    gt_dict = {item.get('id', i): item for i, item in enumerate(ground_truth)}
    pred_dict = {item.get('id', i): item for i, item in enumerate(predictions)}
    
    # 共通のIDを抽出
    common_ids = set(gt_dict.keys()) & set(pred_dict.keys())
    
    # 条件に一致するデータをカウント
    matching_items = []
    for item_id in common_ids:
        gt_item = gt_dict[item_id]
        pred_item = pred_dict[item_id]
        
        gt_eq = gt_item.get('evidence_quality', 'N/A')
        pred_eq = pred_item.get('evidence_quality', 'N/A')
        
        # 指定された条件に一致するかチェック
        if gt_eq == true_value and pred_eq == pred_value:
            matching_items.append({
                'id': item_id,
                'URL': gt_item.get('URL', ''),
                'ESG_type': gt_item.get('ESG_type', ''),
                'ground_truth_evidence_quality': gt_eq,
                'prediction_evidence_quality': pred_eq
            })
    
    # 総カウント
    total_count = len(matching_items)
    
    # ESG_typeごとのカウント
    esg_counts = Counter([item.get('ESG_type', 'unknown') for item in matching_items])
    
    return {
        'total_count': total_count,
        'matching_items': matching_items,
        'esg_counts': dict(esg_counts)
    }

def main():
    # ==================== 分析条件を設定 ====================
    
    ### フィルタリング条件（どちらか一方を指定し、もう一方はコメントアウト）
    industry = "trade"  # "automobile", "energy", "trade" のいずれかを指定
    # esg_type = "E"  # "E", "S", "G" のいずれかを指定
    
    # evidence_qualityの条件
    true_value = "N/A"  # 正解データの条件: "Clear", "Not Clear", "Misleading", "N/A"
    pred_value = "N/A"  # 予測データの条件: "Clear", "Not Clear", "Misleading", "N/A"
    # =====================================================
    
    # データの読み込み
    print("Loading ground truth data...")
    ground_truth = load_all_data("master/data/Japanese_merged1000_added_index.json")
    
    print("Loading prediction data...")
    predictions = load_all_predictions("master/experiment/result/gpt/2_rag")
    
    print(f"Loaded {len(ground_truth)} ground truth items and {len(predictions)} prediction items")
    
    # フィルタリング
    if 'industry' in locals() and industry:
        print(f"Filtering by industry: {industry}")
        filtered_ground_truth = filter_data_by_industry(ground_truth, industry)
        filtered_predictions = filter_data_by_industry(predictions, industry)
    elif 'esg_type' in locals() and esg_type:
        print(f"Filtering by ESG type: {esg_type}")
        filtered_ground_truth = filter_data_by_esg_type(ground_truth, esg_type)
        filtered_predictions = filter_data_by_esg_type(predictions, esg_type)
    else:
        print("エラー: industry または esg_type のどちらかを指定してください")
        return
    
    print(f"After filtering: {len(filtered_ground_truth)} ground truth items, {len(filtered_predictions)} prediction items")
    
    # 分析
    print(f"\nAnalyzing where ground truth is '{true_value}' and prediction is '{pred_value}'...")
    results = analyze_evidence_quality(filtered_ground_truth, filtered_predictions, 
                                     true_value, pred_value)
    
    # 結果出力（統計量のみ）
    print(f"\nResults:")
    print(f"Total matching items: {results['total_count']}") # こいつが該当する条件の数
    print(f"ESG type breakdown: {results['esg_counts']}")
    
if __name__ == "__main__":
    main()
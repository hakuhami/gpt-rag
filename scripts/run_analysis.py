import sys
import os
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_annotated_data_from_excel, save_json_data
from src.data_preprocessor import preprocess_data, split_data
from src.rag_model import RAGModel
from src.evaluator import calculate_f1_scores
import yaml

def run_analysis(config_path: str) -> None:
    """
    分析を実行する

    Args:
        config_path (str): 設定ファイルのパス
    """
    # 設定を読み込む
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # データを読み込む
    df = load_annotated_data_from_excel(config['raw_data_path'])
    json_data = preprocess_data(df)

    # データを分割する
    search_data, test_data = split_data(json_data, test_size=config['test_size'])

    # 分割したデータを保存する
    save_json_data(search_data, config['search_data_path'])
    save_json_data(test_data, config['test_data_path'])

    # RAGモデルを初期化し、検索用データを準備する
    rag_model = RAGModel(api_key=config['openai_api_key'], model_name=config['model_name'])
    rag_model.prepare_documents(search_data)

    # テストデータに対して分析を実行する
    predictions = []
    for item in test_data:
        result = rag_model.analyze_paragraph(item['paragraph'])
        predictions.append(result)

    # 予測結果を保存する
    save_json_data(predictions, config['output_path'])

    # F1スコアを計算する
    f1_scores = calculate_f1_scores(test_data, predictions)

    print("F1 Scores:")
    for element, score in f1_scores.items():
        print(f"{element}: {score:.4f}")

if __name__ == "__main__":
    config_path = 'config/config.yml'
    run_analysis(config_path)
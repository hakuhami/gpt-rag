import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import save_json_data

# from src.rag_model_8_2 import RAGModel
# from image_deim.image_rag_eq_only import RAGModel
from master.experiment.rag_model_8_2_without_rag_gemini import RAGModel

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
    
    # # Load the search data from the file（ベースラインの実験では使用しない）
    # with open(config['search_data_path'], 'r', encoding='utf-8-sig') as f:
    #     search_data = json.load(f)

    # Load the test data from the file
    with open(config['test_data_path'], 'r', encoding='utf-8-sig') as f:
        test_data = json.load(f)
        
    print("Search and test data is loaded.")

    # 既存の予測結果がある場合は読み込む
    predictions = []
    if os.path.exists(config['generated_data_path']):
        try:
            with open(config['generated_data_path'], 'r', encoding='utf-8-sig') as f:
                predictions = json.load(f)
            print(f"Loaded {len(predictions)} existing predictions.")
        except:
            print("Failed to load existing predictions, starting fresh.")
    
    # 既に分析済みのIDを記録
    processed_ids = {item.get('id', '') for item in predictions}

    # Prepare the RAG model with the search data
    rag_model = RAGModel(api_key=config['gemini_api_key'], model_name=config['model_name'])
    print("RAGModel is loaded.")
    # rag_model.prepare_documents(search_data) #（ベースラインの実験では使用しない）
    # print("Documents are prepared.")
    
    # 未処理のデータ数を計算
    remaining_items = [item for item in test_data if item.get('id', '') not in processed_ids]
    print(f"データ数のトータル: {len(test_data)}, Already processed: {len(predictions)}, Remaining: {len(remaining_items)}")
    print("実験開始！！！")

    # # Analyze the test data
    # predictions = []
    # for item in test_data:
    #     # result = rag_model.analyze_paragraph(item['data'])
    #     ### ↓イメージベースでは、"id"を引き継ぐ（または、根拠の性質以外はベースラインの結果を引き継ぐ）ため、辞書でデータを渡す
    #     #### ↓HICSS向けでは、"id"や""URL"など、全部引き継ぐ
    #     result = rag_model.analyze_paragraph(item)
    #     print(f"{result},")
    #     result_dict = json.loads(result)
    #     predictions.append(result_dict)
    # print("Analysis is completed.")
    
    # (HICSS用のコード)Analyze the remaining test data
    for i, item in enumerate(remaining_items):
        item_id = item.get('id', f'item_{i}')
        print(f"実験するデータのID: {item_id}")
        try:            
            result = rag_model.analyze_paragraph(item)
            print(f"{result},")
            
            result_dict = json.loads(result)
            
            # 元のIDを保持
            if 'id' in item:
                result_dict['id'] = item['id']
                
            # 新しい予測を追加
            predictions.append(result_dict)
            
            # 予測を即座にファイルに保存
            save_json_data(predictions, config['generated_data_path'])
            
        except Exception as e:
            print(f"Error processing item with ID {item_id}: {e}")
            print("You can restart from this ID later.")
            # エラーが発生しても現時点までの結果は保存済み
            break

    # # Save the prediction results
    # save_json_data(predictions, config['generated_data_path'])
    # print("Predictions are saved.")

    # Evaluate the prediction results
    evaluate_scores = evaluate_results(config['test_data_path'], config['generated_data_path'])
    print("Evaluation is completed.")
    
    save_average_results_to_file(evaluate_scores, config['evaluation_results_path'])
    print(f"F1 Scores and ROUGE Scores:{evaluate_scores}")

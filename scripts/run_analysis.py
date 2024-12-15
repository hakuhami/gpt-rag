import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import save_json_data, load_json_data
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
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    
    # Load the search data from the file
    with open(config['search_data_path'], 'r', encoding='utf-8-sig') as f:
        search_data = json.load(f)
        for item in search_data:
            item['id'] = int(item['id'])

    # Load the test data from the file
    with open(config['test_data_path'], 'r', encoding='utf-8-sig') as f:
        test_data = json.load(f)
        for item in test_data:
            item['id'] = int(item['id'])

    # Prepare the RAG model with the search data
    print("Start embedding")
    rag_model = RAGModel(api_key=config['openai_api_key'], model_name=config['model_name'])
    print("Documents are prepared.")

    # Analyze the test data
    predictions = []
    for item in test_data:
        try:
            item['id'] = int(item['id'])
            image_path = os.path.join("data/processed/images_experiment", f"{item['id']}.png")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"画像ファイルが存在しません: {image_path}")
            
            print(f"画像 {item['id']} の処理を開始します")
            image = rag_model.load_image(image_path)
            
            result = rag_model.analyze_paragraph(image, item['id'], item['data'])
            print(f"{result},")
            result_dict = json.loads(result)
            
            required_keys = ["id", "promise_status", "promise_string", "verification_timeline", 
                           "evidence_status", "evidence_string", "evidence_quality"]
            if not all(key in result_dict for key in required_keys):
                print(f"警告: 画像 {item['id']} の結果に必要なキーが不足しています")
            
            predictions.append(result_dict)
            print(f"画像 {item['id']} の処理が完了しました")
            
        except Exception as e:
            print(f"画像 {item['id']} の処理中にエラーが発生しました: {e}")
            raise
        
    # Save the prediction results
    save_json_data(predictions, config['generated_data_path'])
    print("Predictions are saved.")

    # Evaluate the prediction results
    evaluate_scores = evaluate_results(config['test_data_path'], config['generated_data_path'])
    print("Evaluation is completed.")
    
    save_average_results_to_file(evaluate_scores, config['evaluation_results_path'])
    print(f"F1 Scores and ROUGE Scores:{evaluate_scores}")
import json
from typing import Union, Dict, List

def remove_id_fields(data: Union[Dict, List]) -> Union[Dict, List]:
    if isinstance(data, dict):
        return {k: remove_id_fields(v) for k, v in data.items() if k != "id"}
    elif isinstance(data, list):
        return [remove_id_fields(item) for item in data]
    return data

def process_json_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    processed_data = remove_id_fields(data)
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_path = "./data/output/predictions.json"
    output_path = "./data/output/Japanese_withoutRAG.json"
    
    try:
        process_json_file(input_path, output_path)
        print(f"処理が完了しました。\n入力: {input_path}\n出力: {output_path}")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
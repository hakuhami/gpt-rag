import json
import random

# ファイルパスの設定
input_file = "data/deim_output/Japanese_merged600_experiment.json"
output_500 = "data/deim_output/Japanese_5cross_500.json"
output_100 = "data/deim_output/Japanese_validation_100.json"

def split_json_data():
    # JSONファイルの読み込み
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # データがリスト形式であることを確認
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of items")
        
        # データをランダムにシャッフル
        random.shuffle(data)
        
        # データを分割
        data_500 = data[:500]
        data_100 = data[500:600]
        
        # 500件のデータを保存
        with open(output_500, 'w', encoding='utf-8-sig') as f:
            json.dump(data_500, f, ensure_ascii=False, indent=2)
        
        # 100件のデータを保存
        with open(output_100, 'w', encoding='utf-8-sig') as f:
            json.dump(data_100, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully split data:")
        print(f"- {len(data_500)} items written to {output_500}")
        print(f"- {len(data_100)} items written to {output_100}")
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input file")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# メイン処理の実行
if __name__ == "__main__":
    split_json_data()
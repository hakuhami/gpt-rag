import json
import os

# ディレクトリパスの設定
base_dir = "data/deim_output"
set_dirs = [f"{i}_set" for i in range(1, 6)]

def process_json_file(input_path: str, output_path: str):
    """
    JSONファイルからexplanationラベルを削除し、新しいファイルを作成する関数
    """
    try:
        # 入力ファイルの読み込み
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # 各データからexplanationラベルを削除
        processed_data = []
        for item in data:
            # itemのコピーを作成し、explanationを削除
            new_item = item.copy()
            new_item.pop('explanation', None)
            processed_data.append(new_item)
        
        # 新しいファイルに保存
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed: {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_all_files():
    """
    全てのディレクトリのファイルを処理する関数
    """
    for set_dir in set_dirs:
        dir_path = os.path.join(base_dir, set_dir)
        
        # 入力ファイルパス
        test_input = os.path.join(dir_path, "test_100_explanained.json")
        train_input = os.path.join(dir_path, "train_400_explanained.json")
        
        # 出力ファイルパス
        test_output = os.path.join(dir_path, "test_100.json")
        train_output = os.path.join(dir_path, "train_400.json")
        
        # ファイルの処理
        print(f"\nProcessing files in {set_dir}:")
        if os.path.exists(test_input):
            process_json_file(test_input, test_output)
        else:
            print(f"Warning: {test_input} not found")
            
        if os.path.exists(train_input):
            process_json_file(train_input, train_output)
        else:
            print(f"Warning: {train_input} not found")

if __name__ == "__main__":
    # メイン処理の実行
    process_all_files()
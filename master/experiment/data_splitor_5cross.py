import json
import random
import os

# 入力ファイルパス
input_file = "master/data/Japanese_merged1000_added_index.json"

# 出力ディレクトリとファイル名の設定
output_dirs = [
    "master/experiment/1_set",
    "master/experiment/2_set",
    "master/experiment/3_set",
    "master/experiment/4_set",
    "master/experiment/5_set"
]

test_file = "test_200.json"
train_file = "train_800.json"

def create_cross_validation_sets():
    try:
        # JSONファイルの読み込み
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # データ数の検証
        if len(data) != 1000:
            raise ValueError(f"Expected 500 items, but got {len(data)} items")
                
        # データをランダムにシャッフル
        random.shuffle(data)
        
        # 200件ずつの5グループに分割
        chunks = [data[i:i+200] for i in range(0, 1000, 200)]
        
        # 各セットのディレクトリとファイルを作成
        for i, output_dir in enumerate(output_dirs):
            # ディレクトリが存在しない場合は作成
            os.makedirs(output_dir, exist_ok=True)
            
            # テストデータ（200件）
            test_data = chunks[i]
            
            # 訓練データ（残りの800件）
            train_data = []
            for j in range(5):
                if j != i:
                    train_data.extend(chunks[j])
            
            # テストデータの保存
            test_path = os.path.join(output_dir, test_file)
            with open(test_path, 'w', encoding='utf-8-sig') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            # 訓練データの保存
            train_path = os.path.join(output_dir, train_file)
            with open(train_path, 'w', encoding='utf-8-sig') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            print(f"Set {i+1} created successfully:")
            print(f"- Test data ({len(test_data)} items): {test_path}")
            print(f"- Train data ({len(train_data)} items): {train_path}")
            print("")
            
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input file")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# メイン処理の実行
if __name__ == "__main__":
    create_cross_validation_sets()
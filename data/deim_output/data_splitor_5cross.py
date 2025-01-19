import json
import random
import os

# 入力ファイルパス
input_file = "data/deim_output/Japanese_5cross_500_explanained.json"

# 出力ディレクトリとファイル名の設定
output_dirs = [
    "data/deim_output/1_set",
    "data/deim_output/2_set",
    "data/deim_output/3_set",
    "data/deim_output/4_set",
    "data/deim_output/5_set"
]

test_file = "test_100.json"
train_file = "train_400.json"

def create_cross_validation_sets():
    try:
        # JSONファイルの読み込み
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # データ数の検証
        if len(data) != 500:
            raise ValueError(f"Expected 500 items, but got {len(data)} items")
        
        # データをランダムにシャッフル
        random.shuffle(data)
        
        # 100件ずつの5グループに分割
        chunks = [data[i:i+100] for i in range(0, 500, 100)]
        
        # 各セットのディレクトリとファイルを作成
        for i, output_dir in enumerate(output_dirs):
            # ディレクトリが存在しない場合は作成
            os.makedirs(output_dir, exist_ok=True)
            
            # テストデータ（100件）
            test_data = chunks[i]
            
            # 訓練データ（残りの400件）
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

# # 検証用の関数
# def verify_data_split():
#     """データ分割の検証を行う関数"""
#     for i, output_dir in enumerate(output_dirs):
#         test_path = os.path.join(output_dir, test_file)
#         train_path = os.path.join(output_dir, train_file)
        
#         with open(test_path, 'r', encoding='utf-8') as f:
#             test_data = json.load(f)
#         with open(train_path, 'r', encoding='utf-8') as f:
#             train_data = json.load(f)
        
#         print(f"\nVerifying Set {i+1}:")
#         print(f"- Test data size: {len(test_data)}")
#         print(f"- Train data size: {len(train_data)}")
        
#         # データの重複チェック
#         test_ids = set(item['id'] for item in test_data)
#         train_ids = set(item['id'] for item in train_data)
#         intersection = test_ids.intersection(train_ids)
        
#         if intersection:
#             print(f"Warning: Found {len(intersection)} overlapping items in set {i+1}")
#         else:
#             print("- No overlapping items found")

# メイン処理の実行
if __name__ == "__main__":
    create_cross_validation_sets()
    # verify_data_split()
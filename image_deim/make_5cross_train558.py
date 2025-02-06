### 5分割したそれぞれのテストデータ18件に対し、検証用24を差し引いた、合計558件の訓練データ（検索対象データ）を作成する

import json
import os

# グローバル変数としてファイルパスを指定
base_dir = "image_deim"
main_data_path = os.path.join(base_dir, "Japanese_merged600_experiment_added_index copy.json")
validation_data_path = os.path.join(base_dir, "Japanese_merged24_image_change0_validation.json")

def load_json_file(file_path):
    """JSONファイルを読み込む関数"""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"エラー: {file_path} の読み込み中にエラーが発生しました。\n{str(e)}")
        return None

def get_ids_to_exclude(validation_path, test_path):
    """除外すべきIDのセットを取得する関数"""
    validation_data = load_json_file(validation_path)
    test_data = load_json_file(test_path)
    
    if validation_data is None or test_data is None:
        return None
    
    validation_ids = {item['id'] for item in validation_data}
    test_ids = {item['id'] for item in test_data}
    
    return validation_ids.union(test_ids)

def create_training_data(set_number):
    """各セット用の訓練データを作成する関数"""
    # テストデータのパスを構築
    set_dir = f"{set_number}_set"
    test_path = os.path.join(base_dir, set_dir, "test_18.json")
    
    # 除外すべきIDを取得
    exclude_ids = get_ids_to_exclude(validation_data_path, test_path)
    if exclude_ids is None:
        return False
    
    # メインデータを読み込む
    main_data = load_json_file(main_data_path)
    if main_data is None:
        return False
    
    # 訓練データを作成（IDを除外）
    train_data = []
    for item in main_data:
        if item['id'] not in exclude_ids:
            item_copy = item.copy()
            del item_copy['id']  # idラベルを削除
            train_data.append(item_copy)
    
    # 出力ディレクトリが存在することを確認
    output_dir = os.path.join(base_dir, set_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 訓練データを保存
    output_path = os.path.join(output_dir, "train_rag_558.json")
    try:
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"エラー: {output_path} の書き込み中にエラーが発生しました。\n{str(e)}")
        return False

def main():
    """メイン処理"""
    print("訓練データの作成を開始します...")
    
    for set_num in range(1, 6):
        print(f"\nセット {set_num} の処理中...")
        if create_training_data(set_num):
            print(f"セット {set_num} の訓練データを作成しました。")
        else:
            print(f"セット {set_num} の処理中にエラーが発生しました。")
    
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main()
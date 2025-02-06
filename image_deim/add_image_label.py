### "image"ラベルをとりあえず追加するスクリプト

import json

# グローバル変数として入力・出力ファイルのパスを指定
input_json_path = "image_deim/Japanese_merged600_experiment_added_index copy.json"
output_json_path = "image_deim/Japanese_merged600_experiment.json"

def add_image_field():
    # JSONファイルを読み込む
    with open(input_json_path, 'r', encoding='utf-8-sig') as f:
        input_data = json.load(f)
    
    # リスト内の各JSONオブジェクトを処理
    modified_data = []
    for item in input_data:
        # 新しい辞書を作成して順序を保持
        new_item = {}
        for i, (key, value) in enumerate(item.items()):
            if i == 1:  # idの後にimageを挿入
                new_item["image"] = "Yes"
            new_item[key] = value
        modified_data.append(new_item)
    
    # 変更したデータを新しいファイルに書き出す
    with open(output_json_path, 'w', encoding='utf-8-sig') as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)
    
    return modified_data

# 実行
try:
    modified_json = add_image_field()
    print(f"処理が完了しました。{len(modified_json)}件のデータを '{output_json_path}' に保存しました。")
    
    # 最初の1件を例として表示
    print("\n最初のデータの構造:")
    print(json.dumps(modified_json[0], indent=2, ensure_ascii=False))
    
except FileNotFoundError as e:
    print(f"エラー: ファイル '{e.filename}' が見つかりません。")
except json.JSONDecodeError:
    print("エラー: 入力JSONファイルの形式が正しくありません。")
except PermissionError:
    print("エラー: ファイルの読み書き権限がありません。")
except Exception as e:
    print(f"エラーが発生しました: {str(e)}")
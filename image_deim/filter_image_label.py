### "Japanese_image600_experiment.json"から"image": "Yes"のみを抽出するスクリプト"

import json

# グローバル変数として入力・出力ファイルのパスを指定
input_json_path = "image_deim/Japanese_merged600_experiment.json"
output_json_path = "image_deim/Japanese_merged441_NOimage.json"

def filter_image_yes():
    """
    imageラベルの値が"Yes"のデータのみを抽出して新しいJSONファイルに保存する関数
    """
    try:
        # JSONファイルを読み込む
        with open(input_json_path, 'r', encoding='utf-8-sig') as f:
            input_data = json.load(f)
        
        # # imageが"Yes"のデータのみを抽出
        # filtered_data = [item for item in input_data if item.get("image") == "Yes"]
        
        # imageが"No"のデータのみを抽出
        filtered_data = [item for item in input_data if item.get("image") == "No"]
        
        # フィルタリングされたデータを新しいファイルに書き出す
        with open(output_json_path, 'w', encoding='utf-8-sig') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        return filtered_data
    
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_json_path}' が見つかりません。")
        return None
    except json.JSONDecodeError:
        print("エラー: 入力JSONファイルの形式が正しくありません。")
        return None
    except PermissionError:
        print("エラー: ファイルの読み書き権限がありません。")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
        return None

def main():
    # フィルタリング実行
    filtered_json = filter_image_yes()
    
    if filtered_json is not None:
        total_count = len(filtered_json)
        print(f"処理が完了しました。")
        print(f"抽出されたデータ数: {total_count}")
        print(f"出力ファイル: {output_json_path}")
        
        # 最初の1件を例として表示（データが存在する場合）
        if total_count > 0:
            print("\n最初のデータの構造:")
            print(json.dumps(filtered_json[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
import json

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def save_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ファイル名を定義
labeled_file = "./data/processed/used_in_text-based/pdf_Korean_train_converted_selectedLabel.json"
original_file = "./data/raw/Korean_train.json"
output_file = "./data/processed/Korean_train_converted400.json"

# データを読み込む
labeled_data = load_json_file(labeled_file)
original_data = load_json_file(original_file)

# エントリ数が一致するか確認
if len(labeled_data) != len(original_data):
    raise ValueError("ラベル付きファイルと元のファイルのエントリ数が一致しません。")

# 'selected' ラベルを転送
for labeled_item, original_item in zip(labeled_data, original_data):
    original_item['selected'] = labeled_item['selected']

# 更新されたデータを新しいファイルに保存
save_json_file(output_file, original_data)

print(f"処理が完了しました。結果は {output_file} に保存されました。")

# 統計情報を表示
selected_count = sum(1 for item in original_data if item['selected'] == "Yes")
total_count = len(original_data)

print(f"総エントリ数: {total_count}")
print(f"'selected': 'Yes' のエントリ数: {selected_count}")
print(f"'selected': 'No' のエントリ数: {total_count - selected_count}")

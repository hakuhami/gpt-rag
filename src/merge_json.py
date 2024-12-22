import json

json_files = ['data/processed/French_sample_converted.json', 'data/processed/French_train_converted.json']
finally_output_json_file_path = 'data/processed/French_sample_and_train_converted.json'

# 結合されたデータを格納するリスト
merged_data = []

# 各JSONファイルを読み込み、データを結合
for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        merged_data.extend(data)

# 結合されたデータを新しいJSONファイルに書き込み
with open(finally_output_json_file_path, 'w', encoding='utf-8-sig') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

# 結合されたデータの行の塊の数を出力
print(f"出力された行の塊の数: {len(merged_data)}")
print(f"3つのJSONファイルが結合されました。")

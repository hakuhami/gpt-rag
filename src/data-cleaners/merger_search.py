import json

# 結合するJSONファイルのパスを指定
json_file_1 = './data/processed/Chinese_sample_converted.json'
json_file_2 = './data/processed/Chinese_train_converted.json'
output_file = './data/processed/Chinese_search_converted.json'

# 1つ目のJSONファイルを読み込み
with open(json_file_1, 'r', encoding='utf-8-sig') as file:
    data1 = json.load(file)

# 2つ目のJSONファイルを読み込み
with open(json_file_2, 'r', encoding='utf-8-sig') as file:
    data2 = json.load(file)

# 2つのデータを結合
combined_data = data1 + data2

# 結合されたデータを新しいJSONファイルに保存
with open(output_file, 'w', encoding='utf-8-sig') as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=2)

print(f"Number of records in the combined data: {len(combined_data)}")
import json
from collections import defaultdict

data_file = "Chinese_sample.json"
data_file_path = "./data/raw/Chinese_sample.json"
output_file_path = "./data/processed/statistics/Chinese_sample_statistics.json"

labels = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

label_counts = {label: defaultdict(int) for label in labels}

# データセットを読み込み
with open(data_file_path, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# データ数をプリントで出力
total_records = len(data)
print(f"Total number of records in the dataset: {total_records}")

# 各ラベルの値をカウント
for item in data:
    for label in labels:
        if label in item:
            label_counts[label][item[label]] += 1

# 辞書を通常のdictに変換して保存
label_counts = {label: dict(counts) for label, counts in label_counts.items()}

# 総数情報を追加
output_data = {
    "data_file": data_file,
    "total_records": total_records,
    "label_counts": label_counts
}

# 結果をJSONファイルに保存
with open(output_file_path, 'w', encoding='utf-8-sig') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"Label counts and total records have been saved to {output_file_path}.")

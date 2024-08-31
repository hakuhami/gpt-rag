import json
from collections import defaultdict

# 3つのJSONファイルのパスを指定
data_file = "All French data"
sample_file_path = "./data/processed/statistics/French_sample_statistics.json"
test_file_path = "./data/processed/statistics/French_test_statistics.json"
train_file_path = "./data/processed/statistics/French_train_statistics.json"
output_file_path = "./data/processed/statistics/French_all_statistics.json"  # 出力する総和のJSONファイルのパス

# データを初期化
total_counts = defaultdict(lambda: defaultdict(int))
total_records = 0

# 関数: 各ファイルからデータを読み込み、集計に加える
def accumulate_counts(file_path):
    global total_records
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        total_records += data["total_records"]
        for label, counts in data["label_counts"].items():
            for value, count in counts.items():
                total_counts[label][value] += count

# 3つのファイルからデータを読み込んで集計
accumulate_counts(sample_file_path)
accumulate_counts(test_file_path)
accumulate_counts(train_file_path)

# 結果を通常のdictに変換
total_counts = {label: dict(counts) for label, counts in total_counts.items()}

# 総和のデータを保存
output_data = {
    "data_file": data_file,
    "total_records": total_records,
    "label_counts": total_counts
}

# JSONファイルに保存
with open(output_file_path, 'w', encoding='utf-8-sig') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"Total label counts have been saved to {output_file_path}. Total records: {total_records}")
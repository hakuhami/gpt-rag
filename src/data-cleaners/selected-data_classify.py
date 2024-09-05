# 韓国語は先にテキストベースでtest:200, train:400とランダムサンプリングしているので、同じデータを使うべくどのデータが選択されたかをラベル付けする必要がある。

import json
from collections import Counter

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

# 元のファイルと抽出されたファイルの名前
original_file = "./data/processed/used_in_text-based/pdf_Korean_train_converted.json"
extracted_file = "./data/processed/used_in_text-based/pdf_Korean_train_removed400.json"
output_file = "./data/processed/used_in_text-based/pdf_Korean_train_converted_selectedLabel.json"

# データを読み込む
original_data = load_json_file(original_file)
extracted_data = load_json_file(extracted_file)

# 抽出されたデータの数を計測
extracted_count = len(extracted_data)

# 抽出されたデータの "data" 列の内容をカウンター（辞書）として保持
extracted_data_counter = Counter(item['data'] for item in extracted_data)

# 選択プロセスを追跡
selected_count = 0
duplicate_in_original = []

# 元のデータにラベルを付ける
for item in original_data:
    if item['data'] in extracted_data_counter and extracted_data_counter[item['data']] > 0:
        item['selected'] = "Yes"
        selected_count += 1
        extracted_data_counter[item['data']] -= 1
    else:
        item['selected'] = "No"
    
    # 元のデータ内の重複をチェック
    if sum(1 for x in original_data if x['data'] == item['data']) > 1:
        duplicate_in_original.append(item['data'])

# 更新されたデータを新しいファイルに書き込む
with open(output_file, 'w', encoding='utf-8-sig') as f:
    json.dump(original_data, f, ensure_ascii=False, indent=2)

print(f"処理が完了しました。結果は {output_file} に保存されました。")
print(f"抽出されたデータ数: {extracted_count}")
print(f"選択されたデータ数 ('selected': 'Yes'): {selected_count}")

if extracted_count == selected_count:
    print("抽出されたデータ数と選択されたデータ数が一致しています。")
else:
    print("警告: 抽出されたデータ数と選択されたデータ数が一致しません。")
    print(f"差異: {abs(extracted_count - selected_count)}")

    # 不一致の詳細を表示
    unmatched_extracted = [data for data, count in extracted_data_counter.items() if count > 0]
    unmatched_selected = [item['data'] for item in original_data if item['selected'] == "Yes" and item['data'] not in extracted_data]

    if unmatched_extracted:
        print(f"\n抽出されたが選択されなかったデータ数: {len(unmatched_extracted)}")
        print("例:")
        print(unmatched_extracted[0][:200] + "..." if len(unmatched_extracted[0]) > 200 else unmatched_extracted[0])

    if unmatched_selected:
        print(f"\n選択されたが抽出されていなかったデータ数: {len(unmatched_selected)}")
        print("例:")
        print(unmatched_selected[0][:200] + "..." if len(unmatched_selected[0]) > 200 else unmatched_selected[0])

    if duplicate_in_original:
        print(f"\n元のデータ内に重複が見つかりました。重複数: {len(duplicate_in_original)}")
        print("重複の例:")
        print(duplicate_in_original[0][:200] + "..." if len(duplicate_in_original[0]) > 200 else duplicate_in_original[0])

# ファイル内の重複をチェック
original_duplicates = [item for item, count in Counter(item['data'] for item in original_data).items() if count > 1]
extracted_duplicates = [item for item, count in Counter(item['data'] for item in extracted_data).items() if count > 1]

if original_duplicates:
    print(f"\n警告: 元のファイル内に重複したデータが {len(original_duplicates)} 件あります。")
if extracted_duplicates:
    print(f"\n警告: 抽出されたファイル内に重複したデータが {len(extracted_duplicates)} 件あります。")
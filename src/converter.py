# import json

# # Original English JSON file.
# input_file = './data/raw/PromiseEval_Sample_Trainset_Chinese.json'
# # New JSON file limited to only the labels needed for the experiment.
# output_file = './data/raw/Chinese_experiment_data.json'

# with open(input_file, 'r', encoding='utf-8-sig') as file:
#     data = json.load(file)

# # Remove labels that are not needed for the experiment.
# def remove_labels(record):
#     labels_to_remove = ["URL", "page_number"]
#     for label in labels_to_remove:
#         if label in record:
#             del record[label]
#     return record

# filtered_data = [remove_labels(record) for record in data]

# with open(output_file, 'w', encoding='utf-8-sig') as file:
#     json.dump(filtered_data, file, ensure_ascii=False, indent=2)

import json

# 入力ファイルと出力ファイルのパスを設定
input_file = './data/raw/Chinese_text-extracted-from-URL.json'
output_file = './data/raw/Chinese_experiment_data.json'

def decode_unicode_escape(text):
    """Unicodeエスケープシーケンスをデコードする関数"""
    if isinstance(text, str):
        return text.encode('utf-8-sig').decode('unicode_escape')
    return text

def process_record(record):
    """レコードを処理する関数：不要なラベルを削除し、文字列をデコードする"""
    # 削除するラベル
    labels_to_remove = ["URL", "page_number"]
    
    # 不要なラベルを削除
    for label in labels_to_remove:
        record.pop(label, None)
    
    # promise_stringとevidence_stringをデコード
    for key in ['promise_string', 'evidence_string']:
        if key in record and record[key] != 'N/A':
            record[key] = decode_unicode_escape(record[key])
    
    return record

# メイン処理
try:
    with open(input_file, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)

    # 各レコードを処理
    processed_data = [process_record(record) for record in data]

    # 処理したデータを新しいJSONファイルとして出力
    with open(output_file, 'w', encoding='utf-8-sig') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=2)

    print(f"処理が完了しました。出力ファイル: {output_file}")

except Exception as e:
    print(f"エラーが発生しました: {str(e)}")
    
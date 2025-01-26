# (01/26)解説文を新たに作り直したので、以前の5分割した内容に合わせる

import json

# グローバル変数の定義
original_file = 'data/deim_output/Japanese_5cross_500_explanained_JaAgain.json'
split_file = 'data/deim_output/5_set/test_100.json'
matched_output = 'data/deim_output/5_set/test_100_explanained_JaAgain.json'
unmatched_output = 'data/deim_output/5_set/train_400_explanained_JaAgain.json'

def split_data_by_match():
   with open(original_file, 'r', encoding='utf-8-sig') as f:
       original_data = json.load(f)
   with open(split_file, 'r', encoding='utf-8-sig') as f:
       split_data = json.load(f)
   
   # split_fileの順序を保持するための辞書を作成
   data_to_index = {item['data']: i for i, item in enumerate(split_data)}
   
   matched_data = []
   unmatched_data = []
   
   # まずoriginal_dataからマッチするデータを見つけて、一時的なリストに保存
   temp_matched = []
   for item in original_data:
       if item['data'] in data_to_index:
           temp_matched.append((data_to_index[item['data']], item))
       else:
           unmatched_data.append(item)
   
   # インデックスでソートしてから、itemだけを取り出す
   matched_data = [item for _, item in sorted(temp_matched)]
   
   with open(matched_output, 'w', encoding='utf-8-sig') as f:
       json.dump(matched_data, f, ensure_ascii=False, indent=2)
   with open(unmatched_output, 'w', encoding='utf-8-sig') as f:
       json.dump(unmatched_data, f, ensure_ascii=False, indent=2)
   
   return len(matched_data), len(unmatched_data)

matches, unmatches = split_data_by_match()
print(f"Matched: {matches}, Unmatched: {unmatches}")
import json
from urllib.parse import urlparse
from collections import OrderedDict

url_domain_to_pdf_file = "./data/raw/PDFs/.Korean_URL-PDF_List.json"
data_file_path = "./data/raw/Korean_sample.json"
output_file_path = "./data/processed/pdf_Korean_sample.json"

# ドメイン名と対応するPDFファイル名のマッピングを読み込む
with open(url_domain_to_pdf_file, 'r', encoding='utf-8-sig') as f:
    domain_to_pdf = json.load(f)

# データセットを読み込む
with open(data_file_path, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# データセットの各項目に対してPDF名を追加
for item in data:
    url = item.get('URL')
    if url:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # ドメイン名に対応するPDFファイル名を取得して追加
        if domain in domain_to_pdf:
            item['pdf'] = domain_to_pdf[domain]
    
    # URL列を削除
    if 'URL' in item:
        del item['URL']

# 順序を変更してpdf列を最初に
for i, item in enumerate(data):
    # pdf列を先頭に移動させたOrderedDictを作成
    ordered_item = OrderedDict([('pdf', item['pdf'])])
    for key, value in item.items():
        if key != 'pdf':
            ordered_item[key] = value
    # 元のリストの要素を置き換え
    data[i] = ordered_item

print(f"data amount:{len(data)}")

# 新しいデータセットを保存
with open(output_file_path, 'w', encoding='utf-8-sig') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

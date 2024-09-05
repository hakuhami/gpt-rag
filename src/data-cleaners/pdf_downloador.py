# 韓国語は先にテキストベースでtest:200, train:400とランダムサンプリングしているので、それらのデータからダウンロード(先生にやってもらってテキスト抽出できなかったpdf(2件)は除去)

import json
import requests
import os
from urllib.parse import urlparse

# # JSONファイルのパスとPDFを保存するディレクトリの設定
# json_file_path = "./data/raw/Korean_train.json"
# save_dir = "./data/raw/PDFs"
# output_json_file = "./data/raw/PDFs/.Korean_URL-PDF_List.json"

# JSONファイルのパスとPDFを保存するディレクトリの設定
json_file_path = "./data/processed/Korean_train_selectedLabel.json"
save_dir = "./data/raw/PDFs"
output_json_file = "./data/raw/PDFs/.Korean_URL-PDF_List.json"

# JSONデータの読み込み
with open(json_file_path, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# ユニークなURLを格納するためのセットを作成
unique_urls = set()

# # URLの種類をすべて取得するループ
# for item in data:
#     url = item.get('URL')
#     if url and url not in unique_urls:
#         unique_urls.add(url)

# URLの種類をすべて取得するループ（"selected": "Yes" のみ）
for item in data:
    if item.get('selected') == "Yes":
        url = item.get('URL')
        if url and url not in unique_urls:
            unique_urls.add(url)

# URLリストをリスト形式に変換
url_list = sorted(list(unique_urls))
print("Korean_train.json URL_List:")
for i, url in enumerate(url_list):
    print(f"{i+1}: {url}")

# URLリストを辞書形式に変換
url_dict = {url: f"company{i+1}.pdf" for i, url in enumerate(url_list)}

# 辞書をJSONファイルに保存
with open(output_json_file, 'w', encoding='utf-8-sig') as f:
    json.dump(url_dict, f, ensure_ascii=False, indent=2)

# PDFをダウンロードして保存する関数
def download_pdfs(url_list, save_directory):
    for i, url in enumerate(url_list):
        response = requests.get(url)
        pdf_filename = f'company{i+1}.pdf'  # ファイル名を設定
        pdf_path = os.path.join(save_directory, pdf_filename)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {pdf_filename}")

# PDFをダウンロード
download_pdfs(url_list, save_dir)
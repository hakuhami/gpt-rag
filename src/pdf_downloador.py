import json
import requests
import os

# JSONファイルのパスとPDFを保存するディレクトリの設定
json_file_path = "./data/raw/Chinese_sample.json"
save_dir = "./data/raw"

# JSONデータの読み込み
with open(json_file_path, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# ユニークなURLを格納するためのセットを作成
unique_urls = set()

# URLの種類が9件になるまでループ
for item in data:
    url = item.get('URL')
    if url and url not in unique_urls:
        unique_urls.add(url)
    if len(unique_urls) == 9:
        break

# URLリストをリスト形式に変換
url_list = sorted(list(unique_urls))
print("Chinese_train.json URL_List:")
for i, url in enumerate(url_list):
    print(f"{i+1}: {url}")

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

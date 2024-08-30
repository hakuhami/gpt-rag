import json
import requests
import os
from urllib.parse import urlparse

# JSONファイルのパスとPDFを保存するディレクトリの設定
json_file_path = "./data/raw/Chinese_sample.json"
save_dir = "./data/raw"
output_json_file = "./data/raw/PDFs/.Chinese_URL-PDF_List.json"

# ドメイン名と対応するPDFファイル名のマッピング
url_domain_to_pdf = {
    "download.geniusnet.com": "KYE.pdf",
    "drive.google.com": "NPC.pdf",
    "esg.tsmc.com": "TSMC.pdf",
    "fpcc-esg.com": "FPCC.pdf",
    "www.alchip.com": "Alchip.pdf",
    "www.pegavision.com": "pegavision.pdf",
    "www.psi.com.tw": "PSI.pdf",
    "www.scinopharm.com.tw": "SPT.pdf",
    "www.standard.com.tw": "Standard.pdf",
    "www.taipeigas.com.tw": "GTG.pdf"
}

# JSONデータの読み込み
with open(json_file_path, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# 出力済みのドメイン名を追跡するためのセット
processed_domains = set()

# URLリストを辞書形式に変換
url_dict = {}

for item in data:
    url = item.get('URL')
    if url:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # まだ処理されていないドメイン名の場合のみPDFファイル名を割り当てる
        if domain in url_domain_to_pdf and domain not in processed_domains:
            url_dict[domain] = url_domain_to_pdf[domain]
            processed_domains.add(domain)
            
# 辞書をJSONファイルに保存
with open(output_json_file, 'w', encoding='utf-8-sig') as f:
    json.dump(url_dict, f, ensure_ascii=False, indent=2)

# # ユニークなURLを格納するためのセットを作成
# unique_urls = set()

# # URLの種類をすべて取得するループ
# for item in data:
#     url = item.get('URL')
#     if url and url not in unique_urls:
#         unique_urls.add(url)

# # URLリストをリスト形式に変換
# url_list = sorted(list(unique_urls))
# print("Chinese_train.json URL_List:")
# for i, url in enumerate(url_list):
#     print(f"{i+1}: {url}")

# # URLリストを辞書形式に変換
# url_dict = {url: f"company{i+1}.pdf" for i, url in enumerate(url_list)}

# # 辞書をJSONファイルに保存
# with open(output_json_file, 'w', encoding='utf-8-sig') as f:
#     json.dump(url_dict, f, ensure_ascii=False, indent=2)

# # PDFをダウンロードして保存する関数
# def download_pdfs(url_list, save_directory):
#     for i, url in enumerate(url_list):
#         response = requests.get(url)
#         pdf_filename = f'company{i+1}.pdf'  # ファイル名を設定
#         pdf_path = os.path.join(save_directory, pdf_filename)
#         with open(pdf_path, 'wb') as f:
#             f.write(response.content)
#         print(f"Downloaded: {pdf_filename}")

# # PDFをダウンロード
# download_pdfs(url_list, save_dir)

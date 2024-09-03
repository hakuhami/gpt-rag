import json
import requests
import os
from urllib.parse import urlparse

# JSONファイルのパスとPDFを保存するディレクトリの設定
json_file_path = "./data/raw/Korean_sample.json"
save_dir = "./data/raw"
output_json_file = "./data/raw/PDFs/.Korean_URL-PDF_List.json"

# ドメイン名と対応するPDFファイル名のマッピング
# ↓中国語データのリストになっているので、韓国語データに合わせる
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

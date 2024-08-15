import json
import requests
import certifi
import io
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from io import StringIO
from PyPDF2 import PdfReader

def download_pdf(url):
    response = requests.get(url, verify=certifi.where())
    return io.BytesIO(response.content)

def extract_text_from_page(pdf_file, page_number):
    reader = PdfReader(pdf_file, strict=False)
    if page_number <= len(reader.pages):
        page = reader.pages[page_number - 1]  # PyPDF2はページを0から数えるため、1を引く
        return page.extract_text()
    else:
        return "ページが存在しません"
    
# def extract_text_from_page(pdf_file, page_number):
#     output_string = StringIO()
#     laparams = LAParams()
#     extract_text_to_fp(pdf_file, output_string, page_numbers=[page_number-1], laparams=laparams)
#     text = output_string.getvalue()
#     return text if text.strip() else "ページが存在しないか、テキストを抽出できません"

def process_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)

    for item in data:
        url = item['URL']
        page_number = int(item['page_number'])        
        print(f"処理中: {url}, ページ {page_number}")
        
        try:
            pdf_file = download_pdf(url)
            text = extract_text_from_page(pdf_file, page_number)
            item['data'] = text
            print("○テキスト抽出が成功しました")

        except Exception as e:
            print(f"×エラーが発生しました: {url}, ページ {page_number}")
            print(f"×エラー内容: {str(e)}")
            item['data'] = f"テキスト抽出に失敗しました。エラー内容: {str(e)}"

    output_file_path = './data/raw/Chinese_experiment_data-2.json'
    with open(output_file_path, 'w', encoding='utf-8-sig') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"更新されたデータが {output_file_path} に保存されました。")

json_file_path = './data/raw/PromiseEval_Sample_Trainset_Chinese.json'
process_json_data(json_file_path)
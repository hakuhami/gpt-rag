# import json
# import os
# from PyPDF2 import PdfReader
# import glob

# # 入力JSONファイルと出力JSONファイルのパス
# input_json_path = './data/processed/pdf_Chinese_sample.json'
# output_json_path = './data/processed/pdf_Chinese_sample_extracted.json'

# def extract_text_from_pdf(pdf_path, page_number):
#     reader = PdfReader(pdf_path)
#     page = reader.pages[page_number - 1]  # PyPDF2はページ番号が0から始まるため、1を引く
#     return page.extract_text()

# def process_json(input_file, output_file):
#     # JSONファイルを読み込む
#     with open(input_file, 'r', encoding='utf-8-sig') as f:
#         data = json.load(f)

#     # PDFファイルが格納されているディレクトリ
#     pdf_dir = "./data/raw/PDFs"

#     # 各エントリを処理
#     for entry in data:
#         pdf_name = entry['pdf']
#         page_number = int(entry['page_number'])
#         # PDFファイルのフルパスを取得
#         pdf_path = os.path.join(pdf_dir, pdf_name)
#         # PDFからテキストを抽出
#         text = extract_text_from_pdf(pdf_path, page_number)
#         # "data"列を追加
#         entry['data'] = text
#         print(f"data: {text}")

#     # 更新されたデータを新しいJSONファイルに保存
#     with open(output_file, 'w', encoding='utf-sig') as f:
#         json.dump(data, f, indent=2)

# process_json(input_json_path, output_json_path)

# import json
# import os
# import pdfplumber

# # 入力JSONファイルと出力JSONファイルのパス
# input_json_path = './data/processed/pdf_Chinese_sample.json'
# output_json_path = './data/processed/pdf_Chinese_sample_extracted.json'

# def extract_text_from_pdf(pdf_path, page_number):
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             if page_number <= len(pdf.pages):
#                 page = pdf.pages[page_number - 1]
#                 return page.extract_text()
#             else:
#                 return f"Error: Page {page_number} does not exist in the PDF."
#     except Exception as e:
#         return f"Error extracting text from PDF: {str(e)}"

# def process_json(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8-sig') as f:
#         data = json.load(f)

#     pdf_dir = "./data/raw/PDFs"

#     for entry in data:
#         pdf_name = entry['pdf']
#         page_number = int(entry['page_number'])
#         pdf_path = os.path.join(pdf_dir, pdf_name)

#         if os.path.exists(pdf_path):
#             text = extract_text_from_pdf(pdf_path, page_number)
#             entry['data'] = text
#         else:
#             entry['data'] = f"Text extraction failed."
#         print(f"data: {entry['data']}")

#     with open(output_file, 'w', encoding='utf-8-sig') as f:
#         json.dump(data, f, indent=2)

# process_json(input_json_path, output_json_path)

import json
import os
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

input_json_path = './data/processed/pdf_Chinese_test.json'
output_json_path = './data/processed/pdf_Chinese_test_extracted.json'

def extract_text_from_pdf(pdf_path, page_number):
    try:
        laparams = LAParams()
        text = extract_text(pdf_path, page_numbers=[page_number-1], laparams=laparams)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    pdf_dir = "./data/raw/PDFs"

    for entry in data:
        pdf_name = entry['pdf']
        page_number = int(entry['page_number'])
        pdf_path = os.path.join(pdf_dir, pdf_name)

        if os.path.exists(pdf_path):
            text = extract_text_from_pdf(pdf_path, page_number)
            entry['data'] = text
        else:
            entry['data'] = f"Text extraction failed."
        print(f"data: {entry['data'][:100]}...")  # 最初の100文字だけ表示

    with open(output_file, 'w', encoding='utf-8-sig') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"lengs of data: {len(data)}")

process_json(input_json_path, output_json_path)

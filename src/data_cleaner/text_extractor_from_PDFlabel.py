import json
import os
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

input_json_path = './data/processed/pdf_Korean_sample.json'
output_json_path = './data/processed/pdf_Korean_sample_extracted.json'

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
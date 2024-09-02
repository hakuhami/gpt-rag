import json
import requests
import certifi
import io
from PyPDF2 import PdfReader

def download_pdf(url):
    response = requests.get(url, verify=certifi.where())
    return io.BytesIO(response.content)

def extract_text_from_page(pdf_file, page_number):
    reader = PdfReader(pdf_file, strict=False)
    if page_number <= len(reader.pages):
        # Since PyPDF2 counts pages starting from 0, subtract 1.
        page = reader.pages[page_number - 1]
        return page.extract_text()
    else:
        return "The page does not exist."

def process_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)

    for item in data:
        url = item['URL']
        page_number = int(item['page_number'])        
        print(f"processing: {url}, page: {page_number}")
        
        try:
            pdf_file = download_pdf(url)
            text = extract_text_from_page(pdf_file, page_number)
            item['data'] = text
            print("○ Text extraction successed.")

        except Exception as e:
            print(f"× Text extraction failed. Error : {str(e)}")
            item['data'] = f"Text extraction failed."

    output_file_path = './data/processed/Chinese_test_extracted.json'
    with open(output_file_path, 'w', encoding='utf-8-sig') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"The updated JSON data has been saved to {output_file_path}.")

json_file_path = './data/raw/Chinese_test.json'
process_json_data(json_file_path)
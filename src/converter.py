import json
from collections import OrderedDict

# Original JSON file.
input_file = './data/raw/Chinese_sample.json'
# PDF URL list JSON file.
url_list_file = './data/raw/Chinese_sample_URL_List.json'
# New JSON file limited to only the labels needed for the experiment.
output_file = './data/raw/Chinese_sample_converted.json'

with open(input_file, 'r', encoding='utf-8-sig') as file:
    data = json.load(file)
    
with open(url_list_file, 'r', encoding='utf-8-sig') as file:
    url_list = json.load(file)

# Convert the URL to a PDF file name and store it in the 'company_PDF' key.
for item in data:
    url = item.get('URL')
    if url in url_list.values():
        pdf_name = [key for key, value in url_list.items() if value == url][0]
        item['company_PDF'] = pdf_name

# Remove labels that are not needed for the experiment.
def remove_labels(record):
    labels_to_remove = ["URL"]
    for label in labels_to_remove:
        if label in record:
            del record[label]
    return record
        
# Move the 'company_PDF' key to the top.
def order_data(records):
    ordered_records = []
    for record in records:
        if 'company_PDF' not in record:
            print(f"Skipping record: {record}")
            continue
        
        cleaned_record = remove_labels(record)
           
        # Create a new OrderedDict with 'data' key first
        ordered_record = OrderedDict()
        ordered_record['company_PDF'] = cleaned_record['company_PDF']
        for key, value in record.items():
            if key != 'company_PDF':  # Skip the 'company_PDF' key since it's already added
                ordered_record[key] = value
        
        ordered_records.append(ordered_record)
    
    return ordered_records

ordered_data = order_data(data)

print(f"Number of records: {len(ordered_data)}")

with open(output_file, 'w', encoding='utf-8-sig') as f:
    json.dump(ordered_data, f, ensure_ascii=False, indent=2)
    
import json
from collections import OrderedDict

# Original English JSON file.
input_file = './data/raw/Chinese_experiment_data_test20240819.json'
# New JSON file limited to only the labels needed for the experiment.
output_file = './data/raw/Chinese_experiment_data_test20240819-Text-Extracted.json'

with open(input_file, 'r', encoding='utf-8-sig') as file:
    data = json.load(file)

# Remove labels that are not needed for the experiment.
def remove_labels(record):
    labels_to_remove = ["URL", "page_number"]
    for label in labels_to_remove:
        if label in record:
            del record[label]
    return record

# Filter out records where "data" key has the value "Text extraction failed."
def filter_and_order_data(records):
    filtered_records = []
    for record in records:
        if record.get('data') != "Text extraction failed.":
            # Remove unnecessary labels
            cleaned_record = remove_labels(record)
            
            # Create a new OrderedDict with 'data' key first
            ordered_record = OrderedDict()
            ordered_record['data'] = cleaned_record['data']
            for key, value in cleaned_record.items():
                if key != 'data':  # Skip the 'data' key since it's already added
                    ordered_record[key] = value
            
            filtered_records.append(ordered_record)
    
    return filtered_records

filtered_data = filter_and_order_data(data)

print(f"Number of records: {len(filtered_data)}")

with open(output_file, 'w', encoding='utf-8-sig') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=2)
    
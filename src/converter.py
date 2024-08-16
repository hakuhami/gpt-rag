import json

# Original English JSON file.
input_file = './data/raw/Chinese_experiment_data.json'
# New JSON file limited to only the labels needed for the experiment.
output_file = './data/raw/Chinese_experiment_data-Text-Extracted.json'

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
filtered_data = [
    remove_labels(record) 
    for record in data 
    if record.get('data') != "Text extraction failed."
]
print(f"Number of records: {len(filtered_data)}")

with open(output_file, 'w', encoding='utf-8-sig') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=2)
    
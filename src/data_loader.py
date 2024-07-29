import json
from typing import List, Dict

def save_json_data(data: List[Dict], file_path: str) -> None:
    """
    Save data as a JSON file

    Args:
        data (List[Dict]): Data to be saved
        file_path (str): File path for saving the data
    """
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json_data(file_path: str) -> List[Dict]:
    """
    Load data from a JSON file

    Args:
        file_path (str): File path for loading the data

    Returns:
        List[Dict]: Loaded data
    """
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)
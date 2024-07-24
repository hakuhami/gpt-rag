import pandas as pd
import json
from typing import List, Dict

# def load_annotated_data_from_excel(file_path: str) -> pd.DataFrame:
#     """
#     エクセルファイルからアノテーションデータを読み込む

#     Args:
#         file_path (str): エクセルファイルのパス

#     Returns:
#         pd.DataFrame: アノテーションデータのDataFrame
#     """
#     return pd.read_excel(file_path)

def save_json_data(data: List[Dict], file_path: str) -> None:
    """
    データをJSONファイルとして保存する

    Args:
        data (List[Dict]): 保存するデータ
        file_path (str): 保存先のファイルパス
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json_data(file_path: str) -> List[Dict]:
    """
    JSONファイルからデータを読み込む

    Args:
        file_path (str): 読み込むJSONファイルのパス

    Returns:
        List[Dict]: 読み込んだデータ
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
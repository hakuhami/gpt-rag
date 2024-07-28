import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

# def preprocess_data(df: pd.DataFrame) -> List[Dict]:
#     """
#     DataFrameをJSONフォーマットに変換する

#     Args:
#         df (pd.DataFrame): 入力のDataFrame

#     Returns:
#         List[Dict]: JSON形式のデータ
#     """
#     json_data = []
#     for _, row in df.iterrows():
#         item = {
#             "paragraph": row['パラグラフ中の文章内容'],
#             "commitment_present": int(row['公約が含まれるか']),
#             "commitment_text": row['公約の具体的な箇所'] if row['公約が含まれるか'] == 1 else None,
#             "commitment_timing": int(row['公約を検証できるタイミング']) if row['公約が含まれるか'] == 1 else None,
#             "evidence_present": int(row['根拠が含まれるか']) if row['公約が含まれるか'] == 1 else None,
#             "evidence_text": row['根拠の箇所'] if row['公約が含まれるか'] == 1 and row['根拠が含まれるか'] == 1 else None,
#             "relation_quality": int(row['公約と根拠の関係の質']) if row['公約が含まれるか'] == 1 and row['根拠が含まれるか'] == 1 else None
#         }
#         json_data.append(item)
#     return json_data

def split_data(data: List[Dict], test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into search and test sets

    Args:
        data (List[Dict]): Data to be split
        test_size (float): Proportion of test data
        random_state (int): Random seed

    Returns:
        Tuple[List[Dict], List[Dict]]: Search data and test data
    """
    search_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return search_data, test_data
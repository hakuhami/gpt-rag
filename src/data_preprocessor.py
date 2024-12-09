from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

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
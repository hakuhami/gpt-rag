import pytest
import pandas as pd
from src.data_preprocessor import preprocess_data, split_data

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'パラグラフ中の文章内容': ['テスト文章1', 'テスト文章2'],
        '公約が含まれるか': [1, 0],
        '公約の具体的な箇所': ['テスト公約', None],
        '公約を検証できるタイミング': [1, None],
        '根拠が含まれるか': [1, None],
        '根拠の箇所': ['テスト根拠', None],
        '公約と根拠の関係の質': [1, None]
    })

def test_preprocess_data(sample_dataframe):
    json_data = preprocess_data(sample_dataframe)
    assert len(json_data) == 2
    assert json_data[0]['paragraph'] == 'テスト文章1'
    assert json_data[0]['commitment_present'] == 1
    assert json_data[0]['commitment_text'] == 'テスト公約'
    assert json_data[1]['commitment_present'] == 0
    assert json_data[1]['commitment_text'] is None

def test_split_data():
    data = [{'id': i} for i in range(100)]
    search_data, test_data = split_data(data, test_size=0.2, random_state=42)
    assert len(search_data) == 80
    assert len(test_data) == 20
    assert set(range(100)) == set(item['id'] for item in search_data + test_data)
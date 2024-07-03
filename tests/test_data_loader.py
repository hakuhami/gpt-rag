import pytest
import pandas as pd
import json
import os
from src.data_loader import load_annotated_data_from_excel, save_json_data, load_json_data

@pytest.fixture
def sample_excel_file(tmp_path):
    df = pd.DataFrame({
        'パラグラフ中の文章内容': ['テスト文章1', 'テスト文章2'],
        '公約が含まれるか': [1, 0],
        '公約の具体的な箇所': ['テスト公約', None],
        '公約を検証できるタイミング': [1, None],
        '根拠が含まれるか': [1, None],
        '根拠の箇所': ['テスト根拠', None],
        '公約と根拠の関係の質': [1, None]
    })
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False)
    return file_path

@pytest.fixture
def sample_json_data():
    return [
        {"key1": "value1"},
        {"key2": "value2"}
    ]

def test_load_annotated_data_from_excel(sample_excel_file):
    df = load_annotated_data_from_excel(sample_excel_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'パラグラフ中の文章内容' in df.columns

def test_save_and_load_json_data(tmp_path, sample_json_data):
    file_path = tmp_path / "test_data.json"
    save_json_data(sample_json_data, file_path)
    assert os.path.exists(file_path)

    loaded_data = load_json_data(file_path)
    assert loaded_data == sample_json_data
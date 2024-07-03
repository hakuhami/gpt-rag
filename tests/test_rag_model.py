import pytest
from unittest.mock import patch, MagicMock
from src.rag_model import RAGModel

@pytest.fixture
def sample_search_data():
    return [
        {
            "paragraph": "当社は2030年までにCO2排出量を50%削減する目標を掲げています。",
            "commitment_present": 1,
            "commitment_text": "2030年までにCO2排出量を50%削減する",
            "commitment_timing": 0,
            "evidence_present": 0,
            "evidence_text": None,
            "relation_quality": None
        },
        {
            "paragraph": "2022年度の再生可能エネルギー使用率は30%に達しました。",
            "commitment_present": 0,
            "commitment_text": None,
            "commitment_timing": None,
            "evidence_present": None,
            "evidence_text": None,
            "relation_quality": None
        }
    ]

@pytest.fixture
def rag_model():
    with patch('src.rag_model.openai'):
        model = RAGModel(api_key="dummy_key")
        model.prepare_documents(sample_search_data())
        return model

def test_prepare_documents(rag_model, sample_search_data):
    assert len(rag_model.search_data) == len(sample_search_data)
    assert len(rag_model.documents) == len(sample_search_data)
    assert rag_model.doc_embeddings.shape[0] == len(sample_search_data)

def test_get_relevant_context(rag_model):
    query = "CO2排出量削減目標"
    relevant_docs = rag_model.get_relevant_context(query, top_k=1)
    assert len(relevant_docs) == 1
    assert "CO2排出量" in relevant_docs[0]['paragraph']

@patch('src.rag_model.openai.ChatCompletion.create')
def test_analyze_paragraph(mock_create, rag_model):
    mock_create.return_value = MagicMock(choices=[MagicMock(message={'content': '{"commitment_present": 1, "commitment_text": "テスト公約", "commitment_timing": 1, "evidence_present": 0, "evidence_text": null, "relation_quality": null}'})])

    paragraph = "テスト文章"
    result = rag_model.analyze_paragraph(paragraph)

    assert result['commitment_present'] == 1
    assert result['commitment_text'] == "テスト公約"
    assert result['commitment_timing'] == 1
    assert result['evidence_present'] == 0
    assert result['evidence_text'] is None
    assert result['relation_quality'] is None

    mock_create.assert_called_once()
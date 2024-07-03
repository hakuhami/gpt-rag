import pytest
from src.evaluator import calculate_f1_scores

@pytest.fixture
def sample_true_data():
    return [
        {"commitment_present": 1, "commitment_timing": 1, "evidence_present": 1, "relation_quality": 1},
        {"commitment_present": 0, "commitment_timing": None, "evidence_present": None, "relation_quality": None},
        {"commitment_present": 1, "commitment_timing": 0, "evidence_present": 0, "relation_quality": None}
    ]

@pytest.fixture
def sample_pred_data():
    return [
        {"commitment_present": 1, "commitment_timing": 1, "evidence_present": 1, "relation_quality": 0},
        {"commitment_present": 0, "commitment_timing": None, "evidence_present": None, "relation_quality": None},
        {"commitment_present": 1, "commitment_timing": 1, "evidence_present": 1, "relation_quality": 1}
    ]

def test_calculate_f1_scores(sample_true_data, sample_pred_data):
    f1_scores = calculate_f1_scores(sample_true_data, sample_pred_data)

    assert 'commitment_present' in f1_scores
    assert 'commitment_timing' in f1_scores
    assert 'evidence_present' in f1_scores
    assert 'relation_quality' in f1_scores

    assert 0 <= f1_scores['commitment_present'] <= 1
    assert 0 <= f1_scores['commitment_timing'] <= 1
    assert 0 <= f1_scores['evidence_present'] <= 1
    assert 0 <= f1_scores['relation_quality'] <= 1

    # 完全に一致する'commitment_present'のF1スコアは1.0になるはず
    assert f1_scores['commitment_present'] == 1.0
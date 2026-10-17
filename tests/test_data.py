import pandas as pd
import pytest

from src.data import read_dataset
from src.config import FEATURES


def test_binary_target_mapping(tmp_path):
    source = tmp_path / "tiny.csv"
    targets = [0, 1, 2, 3, 4] * 12
    rows = [
        [55, 1, 0, 130, 230, 0, 1, 150, 0, 1.0, 1, 0, 2, target]
        for target in targets
    ]
    pd.DataFrame(rows).to_csv(source, index=False, header=False)
    result = read_dataset(source)
    assert result["target"].tolist()[:5] == [0, 1, 1, 1, 1]
    assert list(result.columns) == FEATURES + ["target"]


def test_invalid_column_count_rejected(tmp_path):
    source = tmp_path / "invalid.csv"
    pd.DataFrame([[1, 2, 3]]).to_csv(source, index=False, header=False)
    with pytest.raises(ValueError):
        read_dataset(source)

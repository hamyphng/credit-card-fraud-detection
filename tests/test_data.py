import pandas as pd
import pytest

from src.data_loader import load_data


def test_load_data_reads_csv(tmp_path):
    filepath = tmp_path / "transactions.csv"
    expected = pd.DataFrame({"Amount": [10.0, 20.0], "Class": [0, 1]})
    expected.to_csv(filepath, index=False)

    actual = load_data(str(filepath))

    pd.testing.assert_frame_equal(actual, expected)


def test_load_data_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / "missing.csv"))

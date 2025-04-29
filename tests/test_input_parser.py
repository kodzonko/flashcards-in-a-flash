import pathlib

import pandas as pd
import pytest
from pandas.errors import EmptyDataError

from flashcards_in_a_flash.input_parser import (
    EMPTY_CSV_ERROR,
    merge_dataframes,
    parse_csv,
)


@pytest.fixture
def valid_csv_path(tmp_path):
    """Create a temporary valid CSV file for testing."""
    csv_content = "question;answer\nfood;cibus\napple;malum"
    csv_file = tmp_path / "valid_flashcards.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def empty_csv_path(tmp_path):
    """Create a temporary empty CSV file for testing."""
    csv_file = tmp_path / "empty_flashcards.csv"
    csv_file.write_text("question;answer")
    return csv_file


@pytest.fixture
def completely_empty_csv_path(tmp_path):
    """Create a temporary completely empty CSV file for testing."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    return csv_file


@pytest.fixture
def duplicate_entries_csv_path(tmp_path):
    """Create a temporary CSV file with duplicate entries."""
    csv_content = "native;learning\nfood;cibus\napple;malum\nfood;cibus"
    csv_file = tmp_path / "duplicate_flashcards.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def df1():
    """Create a sample DataFrame for testing merge_dataframes."""
    return pd.DataFrame(
        {"native": ["apple", "banana"], "learning": ["malum", "banana"]}
    )


@pytest.fixture
def df2():
    """Create another sample DataFrame for testing merge_dataframes."""
    return pd.DataFrame(
        {"native": ["banana", "cherry"], "learning": ["banana", "cerasus"]}
    )


@pytest.fixture
def df_with_nan():
    """Create a DataFrame with NaN values."""
    return pd.DataFrame({"native": ["apple", None], "learning": ["malum", "missing"]})


@pytest.fixture
def malformed_csv_path(tmp_path):
    """Create a temporary malformed CSV file that will trigger EmptyDataError."""
    csv_file = tmp_path / "malformed.csv"
    # Create a file with only newlines but no data
    csv_file.write_text("\n\n\n")
    return csv_file


def test_parse_csv_valid(valid_csv_path):
    """Test parsing a valid CSV file."""
    result = parse_csv(valid_csv_path)

    # Check if result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check the content of the DataFrame
    assert len(result) == 2
    assert list(result.columns) == ["native", "learning"]
    assert result.iloc[0]["native"] == "food"
    assert result.iloc[0]["learning"] == "cibus"
    assert result.iloc[1]["native"] == "apple"
    assert result.iloc[1]["learning"] == "malum"


def test_parse_csv_empty(empty_csv_path):
    """Test parsing an empty CSV file raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        parse_csv(empty_csv_path)

    assert EMPTY_CSV_ERROR in str(excinfo.value)


def test_parse_csv_completely_empty(completely_empty_csv_path):
    """Test parsing a completely empty CSV file raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        parse_csv(completely_empty_csv_path)

    assert EMPTY_CSV_ERROR in str(excinfo.value)


def test_parse_csv_removes_duplicates(duplicate_entries_csv_path):
    """Test that parse_csv removes duplicate entries."""
    result = parse_csv(duplicate_entries_csv_path)

    # Check if duplicates were removed
    assert len(result) == 2
    assert result["native"].value_counts()["food"] == 1


def test_parse_csv_invalid_path():
    """Test parsing a non-existent CSV file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_csv(pathlib.Path("nonexistent_file.csv"))


def test_parse_csv_empty_data_error(malformed_csv_path, monkeypatch):
    """Test that EmptyDataError from pandas is properly handled."""

    # Mock pd.read_csv to raise EmptyDataError
    def mock_read_csv(*args, **kwargs):
        raise EmptyDataError("No columns to parse from file")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    with pytest.raises(ValueError) as excinfo:
        parse_csv(malformed_csv_path)

    assert EMPTY_CSV_ERROR in str(excinfo.value)


def test_merge_dataframes_valid(df1, df2):
    """Test merging two valid DataFrames."""
    result = merge_dataframes(df1, df2)

    # Check if result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check the content of the DataFrame
    assert len(result) == 3
    assert set(result["native"]) == {"apple", "banana", "cherry"}
    assert set(result["learning"]) == {"malum", "banana", "cerasus"}


def test_merge_dataframes_with_nan(df1, df_with_nan):
    """Test merging DataFrames with NaN values raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        merge_dataframes(df1, df_with_nan)

    assert "contains empty cells" in str(excinfo.value)


def test_merge_dataframes_identical(df1):
    """Test merging identical DataFrames."""
    result = merge_dataframes(df1, df1.copy())

    # Check if result is identical to original
    assert len(result) == len(df1)
    pd.testing.assert_frame_equal(result, df1)

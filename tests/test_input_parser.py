import os
import pathlib

import pandas as pd
import pytest

from flashcards_in_a_flash.input_parser import parse_csv


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


def test_parse_csv_valid(valid_csv_path):
    """Test parsing a valid CSV file."""
    result = parse_csv(valid_csv_path)

    # Check if result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check the content of the DataFrame
    assert len(result) == 2
    assert list(result.columns) == ["question", "answer"]
    assert result.iloc[0]["question"] == "food"
    assert result.iloc[0]["answer"] == "cibus"
    assert result.iloc[1]["question"] == "apple"
    assert result.iloc[1]["answer"] == "malum"


def test_parse_csv_empty(empty_csv_path):
    """Test parsing an empty CSV file raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        parse_csv(empty_csv_path)

    assert "CSV file is empty or contains no valid flashcard data" in str(excinfo.value)

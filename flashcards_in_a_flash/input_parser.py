import pathlib

import pandas as pd

EMPTY_CSV_ERROR = "CSV file is empty or contains no valid flashcard data"


def parse_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Parse a CSV file containing flashcard data.

    Reads CSV file with question-answer pairs and converts them to flashcard format.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        list: A list of tuples, each containing 'front' and 'back' keys.

    Raises:
        ValueError: If the CSV file is empty or contains no valid flashcard data.

    Example format of CSV:
        question;answer
        food;cibus
        apple;malum
    """
    try:
        df = pd.read_csv(csv_path, sep=";", header=0, names=["question", "answer"])
        if df.empty:
            raise ValueError(EMPTY_CSV_ERROR)
        return df.drop_duplicates()
    except pd.errors.EmptyDataError as e:
        raise ValueError(EMPTY_CSV_ERROR) from e

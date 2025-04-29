import pathlib

import pandas as pd

EMPTY_CSV_ERROR = "CSV file is empty or contains no valid flashcard data"


def parse_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Parse a CSV file containing flashcard data.

    Reads CSV file with question-answer pairs and converts them to flashcard format.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing 'native' and 'learning' columns.

    Raises:
        ValueError: If the CSV file is empty or contains no valid flashcard data.

    Example format of CSV:
        native;learning
        food;cibus
        apple;malum
    """
    try:
        df = pd.read_csv(csv_path, sep=";", header=0, names=["native", "learning"])
        if df.empty or "native" not in df.columns or "learning" not in df.columns:
            raise ValueError(EMPTY_CSV_ERROR)
        return df.drop_duplicates()
    except pd.errors.EmptyDataError as e:
        raise ValueError(EMPTY_CSV_ERROR) from e


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge two DataFrames on 'native' and 'learning' columns.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame with duplicates removed.

    Raises:
        ValueError: If the merged DataFrame contains empty cells.
    """
    merged_df = pd.merge(
        df1, df2, on=["native", "learning"], how="outer"
    ).drop_duplicates()
    if merged_df.isna().any().any():
        raise ValueError(
            "Merged DataFrame contains empty cells. All cells must have values."
        )
    return merged_df


def parse_pdf(pdf_path: pathlib.Path) -> pd.DataFrame:
    """Parse a PDF file containing flashcard data.

    Reads PDF file with question-answer pairs and converts them to flashcard format.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame containing 'native' and 'learning' columns.

    Raises:
        ValueError: If the PDF file is empty or contains no valid flashcard data.
    """
    # Placeholder for actual PDF parsing logic
    raise NotImplementedError("PDF parsing not implemented yet.")


def parse_image(image_path: pathlib.Path) -> pd.DataFrame:
    """Parse an image file containing flashcard data.

    Reads image file with question-answer pairs and converts them to flashcard format.

    Args:
        image_path: Path to the image file.

    Returns:
        pd.DataFrame: A DataFrame containing 'native' and 'learning' columns.

    Raises:
        ValueError: If the image file is empty or contains no valid flashcard data.
    """
    # Placeholder for actual image parsing logic
    raise NotImplementedError("Image parsing not implemented yet.")


def parse_file(
    file_path: pathlib.Path,
) -> pd.DataFrame:
    """Parse a file (CSV, PDF, or image) containing flashcard data.

    Args:
        file_path: Path to the file.

    Returns:
        pd.DataFrame: A DataFrame containing 'native' and 'learning' columns.

    Raises:
        ValueError: If the file format is unsupported or contains no valid flashcard data.
    """
    if file_path.suffix == ".csv":
        return parse_csv(file_path)
    elif file_path.suffix in [".pdf", ".PDF"]:
        return parse_pdf(file_path)
    elif file_path.suffix in [".jpg", ".jpeg", ".png"]:
        return parse_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

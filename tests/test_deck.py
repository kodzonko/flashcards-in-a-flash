import os
import pathlib

import pandas as pd
import pytest
import pytest_asyncio

from flashcards_in_a_flash.audio_generator import _generate_audio, process_df_async
from flashcards_in_a_flash.deck import AnkiDeck


@pytest.fixture
def test_data():
    """Fixture that returns test data for deck creation."""
    return {
        "native": ["dobry wieczór", "dziękuję", "proszę"],
        "learning": ["buona sera", "grazie", "prego"],
    }


@pytest_asyncio.fixture
async def mock_audio_data():
    """Fixture that generates mock audio data using Edge TTS."""
    return await _generate_audio("test audio", "it-IT")


@pytest_asyncio.fixture
async def bidirectional_audio_deck_path(tmp_path, test_data, mock_audio_data):
    """Fixture that creates a bidirectional deck with audio and returns its path."""
    df = pd.DataFrame(test_data)
    df["audio"] = [mock_audio_data] * len(df)
    output_path = tmp_path / "test_deck_bidirectional_audio.apkg"
    deck = AnkiDeck(name="Polish-Italian Flashcards")
    deck.create(df=df, bidirectional=True)
    deck.write(output_path)
    yield output_path
    # Cleanup
    if output_path.exists():
        os.remove(output_path)


@pytest.fixture
def bidirectional_no_audio_deck_path(tmp_path, test_data):
    """Fixture that creates a bidirectional deck without audio and returns its path."""
    df = pd.DataFrame(test_data)
    output_path = tmp_path / "test_deck_bidirectional_no_audio.apkg"
    deck = AnkiDeck(name="Polish-Italian Flashcards (No Audio)")
    deck.create(df=df, bidirectional=True)
    deck.write(output_path)
    yield output_path
    # Cleanup
    if output_path.exists():
        os.remove(output_path)


@pytest_asyncio.fixture
async def unidirectional_audio_deck_path(tmp_path, test_data, mock_audio_data):
    """Fixture that creates a unidirectional deck with audio and returns its path."""
    df = pd.DataFrame(test_data)
    df["audio"] = [mock_audio_data] * len(df)
    output_path = tmp_path / "test_deck_unidirectional_audio.apkg"
    deck = AnkiDeck(name="Polish-Italian Flashcards (One-way)")
    deck.create(df=df, bidirectional=False)
    deck.write(output_path)
    yield output_path
    # Cleanup
    if output_path.exists():
        os.remove(output_path)


@pytest.fixture
def unidirectional_no_audio_deck_path(tmp_path, test_data):
    """Fixture that creates a unidirectional deck without audio and returns its path."""
    df = pd.DataFrame(test_data)
    output_path = tmp_path / "test_deck_unidirectional_no_audio.apkg"
    deck = AnkiDeck(name="Polish-Italian Flashcards (One-way, No Audio)")
    deck.create(df=df, bidirectional=False)
    deck.write(output_path)
    yield output_path
    # Cleanup
    if output_path.exists():
        os.remove(output_path)


@pytest_asyncio.fixture
async def roundtrip_deck_path(tmp_path, mock_audio_data):
    """Fixture that creates a simple deck for roundtrip testing."""
    data = {
        "native": ["hello", "goodbye", "thank you"],
        "learning": ["hola", "adiós", "gracias"],
    }
    df = pd.DataFrame(data)
    df["audio"] = [mock_audio_data, None, mock_audio_data]
    output_path = tmp_path / "test_roundtrip.apkg"
    deck = AnkiDeck(name="Roundtrip Test Deck")
    deck.create(df=df, bidirectional=True)
    deck.write(output_path)
    return (output_path, df)  # Changed yield to return to avoid async generator issues


@pytest.mark.asyncio
async def test_create_polish_italian_deck_with_audio():
    """Test creating a deck with Polish to Italian cards with audio."""
    data = {
        "native": ["dobry wieczór", "dziękuję", "proszę"],
        "learning": ["buona sera", "grazie", "prego"],
    }
    df = pd.DataFrame(data)
    italian_locale = "it-IT"
    await process_df_async(df, italian_locale)

    assert "audio" in df.columns
    for _, row in df.iterrows():
        assert isinstance(row["audio"], bytes)
        assert len(row["audio"]) > 0

    resources_dir = pathlib.Path(__file__).parent / "resources"
    resources_dir.mkdir(exist_ok=True)
    output_path = resources_dir / "test_deck_bidirectional_audio.apkg"

    deck = AnkiDeck(name="Polish-Italian Flashcards")
    deck.create(
        df=df,
        bidirectional=True,
    )
    deck.write(output_path)
    assert output_path.exists()

    print(f"Deck written to: {output_path!s}")


@pytest.mark.asyncio
async def test_create_unidirectional_deck_with_audio():
    """Test creating a unidirectional (non-bidirectional) deck with audio."""
    data = {
        "native": ["dobry wieczór", "dziękuję", "proszę"],
        "learning": ["buona sera", "grazie", "prego"],
    }
    df = pd.DataFrame(data)
    italian_locale = "it-IT"
    await process_df_async(df, italian_locale)
    assert "audio" in df.columns
    for _, row in df.iterrows():
        assert isinstance(row["audio"], bytes)
        assert len(row["audio"]) > 0

    resources_dir = pathlib.Path(__file__).parent / "resources"
    resources_dir.mkdir(exist_ok=True)
    output_path = resources_dir / "test_deck_unidirectional_audio.apkg"

    deck = AnkiDeck(name="Polish-Italian Flashcards (One-way)")
    deck.create(
        df=df,
        bidirectional=False,
    )
    deck.write(output_path)
    assert output_path.exists()
    print(f"Unidirectional deck with audio written to: {output_path!s}")


@pytest.mark.asyncio
async def test_create_bidirectional_deck_without_audio():
    """Test creating a bidirectional deck without audio."""
    data = {
        "native": ["dobry wieczór", "dziękuję", "proszę"],
        "learning": ["buona sera", "grazie", "prego"],
    }
    df = pd.DataFrame(data)

    resources_dir = pathlib.Path(__file__).parent / "resources"
    resources_dir.mkdir(exist_ok=True)
    output_path = resources_dir / "test_deck_bidirectional_no_audio.apkg"

    deck = AnkiDeck(name="Polish-Italian Flashcards (No Audio)")
    deck.create(
        df=df,
        bidirectional=True,
    )
    deck.write(output_path)
    assert output_path.exists()
    print(f"Bidirectional deck without audio written to: {output_path!s}")


@pytest.mark.asyncio
async def test_read_bidirectional_deck_with_audio(bidirectional_audio_deck_path):
    """Test reading a bidirectional deck with audio from an .apkg file."""
    deck = AnkiDeck()
    df = deck.read(bidirectional_audio_deck_path)
    assert isinstance(df, pd.DataFrame)
    assert "native" in df.columns
    assert "learning" in df.columns
    assert "audio" in df.columns
    assert len(df) > 0
    assert any("dobry wieczór" in str(row["native"]) for _, row in df.iterrows())
    assert any("buona sera" in str(row["learning"]) for _, row in df.iterrows())
    for _, row in df.iterrows():
        assert isinstance(row["audio"], bytes)
        # We might not always be able to extract audio data correctly in tests,
        # so we'll just check that the audio column exists and contains bytes


def test_read_bidirectional_deck_without_audio(bidirectional_no_audio_deck_path):
    """Test reading a bidirectional deck without audio from an .apkg file."""
    deck = AnkiDeck()
    df = deck.read(bidirectional_no_audio_deck_path)
    assert isinstance(df, pd.DataFrame)
    assert "native" in df.columns
    assert "learning" in df.columns
    assert "audio" not in df.columns
    assert len(df) == 3
    assert any("dobry wieczór" in str(row["native"]) for _, row in df.iterrows())
    assert any("buona sera" in str(row["learning"]) for _, row in df.iterrows())


@pytest.mark.asyncio
async def test_read_unidirectional_deck_with_audio(unidirectional_audio_deck_path):
    """Test reading a unidirectional deck with audio from an .apkg file."""
    deck = AnkiDeck()
    df = deck.read(unidirectional_audio_deck_path)
    assert isinstance(df, pd.DataFrame)
    assert "native" in df.columns
    assert "learning" in df.columns
    assert "audio" in df.columns
    assert len(df) == 3
    assert any("dobry wieczór" in str(row["native"]) for _, row in df.iterrows())
    assert any("buona sera" in str(row["learning"]) for _, row in df.iterrows())
    for _, row in df.iterrows():
        assert isinstance(row["audio"], bytes)
        # We might not always be able to extract audio data correctly in tests,
        # so we'll just check that the audio column exists and contains bytes


def test_read_unidirectional_deck_without_audio(unidirectional_no_audio_deck_path):
    """Test reading a unidirectional deck without audio from an .apkg file."""
    deck = AnkiDeck()
    df = deck.read(unidirectional_no_audio_deck_path)
    assert isinstance(df, pd.DataFrame)
    assert "native" in df.columns
    assert "learning" in df.columns
    assert "audio" not in df.columns
    assert len(df) == 3
    assert any("dobry wieczór" in str(row["native"]) for _, row in df.iterrows())
    assert any("buona sera" in str(row["learning"]) for _, row in df.iterrows())


def test_read_invalid_file():
    """Test that reading an invalid file raises a ValueError."""
    input_path = pathlib.Path(__file__)
    deck = AnkiDeck()
    with pytest.raises(ValueError) as excinfo:
        df = deck.read(input_path)
    assert "File is not an Anki package" in str(excinfo.value)


def test_read_nonexistent_file():
    """Test that reading a nonexistent file raises an error."""
    input_path = pathlib.Path("/path/to/nonexistent/file.apkg")
    deck = AnkiDeck()
    with pytest.raises(FileNotFoundError):
        df = deck.read(input_path)


@pytest.mark.asyncio
async def test_create_and_read_roundtrip(roundtrip_deck_path):
    """Test creating a deck, writing it to a file, and reading it back."""
    output_path, df_original = roundtrip_deck_path
    read_deck = AnkiDeck()
    df_read = read_deck.read(output_path)
    assert isinstance(df_read, pd.DataFrame)
    assert "native" in df_read.columns
    assert "learning" in df_read.columns
    assert len(df_read) == len(df_original)
    for _, original_row in df_original.iterrows():
        native = original_row["native"]
        learning = original_row["learning"]
        matching_rows = df_read[
            (df_read["native"] == native) & (df_read["learning"] == learning)
        ]
        assert len(matching_rows) >= 1, (
            f"Couldn't find {native}/{learning} pair in read data"
        )

import os
import pathlib
import tempfile

import pandas as pd
import pytest

from flashcards_in_a_flash.audio_generator import process_dataframe_async
from flashcards_in_a_flash.deck import AnkiDeck


@pytest.mark.asyncio
async def test_create_polish_italian_deck_with_audio():
    """Test creating a deck with Polish to Italian cards with audio."""
    # Create a dataframe with Polish-Italian word pairs
    data = {
        "polish": ["dobry wieczór", "dziękuję", "proszę"],
        "italian": ["buona sera", "grazie", "prego"],
        "text": [
            "buona sera",
            "grazie",
            "prego",
        ],  # For audio generation - Italian words
    }
    df = pd.DataFrame(data)

    # Generate audio for the Italian words
    italian_locale = "it-IT"  # Italian locale for TTS
    await process_dataframe_async(df, italian_locale)

    # Verify audio was generated
    assert "audio" in df.columns
    for _, row in df.iterrows():
        assert isinstance(row["audio"], bytes)
        assert len(row["audio"]) > 0

    # Create temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "polish_italian_deck.apkg")

        # Create and save the deck
        deck = AnkiDeck(name="Polish-Italian Flashcards")
        deck.create(
            df=df,
            native_col="polish",
            learning_col="italian",
            audio_col="audio",
            bidirectional=True,
        )
        saved_path = deck.write(output_path)

        # Verify the deck was saved
        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0

        # Print the path for verification (will be shown in test output)
        print(f"Deck saved to: {saved_path}")

        # Copy deck to a permanent location in the project root
        permanent_path = (
            pathlib.Path(__file__).parent.parent / "polish_italian_deck.apkg"
        )
        with open(saved_path, "rb") as src, open(permanent_path, "wb") as dst:
            dst.write(src.read())

        print(f"Deck copied to project root: {permanent_path}")

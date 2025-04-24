import pathlib

import pandas as pd
import pytest

from flashcards_in_a_flash.audio_generator import process_df_async
from flashcards_in_a_flash.deck import AnkiDeck


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

    output_path = (
        pathlib.Path(__file__).parent
        / "resources"
        / "test_deck_bidirectional_audio.apkg"
    )

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

    output_path = (
        pathlib.Path(__file__).parent
        / "resources"
        / "test_deck_unidirectional_audio.apkg"
    )

    deck = AnkiDeck(name="Polish-Italian Flashcards (One-way)")
    deck.create(
        df=df,
        bidirectional=False,  # Create only one-way cards
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

    # No audio generation here - we're testing deck without audio

    output_path = (
        pathlib.Path(__file__).parent
        / "resources"
        / "test_deck_bidirectional_no_audio.apkg"
    )

    deck = AnkiDeck(name="Polish-Italian Flashcards (No Audio)")
    deck.create(
        df=df,
        bidirectional=True,
    )
    deck.write(output_path)
    assert output_path.exists()

    print(f"Bidirectional deck without audio written to: {output_path!s}")

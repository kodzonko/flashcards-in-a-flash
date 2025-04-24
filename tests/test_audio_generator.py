from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from flashcards_in_a_flash.audio_generator import (
    _generate_audio,
    _process_row,
    process_df_async,
)


@pytest.mark.asyncio
async def test_process_dataframe_async():
    test_df = pd.DataFrame({"learning": ["Hello world", "This is a test"]})
    original_columns = list(test_df.columns)
    locale = "en-US"

    await process_df_async(test_df, locale)

    assert set(test_df.columns) == {*original_columns, "audio"}

    for _, row in test_df.iterrows():
        assert "audio" in row
        assert row["audio"] is not None
        assert isinstance(row["audio"], bytes)
        assert len(row["audio"]) > 0


@pytest.mark.asyncio
async def test_generate_audio():
    """Test that generate_audio returns valid audio bytes."""
    text = "Hello, this is a test."
    locale = "en-US"

    audio_bytes = await _generate_audio(text, locale)

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


@pytest.mark.asyncio
async def test_process_row():
    row = pd.Series({"learning": "Hello world"})

    with patch(
        "flashcards_in_a_flash.audio_generator._generate_audio", new_callable=AsyncMock
    ) as mock_generate_audio:
        mock_audio = b"dummy audio data"
        mock_generate_audio.return_value = mock_audio

        audio_result = await _process_row(row, "en-US")

        mock_generate_audio.assert_called_once_with("Hello world", "en-US")

        assert audio_result == mock_audio

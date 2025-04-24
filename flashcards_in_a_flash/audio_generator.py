import asyncio
import io
import random
from functools import cache

import edge_tts
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm.asyncio import tqdm


@cache
def list_supported_languages() -> None:
    """List all supported languages for Edge TTS."""

    async def inner():
        return await edge_tts.list_voices()

    languages = asyncio.run(inner())

    console = Console()
    table = Table(title="Supported Edge TTS Languages")
    table.add_column("Locale", style="cyan")
    table.add_column("Voice Name", style="green")

    prev_locale = languages[0]["Locale"]

    for lang in languages:
        locale = lang["Locale"]
        voice_name = lang["FriendlyName"].replace("Microsoft ", "")
        if locale != prev_locale:
            table.add_section()
            prev_locale = locale
        table.add_row(locale, voice_name)

    console.print(table)


async def _generate_audio(text: str, locale: str) -> bytes:
    """Generate audio using Edge TTS."""
    voices = await edge_tts.VoicesManager.create()
    voice = voices.find(Locale=locale)
    try:
        communicate = edge_tts.Communicate(text, random.choice(voice)["Name"])
    except IndexError as e:
        raise ValueError(f"No voice found for locale: {locale}") from e
    audio_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk_data := chunk.get("data"):
            audio_data.write(chunk_data)
    return audio_data.getvalue()


async def _process_row(row, locale: str) -> bytes:
    """Process a single row of the DataFrame, generating audio."""
    return await _generate_audio(row["learning"], locale)


def process_df(df: pd.DataFrame, locale: str) -> pd.DataFrame:
    """Process a DataFrame, generating audio for each row.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'audio_data' column
            containing the generated audio bytes.
    """
    results = []
    indices = []

    for i, row in tqdm(df.iterrows(), desc="Generating TTS", total=len(df)):
        audio = asyncio.run(_process_row(row, locale))
        results.append(audio)
        indices.append(i)

    df.loc[:, "audio"] = pd.Series(results, index=indices)
    return df


async def process_df_async(df: pd.DataFrame, locale: str) -> pd.DataFrame:
    """Process a DataFrame asynchronously, generating audio for each row.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'audio_data' column
            containing the generated audio bytes.
    """
    tasks = []
    indices = []
    for i, row in df.iterrows():
        tasks.append(_process_row(row, locale))
        indices.append(i)
    results = await tqdm.gather(*tasks, desc="Generating TTS")
    df.loc[:, "audio"] = pd.Series(results, index=indices)
    return df

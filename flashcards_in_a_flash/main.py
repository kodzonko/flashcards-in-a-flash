import pathlib

import typer

from flashcards_in_a_flash.audio_generator import (
    list_supported_languages,
)
from flashcards_in_a_flash.input_parser import parse_csv

app = typer.Typer()


@app.command()
def main(
    deck: pathlib.Path | None = None,
    csv: pathlib.Path | None = None,
    audio: bool = False,
    list_languages: bool = False,
):
    """Create Anki flashcards from a source file."""

    # If list_languages flag is set, show available TTS languages and exit
    if list_languages:
        list_supported_languages()
        return 0

    deck = typer.Option(
        "anki_deck.apkg",
        "--deck",
        help="Path to the Anki deck (existing or new)",
        writable=True,
        resolve_path=True,
        prompt="Path to the Anki deck (existing or new)",
    )(deck)

    csv = typer.Option(None, "--csv", help="Path to the CSV input file", exists=True)(
        csv
    )
    audio = typer.Option(
        False, "--audio", help="Generate audio for flashcards", is_flag=True
    )(audio)
    list_languages = typer.Option(
        False, "--list-languages", help="List available TTS languages", is_flag=True
    )(list_languages)

    flashcards = None
    if csv is not None:
        flashcards = parse_csv(csv)
        print(f"Found {len(flashcards)} flashcards in the CSV file")
    if audio and flashcards is not None:
        print("Generating audio for flashcards...")
        ...


if __name__ == "__main__":
    app()

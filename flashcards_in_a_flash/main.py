import os
import pathlib
import random
from typing import Optional

import genanki
import typer

from flashcards_in_a_flash.audio_generator import (
    generate_audio,
    list_supported_languages,
)
from flashcards_in_a_flash.input_parser import parse_csv

app = typer.Typer()

@app.command()
def main(
    deck: Optional[pathlib.Path] = None,
    csv: Optional[pathlib.Path] = None,
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

    csv = typer.Option(
        None,
        "--csv",
        help="Path to the CSV input file",
        exists=True)(csv)
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
        generate_audio()

        # Create a deck
        deck_id = random.randrange(1 << 30, 1 << 31)
        name = "Flashcards"  # Define a name for the deck
        anki_deck = genanki.Deck(deck_id, name)
        if audio:
            print("Generating audio for flashcards...")
            generate_audio()

        # Create a deck
        deck_id = random.randrange(1 << 30, 1 << 31)
        anki_deck = genanki.Deck(deck_id, name)

        # Add cards to the deck
        for card_data in flashcards:
            note = genanki.Note(
                model=BASIC_MODEL, fields=[card_data["front"], card_data["back"]]
            )
            anki_deck.add_note(note)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(deck)) or ".", exist_ok=True)

        # Make sure the deck has a .apkg extension
        if not deck.lower().endswith(".apkg"):
            deck = f"{deck}.apkg"

        # Save the deck
        genanki.Package(anki_deck).write_to_file(deck)
        print(f"Deck successfully saved to: {deck}")

    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)

    return 0


if __name__ == "__main__":
    app()

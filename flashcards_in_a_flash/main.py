import click
import os
import genanki
import random
from input_parser import parse_csv_file

# Define a model for our flashcards
BASIC_MODEL = genanki.Model(
    random.randrange(1 << 30, 1 << 31),
    "Basic Model",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "{{Front}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
        },
    ],
)


@click.command()
@click.option(
    "--csv",
    required=True,
    type=click.Path(exists=True),
    help="Path to the CSV file containing flashcard data",
)
@click.option(
    "--deck",
    required=True,
    type=click.Path(),
    help="Path where the Anki deck will be saved",
)
@click.option("--name", default="Flashcards", help="Name for the Anki deck")
def main(csv, deck, name):
    """Create Anki flashcards from a CSV file."""
    try:
        # Parse the CSV file
        print(f"Processing CSV file: {csv}")
        flashcards = parse_csv_file(csv)
        print(f"Found {len(flashcards)} flashcards in the CSV file")

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
        click.echo(f"Error: {str(e)}", err=True)
        return 1

    return 0


if __name__ == "__main__":
    main()

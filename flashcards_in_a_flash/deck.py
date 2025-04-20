import os
import random
import tempfile
from pathlib import Path

import genanki  # type: ignore
import pandas as pd

CARD_STYLING = """
    .card {
        font-family: Helvetica, sans-serif;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        padding: 20px;
    }
    hr#answer { margin: 20px 0; }
    """

# Define Anki note models with and without audio
BASIC_MODEL = genanki.Model(
    model_id=random.randrange(1 << 30, 1 << 31),
    name="Basic Flashcard Model",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": '<div class="card basic"><div class="question">{{Question}}</div></div>',
            "afmt": '<div class="card basic"><div class="question">{{Question}}</div><hr id="answer"><div class="answer">{{Answer}}</div></div>',
        },
    ],
    css=CARD_STYLING,
)

BASIC_MODEL_WITH_AUDIO = genanki.Model(
    model_id=random.randrange(1 << 30, 1 << 31),
    name="Basic Flashcard Model with Audio",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
        {"name": "Audio"},
        {
            "name": "IsLearnedLanguageOnFront"
        },  # Flag to indicate if front has learned language
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": '<div class="card basic"><div class="question">{{Question}}</div>{{#IsLearnedLanguageOnFront}}<div>{{Audio}}</div>{{/IsLearnedLanguageOnFront}}</div>',
            "afmt": '<div class="card basic"><div class="question">{{Question}}</div><hr id="answer"><div class="answer">{{Answer}}</div>{{^IsLearnedLanguageOnFront}}<div>{{Audio}}</div>{{/IsLearnedLanguageOnFront}}</div>',
        },
    ],
    css=CARD_STYLING,
)

# Bidirectional model with audio on language learning side
BIDIRECTIONAL_MODEL_WITH_AUDIO = genanki.Model(
    model_id=random.randrange(1 << 30, 1 << 31),
    name="Bidirectional Flashcard Model with Audio",
    fields=[
        {"name": "Native"},
        {"name": "Learning"},
        {"name": "Audio"},
    ],
    templates=[
        {
            "name": "Native to Learning",
            "qfmt": '<div class="card bidirectional native-to-learning"><div class="question native">{{Native}}</div></div>',
            "afmt": '<div class="card bidirectional native-to-learning"><div class="question native">{{Native}}</div><hr id="answer"><div class="answer learning">{{Learning}}</div><div>{{Audio}}</div></div>',
        },
        {
            "name": "Learning to Native",
            "qfmt": '<div class="card bidirectional learning-to-native"><div class="question learning">{{Learning}}</div><div>{{Audio}}</div></div>',
            "afmt": '<div class="card bidirectional learning-to-native"><div class="question learning">{{Learning}}</div><hr id="answer"><div class="answer native">{{Native}}</div></div>',
        },
    ],
    css=CARD_STYLING,
)


class AnkiDeck:
    """Class to create and manage Anki decks with flashcards and optional audio."""

    def __init__(self, name: str = "Flashcards", deck_id: int | None = None):
        """Initialize an Anki deck.

        Args:
            name: Name of the deck
            deck_id: Unique ID for the deck (will be randomly generated if not provided)
        """
        if deck_id is None:
            deck_id = random.randrange(1 << 30, 1 << 31)
        self.deck = genanki.Deck(deck_id, name)
        self.media_files = []
        # Create a temporary directory to store audio files
        self.temp_dir = tempfile.TemporaryDirectory()

    def create_from_dataframe(
        self,
        df: pd.DataFrame,
        question_col: str = "question",
        answer_col: str = "answer",
        audio_col: str | None = "audio",
        audio_format: str = "mp3",
        bidirectional: bool = False,
    ) -> "AnkiDeck":
        """Create an Anki deck from a DataFrame with questions, answers, and optional audio.

        Args:
            df: DataFrame containing flashcard data
            question_col: Column name for questions (native language if bidirectional)
            answer_col: Column name for answers (learning language if bidirectional)
            audio_col: Column name for audio data (as bytes), if None, no audio is used
            audio_format: Format of the audio files (mp3, wav, etc.)
            bidirectional: Whether to create cards in both directions

        Returns:
            self: The AnkiDeck instance for chaining

        Raises:
            ValueError: If required columns are not in the DataFrame
        """
        required_cols = [question_col, answer_col]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        has_audio = audio_col is not None and audio_col in df.columns

        for idx, row in df.iterrows():
            native_text = row[question_col]
            learning_text = row[answer_col]

            if has_audio and not pd.isna(row[audio_col]):
                # Create a unique filename for the audio
                audio_filename = f"audio_{idx}.{audio_format}"
                audio_path = Path(self.temp_dir.name) / audio_filename

                # Write audio bytes to a file in the temporary directory
                with open(audio_path, "wb") as f:
                    f.write(row[audio_col])

                # Add the audio file to media files list with the full path
                self.media_files.append(str(audio_path))

                if bidirectional:
                    # Create bidirectional note with audio on learning language side
                    note = genanki.Note(
                        model=BIDIRECTIONAL_MODEL_WITH_AUDIO,
                        fields=[
                            str(native_text),
                            str(learning_text),
                            f"[sound:{audio_filename}]",
                        ],
                    )
                    self.deck.add_note(note)
                else:
                    # Create a one-way note with audio
                    note = genanki.Note(
                        model=BASIC_MODEL_WITH_AUDIO,
                        fields=[
                            str(native_text),
                            str(learning_text),
                            f"[sound:{audio_filename}]",
                        ],
                    )
                    self.deck.add_note(note)
            else:
                if bidirectional:
                    # When no audio is available but still want bidirectional cards
                    # Use our predefined bidirectional model without audio
                    bidirectional_model = genanki.Model(
                        model_id=random.randrange(1 << 30, 1 << 31),
                        name="Bidirectional Flashcard Model",
                        fields=[{"name": "Native"}, {"name": "Learning"}],
                        templates=[
                            {
                                "name": "Native to Learning",
                                "qfmt": '<div class="card bidirectional native-to-learning"><div class="question native">{{Native}}</div></div>',
                                "afmt": '<div class="card bidirectional native-to-learning"><div class="question native">{{Native}}</div><hr id="answer"><div class="answer learning">{{Learning}}</div></div>',
                            },
                            {
                                "name": "Learning to Native",
                                "qfmt": '<div class="card bidirectional learning-to-native"><div class="question learning">{{Learning}}</div></div>',
                                "afmt": '<div class="card bidirectional learning-to-native"><div class="question learning">{{Learning}}</div><hr id="answer"><div class="answer native">{{Native}}</div></div>',
                            },
                        ],
                        css=CARD_STYLING,
                    )
                    note = genanki.Note(
                        model=bidirectional_model,
                        fields=[str(native_text), str(learning_text)],
                    )
                    self.deck.add_note(note)
                else:
                    # Create a basic one-way note without audio
                    note = genanki.Note(
                        model=BASIC_MODEL, fields=[str(native_text), str(learning_text)]
                    )
                    self.deck.add_note(note)

        return self

    def save(self, output_path: str | Path = "flashcards.apkg") -> str:
        """Save the deck to an Anki package file.

        Args:
            output_path: Path where to save the Anki package

        Returns:
            str: The path to the saved file
        """
        try:
            # Ensure the output has the correct extension
            if not str(output_path).lower().endswith(".apkg"):
                output_path = f"{output_path}.apkg"

            # Ensure the directory exists
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Create the package with the deck and media files
            package = genanki.Package(self.deck)

            # Add media files if they exist
            if self.media_files:
                package.media_files = self.media_files

            # Write the package to a file
            package.write_to_file(output_path)

            return str(output_path)
        finally:
            # Clean up the temporary directory and its contents when done
            self.temp_dir.cleanup()

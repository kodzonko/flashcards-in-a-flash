import random
import tempfile
from pathlib import Path
from typing import Self

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
            "qfmt": (
                '<div class="card basic"><div class="question">{{Question}}</div></div>'
            ),
            "afmt": (
                '<div class="card basic">'
                '<div class="question">{{Question}}</div>'
                '<hr id="answer">'
                '<div class="answer">{{Answer}}</div>'
                "</div>"
            ),
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
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": (
                '<div class="card basic">'
                '<div class="question">{{Question}}</div>'
                "<div>{{Audio}}</div>"
                "</div>"
            ),
            "afmt": (
                '<div class="card basic">'
                '<div class="question">{{Question}}</div>'
                '<hr id="answer">'
                '<div class="answer">{{Answer}}</div>'
                "<div>{{Audio}}</div>"
                "</div>"
            ),
        },
    ],
    css=CARD_STYLING,
)

BIDIRECTIONAL_MODEL = genanki.Model(
    model_id=random.randrange(1 << 30, 1 << 31),
    name="Bidirectional Flashcard Model",
    fields=[
        {"name": "Native"},
        {"name": "Learning"},
    ],
    templates=[
        {
            "name": "Native to Learning",
            "qfmt": (
                '<div class="card bidirectional native-to-learning">'
                '<div class="question native">{{Native}}</div>'
                "</div>"
            ),
            "afmt": (
                '<div class="card bidirectional native-to-learning">'
                '<div class="question native">{{Native}}</div>'
                '<hr id="answer">'
                '<div class="answer learning">{{Learning}}</div>'
                "</div>"
            ),
        },
        {
            "name": "Learning to Native",
            "qfmt": (
                '<div class="card bidirectional learning-to-native">'
                '<div class="question learning">{{Learning}}</div>'
                "</div>"
            ),
            "afmt": (
                '<div class="card bidirectional learning-to-native">'
                '<div class="question learning">{{Learning}}</div>'
                '<hr id="answer">'
                '<div class="answer native">{{Native}}</div>'
                "</div>"
            ),
        },
    ],
    css=CARD_STYLING,
)

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
            "qfmt": (
                '<div class="card bidirectional native-to-learning">'
                '<div class="question native">{{Native}}</div>'
                "</div>"
            ),
            "afmt": (
                '<div class="card bidirectional native-to-learning">'
                '<div class="question native">{{Native}}</div>'
                '<hr id="answer">'
                '<div class="answer learning">{{Learning}}</div>'
                "<div>{{Audio}}</div>"
                "</div>"
            ),
        },
        {
            "name": "Learning to Native",
            "qfmt": (
                '<div class="card bidirectional learning-to-native">'
                '<div class="question learning">{{Learning}}</div>'
                "<div>{{Audio}}</div>"
                "</div>"
            ),
            "afmt": (
                '<div class="card bidirectional learning-to-native">'
                '<div class="question learning">{{Learning}}</div>'
                '<hr id="answer">'
                '<div class="answer native">{{Native}}</div>'
                "</div>"
            ),
        },
    ],
    css=CARD_STYLING,
)


class AnkiDeck:
    """Class to create and manage Anki decks with flashcards and optional audio."""

    def __init__(
        self, name: str = "Flashcards-in-a-flash", deck_id: int | None = None
    ) -> None:
        """Initialize an Anki deck.

        Args:
            name: Name of the deck
            deck_id: Unique ID for the deck (will be randomly generated if not provided)
        """
        if deck_id is None:
            deck_id = random.randrange(1 << 30, 1 << 31)
        self.deck = genanki.Deck(deck_id, name)
        self.media_files: list[str] = []
        # Create a temporary directory to store audio files
        self.temp_dir = tempfile.TemporaryDirectory()

    def create_from_dataframe(
        self,
        df: pd.DataFrame,
        native_col: str = "native",
        learning_col: str = "learning",
        audio_col: str | None = "audio",
        audio_format: str = "mp3",
        bidirectional: bool = True,
    ) -> Self:
        """Create an Anki deck from a DataFrame with questions, answers, and optional audio.

        Args:
            df: DataFrame containing flashcard data
            native_col: Column name for questions (native language if bidirectional)
            learning_col: Column name for answers (learning language if bidirectional)
            audio_col: Column name for audio data (as bytes), if None, no audio is used
            audio_format: Format of the audio files (mp3, wav, etc.)
            bidirectional: Whether to create cards in both directions

        Returns:
            self: The AnkiDeck instance for chaining

        Raises:
            ValueError: If required columns are not in the DataFrame
        """
        required_cols = [native_col, learning_col]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        has_audio = audio_col is not None and audio_col in df.columns

        for idx, row in df.iterrows():
            native_text = row[native_col]
            learning_text = row[learning_col]

            if has_audio and not pd.isna(row[audio_col]):
                audio_filename = f"audio_{idx}.{audio_format}"
                audio_path = Path(self.temp_dir.name) / audio_filename
                with open(audio_path, "wb") as f:
                    f.write(row[audio_col])
                self.media_files.append(str(audio_path))

                if bidirectional:
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
                    note = genanki.Note(
                        model=BIDIRECTIONAL_MODEL,
                        fields=[str(native_text), str(learning_text)],
                    )
                    self.deck.add_note(note)
                else:
                    note = genanki.Note(
                        model=BASIC_MODEL, fields=[str(native_text), str(learning_text)]
                    )
                    self.deck.add_note(note)

        return self

    def save(self, output_path: Path) -> Path:
        """Save the deck to an Anki package file.

        Args:
            output_path: Path where to save the Anki package

        Returns:
            Path: The path to the saved file
        """
        try:
            if not output_path.parent.exists():
                raise ValueError(
                    f"Output path '{output_path.parent!s}' "
                    f"does not exist or I don't have permissions."
                )
            if not str(output_path).lower().endswith(".apkg"):
                output_path = output_path.with_suffix(".apkg")
            package = genanki.Package(self.deck)
            if self.media_files:
                package.media_files = self.media_files
            package.write_to_file(output_path)
            return output_path
        finally:
            self.temp_dir.cleanup()

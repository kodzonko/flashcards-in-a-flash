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

    def create(
        self,
        df: pd.DataFrame,
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
        required_cols = ["native", "learning"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        has_audio = "audio" != None and "audio" in df.columns

        for idx, row in df.iterrows():
            native_text = row["native"]
            learning_text = row["learning"]

            if has_audio and not pd.isna(row["audio"]):
                audio_filename = f"audio_{idx}.mp3"
                audio_path = Path(self.temp_dir.name) / audio_filename
                with open(audio_path, "wb") as f:
                    f.write(row["audio"])
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

    def read(self, apkg_path: Path) -> pd.DataFrame:
        """Load an existing Anki package file into a DataFrame.

        This function extracts notes from an Anki package and converts them into
        a DataFrame that matches the schema used by create_from_dataframe.

        Args:
            apkg_path: Path to the Anki package (.apkg) file

        Returns:
            DataFrame: A DataFrame containing the flashcard data in a format
                       compatible with create_from_dataframe

        Raises:
            ValueError: If the file doesn't exist or isn't a valid .apkg file
            ImportError: If the required libraries aren't available
        """
        import os
        import sqlite3
        import tempfile
        import zipfile

        if not str(apkg_path).lower().endswith(".apkg"):
            raise ValueError(f"File is not an Anki package: {apkg_path}")

        # Create a temporary directory to extract the apkg
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the apkg
            with zipfile.ZipFile(apkg_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Connect to the SQLite database
            db_path = os.path.join(temp_dir, "collection.anki2")
            conn = None
            cursor = None
            data = []

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check which schema version we're working with
                # Try the newer schema first (notetypes table)
                has_notetypes_table = False
                try:
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='notetypes'"
                    )
                    has_notetypes_table = len(cursor.fetchall()) > 0
                except sqlite3.Error:
                    pass

                # Handle different database schemas
                if has_notetypes_table:
                    # Newer Anki schema
                    cursor.execute("SELECT id, name FROM notetypes")
                    models = {model_id: name for model_id, name in cursor.fetchall()}

                    # Query to extract the notes
                    query = """
                    SELECT n.id, n.mid, n.flds, m.name as model_name
                    FROM notes n
                    JOIN notetypes m ON n.mid = m.id
                    ORDER BY n.id
                    """
                else:
                    # Older Anki schema or genanki-generated schema
                    # In this case, we need to get models from the col table JSON
                    import json

                    cursor.execute("SELECT models FROM col")
                    models_json = cursor.fetchone()[0]

                    # Handle potential string or bytes
                    if isinstance(models_json, bytes):
                        models_json = models_json.decode("utf-8")

                    models_dict = json.loads(models_json)
                    models = {
                        int(model_id): model_data.get("name", "")
                        for model_id, model_data in models_dict.items()
                    }

                    # Query to extract the notes with models from the col table
                    query = """
                    SELECT n.id, n.mid, n.flds
                    FROM notes n
                    ORDER BY n.id
                    """

                # Execute the query
                cursor.execute(query)
                notes = cursor.fetchall()

                # Create a DataFrame to hold the flashcard data
                media_dir = os.path.join(temp_dir, "media")

                # Check if media directory exists and is actually a directory
                media_dir_exists = os.path.exists(media_dir) and os.path.isdir(
                    media_dir
                )

                # If the media file exists but isn't a directory, it might be the media file itself
                if os.path.exists(media_dir) and not os.path.isdir(media_dir):
                    # In this case, we need to handle the media mapping differently
                    try:
                        # Try to read the media file which might contain the mapping
                        with open(media_dir) as f:
                            media_content = f.read()
                    except:
                        pass

                for note_data in notes:
                    if has_notetypes_table:
                        note_id, model_id, flds, model_name = note_data
                    else:
                        note_id, model_id, flds = note_data
                        model_name = models.get(model_id, "")

                    fields = flds.split("\x1f")  # Anki separator for fields
                    note_dict = {}

                    # Process based on model type
                    if len(fields) >= 2:
                        note_dict["native"] = fields[0]
                        note_dict["learning"] = fields[1]

                        # Look for audio field
                        if len(fields) >= 3:
                            audio_field = fields[2]
                            import re

                            audio_match = re.search(r"\[sound:(.*?)\]", audio_field)
                            if audio_match:
                                audio_file = audio_match.group(1)
                                if media_dir_exists:  # Only try to access media_dir if it exists and is a directory
                                    audio_path = os.path.join(media_dir, audio_file)
                                    if os.path.exists(audio_path):
                                        with open(audio_path, "rb") as f:
                                            note_dict["audio"] = f.read()
                                    else:
                                        # Try finding any audio file in the media directory
                                        for file in os.listdir(media_dir):
                                            if os.path.isfile(
                                                os.path.join(media_dir, file)
                                            ):
                                                with open(
                                                    os.path.join(media_dir, file), "rb"
                                                ) as f:
                                                    note_dict["audio"] = f.read()
                                                break

                            # If we still don't have audio but it's expected,
                            # add an empty bytes object so the column exists
                            if "audio" not in note_dict and "with Audio" in model_name:
                                note_dict["audio"] = b""

                    # Add the note data to our collection if we have the required fields
                    if "native" in note_dict and "learning" in note_dict:
                        data.append(note_dict)

            finally:
                # Ensure database connection is properly closed
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            # Return the DataFrame after the database connection is closed
            return pd.DataFrame(data)

    def write(self, output_path: Path) -> None:
        """Save the deck to an Anki package file.

        Args:
            output_path: Path where to save the Anki package
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
        finally:
            self.temp_dir.cleanup()

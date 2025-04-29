# Flashcards in a Flash

A Python tool that generates Anki flashcards for language learning with audio pronunciation support. Quickly create flashcards from CSV files with text-to-speech audio using Edge TTS.

## Features

✅ Generate flashcards from CSV files

✅ Generate audio with correct pronunciation using Edge TTS

✅ Create bidirectional cards (native-to-learning and learning-to-native)

✅ Custom styling for flashcards with responsive design

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flashcards-in-a-flash.git
cd flashcards-in-a-flash

# Install dependencies with UV (recommended)
uv pip install -e .

# Alternative: Install with pip
# pip install -e .
```

## Usage

### Basic Usage

```bash
python -m flashcards_in_a_flash.main --csv your_vocabulary.csv --audio
```

### List Available TTS Languages

```bash
python -m flashcards_in_a_flash.main --list-languages
```

### CSV Format

Your CSV file should use semicolons (;) as separators and have a header row:

```
question;answer
food;cibus
apple;malum
```

## Roadmap

- [x] Generate flashcards from CSV files
- [x] Generate audio with correct pronunciation for each flashcard
- [ ] Generate flashcards from PDF files (with OCR) - for underlined text and handwritten notes
- [ ] Generate flashcards from images (with OCR) - against a list of top words (by frequency)
- [ ] Generate mnemonics for each flashcard
- [ ] Generate example sentences for each flashcard
- [ ] Generate flashcards with common words & phrases for a specific topic or situation
- [ ] Generate images to illustrate the meaning of each flashcard, preferably to serve as a mnemonic

## Tech Stack

- [Python](https://www.python.org/) - Core programming language
- [Genanki](https://github.com/kerrickstaley/genanki) - Library for creating Anki decks
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [Edge TTS](https://github.com/rany2/edge-tts) - Text-to-speech for pronunciation audio
- [Typer](https://typer.tiangolo.com/) - Command-line interface

## License

[MIT License](LICENSE)

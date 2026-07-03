"""Enable `python -m ghana_speech_datagen ...` as an alias for the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

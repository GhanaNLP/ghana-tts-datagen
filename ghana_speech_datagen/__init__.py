"""VoxCPM synthetic-speech data generator — turn text datasets into TTS/ASR training data.

Voice-clones built-in male/female reference speakers with the Ghana NLP Community
VoxCPM model (ghana-tts-36k), writing WAVs + a manifest, with parallel instances
and resume.
"""

from .generator import (
    DEFAULT_SR,
    EXPORT_FORMATS,
    MODEL_ID,
    SAMPLE_RATE,
    SPEAKERS,
    auto_instances,
    clean_text,
    export_formats,
    generate,
    pick_gender,
    preview,
    resolve_speakers,
    sanitize_name,
    trim_silences,
)

__all__ = [
    "DEFAULT_SR",
    "EXPORT_FORMATS",
    "MODEL_ID",
    "SAMPLE_RATE",
    "SPEAKERS",
    "auto_instances",
    "clean_text",
    "export_formats",
    "generate",
    "pick_gender",
    "preview",
    "resolve_speakers",
    "sanitize_name",
    "trim_silences",
]

__version__ = "0.1.0"

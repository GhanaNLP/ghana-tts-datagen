"""Core synthetic-speech generation against a vLLM-Omni VoxCPM2 TTS server.

Streams text-reference pairs, synthesises each row with the Ghana NLP Community
VoxCPM2 model served by vLLM-Omni (OpenAI-compatible speech API), and writes
WAVs at the chosen sample rate + manifests. The model runs as a standalone GPU
server (see ``deploy/``); this module is a pure HTTP client.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000       # rate the packaged reference voices are normalised to
DEFAULT_SR = 24000        # default OUTPUT rate (TTS-friendly); override with --sample-rate

_SPEAKER_DIR = Path(__file__).resolve().parent / "speakers"
SPEAKERS: dict[str, dict] = {}
for _g in ("male", "female"):
    _wav = _SPEAKER_DIR / f"{_g}.wav"
    _txt = _SPEAKER_DIR / f"{_g}.txt"
    if _wav.is_file() and _txt.is_file():
        SPEAKERS[_g] = {"wav": str(_wav), "text": _txt.read_text(encoding="utf-8").strip()}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def pick_gender(idx: int, mode: str, male_pct: int) -> str:
    if mode in ("male", "all male"):
        return "male"
    if mode in ("female", "all female"):
        return "female"
    return "male" if (idx * 2654435761) % 100 < male_pct else "female"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-") or "run"


def builtin_speaker_refs() -> list[tuple[str, str]]:
    """Return ``(wav_path, transcript)`` pairs for the packaged reference voices.

    Used as a fallback reference pool when the caller supplies no reference
    audio (e.g. for a language with no default in-language audio).
    """
    return [(s["wav"], s["text"]) for s in SPEAKERS.values()]


def normalize_audio(audio_input, out_dir: str) -> str:
    """Read audio (HF dict or path), convert to 16 kHz mono 16-bit WAV, return path."""
    os.makedirs(os.path.join(out_dir, "_normalized"), exist_ok=True)

    if isinstance(audio_input, dict):
        arr = audio_input.get("array")
        sr = audio_input.get("sampling_rate", SAMPLE_RATE)
        if arr is not None:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != SAMPLE_RATE:
                arr = librosa.resample(arr, orig_sr=int(sr), target_sr=SAMPLE_RATE)
        else:
            arr, _ = librosa.load(audio_input["path"], sr=SAMPLE_RATE, mono=True)
        h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        src = ("array", arr)
    else:
        p = str(audio_input)
        h = hashlib.sha256(p.encode()).hexdigest()[:16]
        src = ("path", p)

    out_path = os.path.join(out_dir, "_normalized", f"{h}.wav")
    if not os.path.isfile(out_path):
        wav = src[1] if src[0] == "array" else librosa.load(src[1], sr=SAMPLE_RATE, mono=True)[0]
        tmp = out_path + ".tmp.wav"
        sf.write(tmp, wav, SAMPLE_RATE, subtype="PCM_16")
        os.replace(tmp, out_path)
    return out_path


def resample(wav, src_sr: int, dst_sr: int):
    wav = np.asarray(wav, dtype=np.float32)
    if src_sr == dst_sr or wav.size == 0:
        return wav
    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr)


# --------------------------------------------------------------------------- #
# Core generation (vLLM-Omni VoxCPM2 server)
# --------------------------------------------------------------------------- #
def generate(
    *,
    out_dir: str,
    pairs: list,
    output_formats=("asr",),
    speaker_labels: list | None = None,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    min_samples: int = 50,
    target_seconds: float = 3600,
    sample_rate: int = DEFAULT_SR,
    cfg_value: float = 2.0,
    on_clip=None,
    on_save=None,
    save_every: int = 0,
    progress=None,
    server_url: str | None = None,
    api_key: str | None = None,
    model: str = "voxcpm2",
    lang: str | None = None,
) -> dict:
    """Synthesise speech from texts against a vLLM-Omni VoxCPM2 server.

    ``pairs`` is a list of ``(text_to_synthesise, ref_audio, ref_text)`` tuples.
    Each text is voiced by its paired reference audio (the voice prompt).

    ``on_save`` / ``save_every`` -- when ``save_every > 0``, the manifests are
    flushed to disk and ``on_save(out_dir)`` is called every ``save_every`` new
    clips (and once at the end). Used to push data incrementally as it is made.

    ``output_formats`` -- which manifest(s) to emit next to ``wavs/`` and the
    always-written ``manifest.jsonl``.  ``"asr"`` writes ``metadata.jsonl``
    (``{"audio","text"}``); ``"ljspeech"`` writes ``metadata.csv``
    (``id|text|text``) -- the standard single-/multi-speaker TTS layout.

    ``speaker_labels`` -- optional list parallel to ``pairs``; when given, each
    manifest row records which speaker voiced it (useful for multi-speaker TTS).

    ``lang`` -- optional language code. When set, the model's language tag
    ``<|lang:CODE|> `` is prepended to each text *for synthesis only* — this is
    exactly how VoxCPM2-Ghana was trained (the tag is learned as plain text, so
    it is required for correct per-language pronunciation, especially across the
    many Latin-script languages). The manifests still store the clean transcript.

    ``server_url`` / ``api_key`` / ``model`` -- how to reach the running
    vLLM-Omni TTS server (defaults come from ``TTS_SERVER_URL`` / ``TTS_API_KEY``
    / ``TTS_MODEL_NAME`` env vars).
    """
    from ghana_speech_datagen.tts_client import VoxCPM2Client

    out_dir = str(out_dir)
    wav_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    tag = f"<|lang:{lang}|> " if lang else ""

    def _flush(rows: list) -> list:
        """Write manifest.jsonl + requested formats; return the format paths."""
        with open(Path(out_dir) / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return export_formats(out_dir, output_formats)

    def _save(rows: list) -> None:
        _flush(rows)
        if on_save is not None:
            try:
                on_save(out_dir)
            except Exception as e:  # a failed upload shouldn't lose the run
                if progress:
                    progress(f"save failed: {e}")

    with VoxCPM2Client(base_url=server_url, api_key=api_key, model=model) as server:
        server.wait_until_ready(timeout=120.0)

        # Normalise + register each unique reference voice once, then map every
        # pair to its voice id.  De-duping by path lets a small reference pool
        # (e.g. the two packaged speakers) drive an arbitrary number of texts.
        path_to_vid: dict[str, str | None] = {}
        pair_vid: list[str | None] = []
        for _, audio_input, ref_text in pairs:
            try:
                path = normalize_audio(audio_input, out_dir)
            except Exception:
                pair_vid.append(None)
                continue
            if path not in path_to_vid:
                vid = f"ref_{len(path_to_vid)}"
                try:
                    server.register_voice(vid, path, ref_text)
                    path_to_vid[path] = vid
                except Exception:
                    path_to_vid[path] = None
            pair_vid.append(path_to_vid[path])

        n_voices = sum(1 for v in set(path_to_vid.values()) if v is not None)
        if n_voices == 0:
            raise RuntimeError("No reference voices could be registered. Aborting.")

        valid: list[dict] = []
        skipped = 0
        duration_dropped = 0
        total_sec = 0.0

        for idx, (text, _, _) in enumerate(pairs):
            if total_sec >= target_seconds:
                break
            vid = pair_vid[idx]
            if not text or vid is None:
                skipped += 1
                continue

            try:
                wav_bytes = server.synthesize(vid, tag + text, response_format="wav")
                data, native_sr = sf.read(io.BytesIO(wav_bytes))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                wav = np.asarray(data, dtype=np.float32)
            except Exception:
                skipped += 1
                continue

            wav = resample(wav, int(native_sr), int(sample_rate))
            dur = float(len(wav)) / sample_rate

            if dur < min_duration or dur > max_duration:
                duration_dropped += 1
                continue

            uid = f"{idx:07d}_{uuid.uuid4().hex[:8]}"
            rel = f"wavs/{uid}.wav"
            out = os.path.join(wav_dir, f"{uid}.wav")
            tmp = out + ".tmp.wav"
            sf.write(tmp, wav, int(sample_rate), subtype="PCM_16")
            os.replace(tmp, out)

            row = {
                "id": uid,
                "file": rel,
                "text": text,
                "duration": round(dur, 3),
            }
            if speaker_labels is not None:
                row["speaker"] = speaker_labels[idx]
            valid.append(row)
            total_sec += dur

            if on_clip:
                on_clip(total_sec)

            if save_every and len(valid) % save_every == 0:
                _save(valid)

    if len(valid) < min_samples:
        raise RuntimeError(
            f"Only {len(valid)} valid samples (need >={min_samples}). "
            f"{skipped} skipped, {duration_dropped} dropped by duration. Aborting."
        )

    # Final flush + push of everything generated.
    written = _flush(valid)
    if on_save is not None:
        try:
            on_save(out_dir)
        except Exception as e:
            if progress:
                progress(f"final save failed: {e}")

    return {
        "rows": len(valid),
        "hours": total_sec / 3600,
        "skipped": skipped,
        "duration_dropped": duration_dropped,
        "out_dir": out_dir,
        "written": written,
    }


def generate_asr(**kwargs) -> dict:
    """Generate an ASR dataset (``wavs/`` + ``metadata.jsonl``).

    Thin wrapper over :func:`generate` with ASR output. ``pairs`` typically come
    from a large, diverse pool of reference audio.
    """
    kwargs.setdefault("output_formats", ("asr",))
    return generate(**kwargs)


def generate_tts(**kwargs) -> dict:
    """Generate a TTS dataset (``wavs/`` + LJSpeech ``metadata.csv``).

    Thin wrapper over :func:`generate` with LJSpeech output. ``pairs`` typically
    come from a small set of speakers (the packaged voices by default); pass
    ``speaker_labels`` to record which speaker voiced each clip.
    """
    kwargs.setdefault("output_formats", ("ljspeech",))
    return generate(**kwargs)


# --------------------------------------------------------------------------- #
# Export manifests
# --------------------------------------------------------------------------- #
EXPORT_FORMATS = ("ljspeech", "asr")


def _manifest_text(s: str) -> str:
    return s.replace("\r", " ").replace("\n", " ").replace("|", " ").strip()


def export_formats(out_dir: str, formats) -> list[str]:
    out = Path(out_dir)
    rows = []
    with open(out / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    fmts = [f for f in formats if f in EXPORT_FORMATS]
    written: list[str] = []

    def _write(name, lines):
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(str(out / name))

    if "ljspeech" in fmts:
        _write("metadata.csv",
               [f"{r['id']}|{_manifest_text(r['text'])}|{_manifest_text(r['text'])}" for r in rows])
    if "asr" in fmts:
        _write("metadata.jsonl",
               [json.dumps({"audio": r["file"],
                            "text": _manifest_text(r["text"])})
                for r in rows])
    return written

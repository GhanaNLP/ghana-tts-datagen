"""Core synthetic-speech generation with VoxCPM (voice-cloned male/female).

Streams a text dataset, synthesises each row with the Ghana NLP Community VoxCPM
model using the built-in male/female reference speakers, trims silence, and writes
WAVs at the chosen sample rate + a manifest locally. Supports parallel model instances,
a target-hours budget, and resume.
"""

from __future__ import annotations

import json
import os
import random
import re
import queue
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("MODELSCOPE_CACHE", "/tmp/modelscope_cache")

MODEL_ID = "ghananlpcommunity/ghana-tts-36k"
SAMPLE_RATE = 16000      # native rate the model synthesises at
DEFAULT_SR = 22050       # default OUTPUT rate (TTS-friendly); override with --sample-rate
SILENCE_TOP_DB = 30
SILENCE_MAX_GAP_S = 0.3

_SPEAKER_DIR = Path(__file__).resolve().parent / "speakers"
SPEAKERS = {
    "male":   {"wav": str(_SPEAKER_DIR / "male.wav"),   "txt": _SPEAKER_DIR / "male.txt"},
    "female": {"wav": str(_SPEAKER_DIR / "female.wav"), "txt": _SPEAKER_DIR / "female.txt"},
}
for _g, _s in SPEAKERS.items():
    _s["text"] = Path(_s["txt"]).read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Pure helpers (no model / GPU needed — unit-testable)
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def pick_gender(idx: int, mode: str, male_pct: int) -> str:
    """Deterministic per-row voice so resumed runs assign the same speaker."""
    if mode in ("male", "all male"):
        return "male"
    if mode in ("female", "all female"):
        return "female"
    return "male" if (idx * 2654435761) % 100 < male_pct else "female"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-") or "run"


def trim_silences(wav, sr=SAMPLE_RATE, top_db=SILENCE_TOP_DB, max_gap_s=SILENCE_MAX_GAP_S):
    """Collapse long internal silences while keeping a small natural gap."""
    import librosa

    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim != 1 or wav.size == 0:
        return wav
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    max_gap = int(max_gap_s * sr)
    pieces, prev_end = [], None
    for start, end in intervals:
        if prev_end is not None:
            keep = min(start - prev_end, max_gap)
            if keep > 0:
                pieces.append(wav[prev_end:prev_end + keep])
        pieces.append(wav[start:end])
        prev_end = end
    return np.concatenate(pieces)


def resample(wav, src_sr: int, dst_sr: int):
    """Resample a mono float array from ``src_sr`` to ``dst_sr`` (no-op if equal)."""
    wav = np.asarray(wav, dtype=np.float32)
    if src_sr == dst_sr or wav.size == 0:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr)


def auto_instances(precision: str = "fp32") -> int:
    """Parallel model instances that fit in VRAM (~4.5 GB fp32, ~2.5 GB half)."""
    per = 2.5 if precision in ("fp16", "bf16") else 4.5
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return max(1, int((vram_gb * 0.8) // per))
    except Exception:
        return 1


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _cast_model(model, dt) -> None:
    """Best-effort cast of the model's torch modules to ``dt`` (fp16/bf16)."""
    import torch
    cast = False
    for obj in (model, getattr(model, "tts_model", None)):
        if obj is None:
            continue
        if isinstance(obj, torch.nn.Module):
            obj.to(dt)
            cast = True
        else:
            for v in vars(obj).values():
                if isinstance(v, torch.nn.Module):
                    v.to(dt)
                    cast = True
    if not cast:
        raise RuntimeError("could not apply half precision to this model build; use --precision fp32")


def load_instance(model_id: str = MODEL_ID, precision: str = "fp32"):
    from voxcpm import VoxCPM
    try:
        model = VoxCPM.from_pretrained(model_id, load_denoiser=False)
    except TypeError:
        model = VoxCPM.from_pretrained(model_id)
    if precision in ("fp16", "bf16"):
        import torch
        if precision == "bf16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise RuntimeError("bf16 needs an Ampere+ GPU (A100/L4/H100…); not available here. "
                               "Use --precision fp16 or fp32.")
        _cast_model(model, torch.float16 if precision == "fp16" else torch.bfloat16)
    return model


def resolve_speakers(overrides: dict | None = None) -> dict:
    """Merge optional speaker overrides with the built-in ``SPEAKERS``.

    ``overrides`` is a dict keyed by gender (``"male"``, ``"female"``) whose
    values can have ``"wav"`` (path to WAV) and optionally ``"text"`` (prompt
    transcript).  If ``"text"`` is omitted it is read from a sibling ``.txt``
    (the WAV path with ``.wav`` → ``.txt``).  Unspecified genders keep the
    bundled speaker.
    """
    resolved = dict(SPEAKERS)
    if overrides:
        for gender, sp in overrides.items():
            wav = sp.get("wav")
            if wav is None:
                continue
            txt_path = sp.get("txt") or Path(str(wav).replace(".wav", ".txt"))
            text = sp.get("text") or txt_path.read_text(encoding="utf-8").strip()
            resolved[gender] = {"wav": str(wav), "txt": str(txt_path), "text": text}
    return resolved


def _generate_one(model, caches: dict, text: str, gender: str,
                  cfg_value: float, steps: int,
                  speakers: dict | None = None) -> np.ndarray:
    spk = speakers or SPEAKERS
    if gender not in caches:
        sp = spk[gender]
        caches[gender] = model.tts_model.build_prompt_cache(
            prompt_text=sp["text"], prompt_wav_path=sp["wav"]
        )
    wav, _, _ = model.tts_model.generate_with_prompt_cache(
        target_text=text,
        prompt_cache=caches[gender],
        max_len=4096,
        cfg_value=float(cfg_value),
        inference_timesteps=int(steps),
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )
    if hasattr(wav, "cpu"):
        wav = wav.squeeze(0).cpu().numpy()
    return trim_silences(wav)


# --------------------------------------------------------------------------- #
# Generation run (local output, parallel workers, resume)
# --------------------------------------------------------------------------- #
class _Run:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.feeding_done = False
        self.rows: dict[str, dict] = {}
        self.total_seconds = 0.0
        self.errors = 0
        self.duration_dropped = 0
        self.fatal = ""
        self.run_id = ""
        self.wav_dir = ""
        self.cfg_value = 2.0
        self.steps = 10
        self.sample_rate = DEFAULT_SR
        self.precision = "fp32"
        self.speakers: dict | None = None
        self.min_duration: float | None = None
        self.max_duration: float | None = None


def _worker(run: _Run, model_id: str):
    try:
        model = load_instance(model_id, run.precision)
    except Exception as e:
        run.fatal = run.fatal or f"model load failed: {e}"
        run.stop.set()
        return
    caches: dict = {}
    while not (run.stop.is_set() and run.q.empty()):
        try:
            idx, text, gender = run.q.get(timeout=2)
        except queue.Empty:
            if run.feeding_done:
                break
            continue
        try:
            wav = _generate_one(model, caches, text, gender, run.cfg_value, run.steps,
                                speakers=run.speakers)
            wav = resample(wav, SAMPLE_RATE, run.sample_rate)
            dur = float(len(wav)) / run.sample_rate
            if (run.min_duration is not None and dur < run.min_duration) or \
               (run.max_duration is not None and dur > run.max_duration):
                with run.lock:
                    run.duration_dropped += 1
                continue
            uid = f"{idx:07d}_{run.run_id}"
            rel = f"wavs/{uid}.wav"
            out = os.path.join(run.wav_dir, f"{uid}.wav")
            tmp = out + ".tmp"
            sf.write(tmp, wav, run.sample_rate, subtype="PCM_16")
            os.replace(tmp, out)
            with run.lock:
                run.rows[str(idx)] = {
                    "id": uid, "file": rel, "text": text,
                    "gender": gender, "speaker": gender, "duration": round(dur, 3),
                }
                run.total_seconds += dur
        except Exception as e:
            with run.lock:
                run.errors += 1
        finally:
            run.q.task_done()


def _write_manifest(out_dir: str, run: _Run, meta: dict):
    with run.lock:
        rows = dict(run.rows)
        total = run.total_seconds
    progress = {**meta, "run_id": run.run_id, "total_seconds": round(total, 2),
                "rows": rows,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    Path(out_dir, "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    with open(Path(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for k in sorted(rows, key=int):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")


def generate(
    *,
    out_dir: str,
    dataset: str | None = None,
    text_column: str | None = None,
    texts: list | None = None,
    config: str | None = None,
    split: str = "train",
    target_hours: float = 1.0,
    voices: str = "custom",
    male_pct: int = 50,
    sample_rate: int = DEFAULT_SR,
    precision: str = "fp32",
    instances: int | None = None,
    cfg_value: float = 2.0,
    steps: int = 10,
    max_chars: int = 400,
    max_samples: int | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    model_id: str = MODEL_ID,
    token: str | None = None,
    save_every: int = 200,
    speakers: dict | None = None,
    on_clip=None,
    on_save=None,
    progress=None,
):
    """Generate synthetic speech for a dataset into ``out_dir``.

    Source is either an HF ``dataset`` (+ ``text_column``) or an in-memory list of
    ``texts`` (one sentence each). Writes ``wavs/*.wav`` (mono, at ``sample_rate``),
    ``manifest.jsonl`` and ``progress.json``. Resumes automatically if
    ``out_dir/progress.json`` exists (skips done rows). ``on_clip(seconds)`` fires
    as clips land; ``progress(msg)`` for status. Returns a summary dict.

    ``max_samples`` — randomly pick at most this many rows from the source (for
    randomised sub-sampling of large datasets).  Applied *before* generation.

    ``min_duration`` / ``max_duration`` — skip generated clips whose audio length
    (seconds) falls outside this range.  Dropped clips count toward the summary
    ``"duration_dropped"`` field.

    ``speakers`` — optional dict keyed by gender (``"male"``, ``"female"``) with
    ``"wav"`` and optionally ``"text"`` keys.  If omitted the bundled reference
    speakers are used.  See :func:`resolve_speakers`.
    ``on_save(out_dir)`` — called after each manifest write (every ``save_every``
    rows and at the end).  Useful for incremental push to Hugging Face.
    """
    if texts is None and not (dataset and text_column):
        raise ValueError("Provide either texts=[...] or dataset=... with text_column=...")

    out_dir = str(out_dir)
    os.makedirs(os.path.join(out_dir, "wavs"), exist_ok=True)
    target_seconds = max(0.0, float(target_hours)) * 3600

    run = _Run()
    run.cfg_value, run.steps = cfg_value, steps
    run.sample_rate = int(sample_rate)
    run.precision = precision
    run.wav_dir = os.path.join(out_dir, "wavs")
    run.speakers = resolve_speakers(speakers)
    run.min_duration = min_duration
    run.max_duration = max_duration

    prog_path = Path(out_dir, "progress.json")
    if prog_path.exists():
        prev = json.loads(prog_path.read_text(encoding="utf-8"))
        run.run_id = prev.get("run_id") or uuid.uuid4().hex[:8]
        run.rows = dict(prev.get("rows", {}))
        run.total_seconds = float(prev.get("total_seconds", 0.0))
        if progress:
            progress(f"resuming {out_dir} — {len(run.rows)} rows / "
                     f"{run.total_seconds/3600:.2f} h already done")
    else:
        run.run_id = uuid.uuid4().hex[:8]

    n_inst = instances if instances and instances > 0 else auto_instances(precision)
    if progress:
        progress(f"loading {n_inst} model instance(s)…")
    workers = [threading.Thread(target=_worker, args=(run, model_id), daemon=True)
               for _ in range(n_inst)]
    for w in workers:
        w.start()

    meta = {"model_id": model_id, "dataset": dataset or "(text-file)",
            "config": config or "", "split": split, "text_column": text_column or "",
            "voices": voices, "male_pct": male_pct, "target_hours": target_hours,
            "sample_rate": int(sample_rate)}

    if texts is not None:
        src_list = list(texts)
        if max_samples and len(src_list) > max_samples:
            src_list = random.sample(src_list, max_samples)
        source = ((i, t) for i, t in enumerate(src_list))
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
        if max_samples:
            ds = ds.shuffle(seed=42).take(max_samples)
        source = ((i, ex.get(text_column, "")) for i, ex in enumerate(ds))

    staged = 0
    for idx, raw in source:
        if run.stop.is_set() or run.fatal:
            break
        if run.total_seconds >= target_seconds:
            break
        if str(idx) in run.rows:
            continue
        text = clean_text(raw)
        if not (2 <= len(text) <= max_chars):
            continue
        gender = pick_gender(idx, voices, male_pct)
        while not run.stop.is_set():
            try:
                run.q.put((idx, text, gender), timeout=2)
                break
            except queue.Full:
                pass
        done = len(run.rows)
        if on_clip:
            on_clip(run.total_seconds)
        staged += 1
        if staged >= save_every:
            _write_manifest(out_dir, run, meta)
            if on_save:
                on_save(out_dir)
            staged = 0

    run.feeding_done = True
    for w in workers:
        w.join()
    _write_manifest(out_dir, run, meta)
    if on_save:
        on_save(out_dir)

    if run.fatal:
        raise RuntimeError(run.fatal)
    return {"rows": len(run.rows), "hours": run.total_seconds / 3600,
            "errors": run.errors, "duration_dropped": run.duration_dropped,
            "out_dir": out_dir, "run_id": run.run_id}


def preview(*, out_dir, dataset=None, text_column=None, texts=None, config=None,
            split="train", voices="custom", male_pct=50, sample_rate=DEFAULT_SR,
            precision="fp32", cfg_value=2.0, steps=10, n=5, max_chars=400,
            model_id=MODEL_ID, token=None, speakers=None):
    """Generate ``n`` preview clips into ``out_dir/preview`` and return their info.

    Source is ``texts=[...]`` or an HF ``dataset`` + ``text_column``.
    ``speakers`` — optional overrides, see :func:`generate`.
    """
    pdir = Path(out_dir, "preview")
    pdir.mkdir(parents=True, exist_ok=True)
    model = load_instance(model_id, precision)
    caches: dict = {}
    spk = resolve_speakers(speakers)
    if texts is not None:
        source = ((i, t) for i, t in enumerate(texts))
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
        source = ((i, ex.get(text_column, "")) for i, ex in enumerate(ds))
    out = []
    for idx, raw in source:
        if len(out) >= n:
            break
        text = clean_text(raw)
        if not (2 <= len(text) <= max_chars):
            continue
        gender = pick_gender(idx, voices, male_pct)
        wav = _generate_one(model, caches, text, gender, cfg_value, steps,
                            speakers=spk)
        wav = resample(wav, SAMPLE_RATE, int(sample_rate))
        path = pdir / f"preview_{len(out)+1}_{gender}.wav"
        sf.write(str(path), wav, int(sample_rate), subtype="PCM_16")
        out.append({"file": str(path), "gender": gender,
                    "duration": round(len(wav) / int(sample_rate), 2), "text": text})
    return out


# --------------------------------------------------------------------------- #
# Export manifests (written beside the generated wavs/)
# --------------------------------------------------------------------------- #
EXPORT_FORMATS = ("ljspeech", "asr")


def _manifest_text(s: str) -> str:
    return s.replace("\r", " ").replace("\n", " ").replace("|", " ").strip()


def export_formats(out_dir: str, formats) -> list[str]:
    """Write standard-formatted manifests for an existing run (beside ``wavs/``).

    Reads ``manifest.jsonl`` and emits, per requested format:
      * ``ljspeech`` → ``metadata.csv``  ``id|text|text`` (TTS)
      * ``asr``      → ``metadata.jsonl``  ``{"audio": "wavs/...", "text": "..."}``
    Returns the paths written.
    """
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

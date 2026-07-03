"""Command-line interface for Ghana Speech Datagen.

Subcommands:
  tts   Generate synthetic speech from text (needs GPU)
  asr   Prepare ASR training data from existing audio+text (no GPU needed)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

from .generator import DEFAULT_SR, MODEL_ID, sanitize_name

DATASET_ORG = "ghananlpcommunity"
MIN_ASR_SAMPLES = 50


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _resolve_token(args) -> str:
    tok = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            import getpass
            tok = getpass.getpass(
                "HF Token (required -- needed to push to your HF account): "
            ).strip()
        except (EOFError, OSError):
            tok = ""
        if not tok:
            sys.exit("No token provided. Set --token or the HF_TOKEN env var.")
    os.environ["HF_TOKEN"] = tok
    return tok


def _push_repo(name: str, token: str, push: str | None = None, private: bool = False) -> str:
    from huggingface_hub import HfApi, create_repo
    if push:
        repo_id = push
    else:
        who = HfApi(token=token).whoami()
        repo_id = f"{who['name']}/ghana-speech-synth-{name}"
    create_repo(repo_id, repo_type="dataset", token=token, private=private, exist_ok=True)
    return repo_id


def _upload(out_dir: str, repo_id: str, token: str, msg: str = "update"):
    from huggingface_hub import HfApi
    HfApi(token=token).upload_folder(
        folder_path=out_dir,
        path_in_repo=os.path.basename(out_dir.rstrip("/")),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
    )


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ghana-speech-datagen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- tts ----
    tts = sub.add_parser("tts", help="Generate synthetic speech from text (needs GPU)",
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    tts_src = tts.add_argument_group("source")
    tts_src.add_argument("--dataset", help="text dataset id on the HF Hub")
    tts_src.add_argument("--config", help="dataset config (optional)")
    tts_src.add_argument("--split", default="train")
    tts_src.add_argument("--text", dest="text_column",
                         help="column with the text to synthesise (with --dataset)")
    tts_src.add_argument("--text-file", help="path to a .txt file, one sentence per line")
    tts_src.add_argument("--max-chars", type=int, default=400,
                         help="skip rows longer than this (default 400)")

    tts_gen = tts.add_argument_group("generation")
    tts_gen.add_argument("--hours", type=float, default=1.0, help="target hours of audio")
    tts_gen.add_argument("--voices", choices=["custom", "male", "female"], default="custom")
    tts_gen.add_argument("--male-pct", type=int, default=50, help="%% male in custom mode")
    tts_gen.add_argument("--sample-rate", type=int, default=DEFAULT_SR,
                         help=f"output WAV rate (default {DEFAULT_SR})")
    tts_gen.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                         help="model precision")
    tts_gen.add_argument("--instances", type=int,
                         help="parallel model instances (default: auto by VRAM)")
    tts_gen.add_argument("--cfg", type=float, default=2.0, dest="cfg_value", help="CFG value")
    tts_gen.add_argument("--steps", type=int, default=10, help="inference timesteps")
    tts_gen.add_argument("--model", default=MODEL_ID, help="VoxCPM model id")
    tts_gen.add_argument("--max-samples", type=int,
                         help="randomly pick at most this many texts")
    tts_gen.add_argument("--min-duration", type=float,
                         help="skip clips shorter than this (seconds)")
    tts_gen.add_argument("--max-duration", type=float,
                         help="skip clips longer than this (seconds)")

    tts_out = tts.add_argument_group("output")
    tts_out.add_argument("--out", help="output directory (default: data/<name>)")
    tts_out.add_argument("--name", help="run name; enables resume")
    tts_out.add_argument("--save-every", type=int, default=200,
                         help="write manifest every N rows")
    tts_out.add_argument("--push", metavar="REPO_ID",
                         help="override auto-generated HF dataset repo")
    tts_out.add_argument("--private", action="store_true",
                         help="make the dataset repo private")
    tts_out.add_argument("--token", help="HF token (required for private model)")

    tts_spk = tts.add_argument_group("speaker reference audio")
    tts_spk.add_argument("--speaker-dir",
                         help="dir with speaker WAV files and optional <id>.txt sidecars; "
                              "speaker IDs are derived from filenames")
    tts_spk.add_argument("--ref-text",
                         help="fallback prompt text for all custom speakers "
                              "(ignored when a speaker has its own <id>.txt)")

    tts_misc = tts.add_argument_group("misc")
    tts_misc.add_argument("--preview", type=int, metavar="N",
                          help="generate N preview clips and exit")
    tts_misc.add_argument("--list-datasets", action="store_true",
                          help=f"list datasets under the {DATASET_ORG} org")

    # ---- asr ----
    asr = sub.add_parser("asr", help="Generate synthetic speech using reference audio pool (GPU required)",
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    asr_txt = asr.add_argument_group("text source (provide one)")
    asr_txt.add_argument("--dataset", help="HF dataset with text to synthesise")
    asr_txt.add_argument("--text", dest="text_column",
                         help="column with text to synthesise (with --dataset)")
    asr_txt.add_argument("--text-file", help="path to a .txt file, one sentence per line")
    asr_txt.add_argument("--config", help="dataset config (optional)")
    asr_txt.add_argument("--split", default="train")

    asr_ref = asr.add_argument_group("reference audio source (provide one)")
    asr_ref.add_argument("--ref-dataset", help="HF dataset id with reference audio+transcript columns")
    asr_ref.add_argument("--audio-column", default="audio",
                         help="column with reference audio (default: audio)")
    asr_ref.add_argument("--ref-text-column", default="text",
                         help="column with reference transcripts (default: text)")
    asr_ref.add_argument("--ref-config", help="ref dataset config (optional)")
    asr_ref.add_argument("--ref-split", default="train")
    asr_ref.add_argument("--ref-audio-dir",
                         help="local dir with reference audio files (use with --ref-metadata)")
    asr_ref.add_argument("--ref-metadata",
                         help="CSV/JSONL mapping ref audio filenames to transcripts")

    asr_val = asr.add_argument_group("generation")
    asr_val.add_argument("--hours", type=float, default=1.0, help="target hours of audio")
    asr_val.add_argument("--min-samples", type=int, default=MIN_ASR_SAMPLES,
                         help=f"minimum valid samples required (default {MIN_ASR_SAMPLES})")
    asr_val.add_argument("--min-duration", type=float, default=1.0,
                         help="drop generated clips shorter than this (seconds)")
    asr_val.add_argument("--max-duration", type=float, default=30.0,
                         help="drop generated clips longer than this (seconds)")
    asr_val.add_argument("--max-samples", type=int,
                         help="randomly pick at most this many texts")

    asr_gen = asr.add_argument_group("model")
    asr_gen.add_argument("--sample-rate", type=int, default=DEFAULT_SR,
                         help=f"output WAV rate (default {DEFAULT_SR})")
    asr_gen.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                         help="model precision")
    asr_gen.add_argument("--cfg", type=float, default=2.0, dest="cfg_value",
                         help="CFG value")
    asr_gen.add_argument("--steps", type=int, default=10, help="inference timesteps")
    asr_gen.add_argument("--model", default=MODEL_ID, help="VoxCPM model id")

    asr_out = asr.add_argument_group("output")
    asr_out.add_argument("--out", help="output directory (default: data/<name>)")
    asr_out.add_argument("--name",
                         help="output name (default: dataset or audio-dir name)")
    asr_out.add_argument("--push", metavar="REPO_ID",
                         help="override auto-generated HF dataset repo")
    asr_out.add_argument("--private", action="store_true",
                         help="make the dataset repo private")
    asr_out.add_argument("--token", help="HF token (for pushing)")

    asr_misc = asr.add_argument_group("misc")
    asr_misc.add_argument("--list-datasets", action="store_true",
                          help=f"list datasets under the {DATASET_ORG} org")

    return p


# --------------------------------------------------------------------------- #
# TTS flow
# --------------------------------------------------------------------------- #

def _build_speakers(args) -> dict | None:
    if not args.speaker_dir:
        return None
    d = Path(args.speaker_dir)
    overrides: dict = {}
    for wav_path in sorted(d.glob("*.wav")):
        spk_id = wav_path.stem
        txt_path = wav_path.with_suffix(".txt")
        text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else (args.ref_text or "")
        if not text:
            sys.exit(f"Speaker '{spk_id}': no {txt_path.name} sidecar and no --ref-text provided. "
                     f"Either create {txt_path.name} or pass --ref-text.")
        overrides[spk_id] = {"wav": str(wav_path), "text": text}
    if not overrides:
        sys.exit(f"No .wav files found in --speaker-dir: {args.speaker_dir}")
    if len(overrides) > 10:
        sys.exit(f"At most 10 speakers supported, got {len(overrides)}")
    return overrides


def _cmd_tts(args):
    from . import generator

    texts = None
    if args.text_file:
        texts = [ln.strip() for ln in open(args.text_file, encoding="utf-8") if ln.strip()]
        default_name = sanitize_name(os.path.splitext(os.path.basename(args.text_file))[0])
    elif args.dataset and args.text_column:
        default_name = sanitize_name(args.dataset.split("/")[-1])
    else:
        sys.exit("Provide --text-file PATH, or --dataset ID with --text COL.")

    speakers = _build_speakers(args)
    name = args.name or default_name
    out_dir = args.out or os.path.join("data", name)

    if args.preview:
        clips = generator.preview(
            out_dir=out_dir, dataset=args.dataset, text_column=args.text_column, texts=texts,
            config=args.config, split=args.split, voices=args.voices, male_pct=args.male_pct,
            sample_rate=args.sample_rate, precision=args.precision,
            cfg_value=args.cfg_value, steps=args.steps,
            n=args.preview, max_chars=args.max_chars, model_id=args.model, token=args.token,
            speakers=speakers,
        )
        for c in clips:
            print(f"  [{c['spk_id']}] {c['duration']}s  {c['file']}\n      {c['text'][:90]}")
        return 0

    token = _resolve_token(args)
    push_repo = _push_repo(name, token, args.push, args.private)
    push_url = f"https://huggingface.co/datasets/{push_repo}"
    print(f"Dataset will be pushed to: {push_url}", file=sys.stderr)

    from tqdm.auto import tqdm
    bar = tqdm(total=round(args.hours * 3600), unit="s", unit_scale=False,
               desc="Synthesising audio", file=sys.stderr)
    state = {"last": 0.0}

    def _on_clip(total_sec):
        delta = total_sec - state["last"]
        if delta > 0:
            bar.update(delta)
            state["last"] = total_sec

    def _on_save(dir_path):
        _upload(dir_path, push_repo, token,
                msg=f"synth data: {bar.n:.0f}s / {bar.total:.0f}s")

    summary = generator.generate(
        out_dir=out_dir, dataset=args.dataset, text_column=args.text_column, texts=texts,
        config=args.config, split=args.split, target_hours=args.hours,
        voices=args.voices, male_pct=args.male_pct, sample_rate=args.sample_rate,
        precision=args.precision, instances=args.instances,
        cfg_value=args.cfg_value, steps=args.steps, max_chars=args.max_chars,
        max_samples=args.max_samples,
        min_duration=args.min_duration, max_duration=args.max_duration,
        model_id=args.model, token=token, save_every=args.save_every,
        speakers=speakers,
        on_clip=_on_clip, on_save=_on_save,
        progress=lambda m: bar.set_description(m[:48]),
    )
    bar.close()

    written = generator.export_formats(out_dir, ["ljspeech"])

    dropped = summary.get("duration_dropped", 0)
    print(f"\n✅ {summary['rows']} clips · {summary['hours']:.2f} h "
          f"({summary['errors']} errors"
          + (f", {dropped} dropped by duration" if dropped else "")
          + f") → {summary['out_dir']}", file=sys.stderr)
    print("   wavs/  manifest.jsonl  progress.json"
          + ("  " + "  ".join(os.path.basename(w) for w in written) if written else ""),
          file=sys.stderr)
    print(f"   pushed to {push_url}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# ASR flow  (generate with reference audio pool)
# --------------------------------------------------------------------------- #

def _load_texts(dataset: str | None, text_column: str | None,
                text_file: str | None, config: str | None, split: str,
                max_samples: int | None, token: str | None) -> list[str]:
    if text_file:
        texts = [ln.strip() for ln in open(text_file, encoding="utf-8") if ln.strip()]
    elif dataset and text_column:
        from datasets import load_dataset
        ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
        if max_samples:
            ds = ds.shuffle(seed=42).take(max_samples)
        texts = [ex.get(text_column, "").strip() for ex in ds]
    else:
        sys.exit("Provide --dataset + --text, or --text-file with texts to synthesise.")
    texts = [t for t in texts if 2 <= len(t) <= 400]
    if max_samples and len(texts) > max_samples:
        texts = random.sample(texts, max_samples)
    if not texts:
        sys.exit("No valid texts found (need 2-400 chars each).")
    return texts


def _load_refs_from_dataset(dataset: str, audio_col: str, text_col: str,
                             config: str | None, split: str,
                             max_samples: int | None, token: str):
    from datasets import load_dataset
    ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
    if max_samples:
        ds = ds.shuffle(seed=42).take(max_samples)
    refs = []
    for ex in ds:
        audio = ex.get(audio_col)
        text = ex.get(text_col)
        if audio is None or text is None:
            continue
        refs.append((audio, str(text).strip()))
    if not refs:
        sys.exit(f"No valid reference audio+text pairs found in {dataset}.")
    return refs


def _load_refs_from_local(audio_dir: str, metadata_path: str,
                           max_samples: int | None) -> list:
    audio_dir = Path(audio_dir)
    meta = Path(metadata_path)
    rows = []
    if meta.suffix == ".jsonl":
        with open(meta, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with open(meta, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    if max_samples and len(rows) > max_samples:
        rows = random.sample(rows, max_samples)
    refs = []
    for row in rows:
        audio_path = row.get("audio") or row.get("file") or row.get("path", "")
        text = row.get("text") or row.get("transcript") or row.get("sentence", "")
        if audio_path and text:
            refs.append((str(audio_dir / audio_path), text.strip()))
    if not refs:
        sys.exit(f"No valid reference audio+text pairs found in {audio_dir}.")
    return refs


def _cmd_asr(args):
    from . import generator

    token = _resolve_token(args) if (args.push or args.dataset) else None

    texts = _load_texts(args.dataset, args.text_column, args.text_file,
                        args.config, args.split, args.max_samples, token)

    if args.ref_dataset:
        refs = _load_refs_from_dataset(
            args.ref_dataset, args.audio_column, args.ref_text_column,
            args.ref_config, args.ref_split, None, token,
        )
        default_name = sanitize_name(args.ref_dataset.split("/")[-1])
    elif args.ref_audio_dir and args.ref_metadata:
        refs = _load_refs_from_local(args.ref_audio_dir, args.ref_metadata, None)
        default_name = sanitize_name(os.path.basename(args.ref_audio_dir.rstrip("/")))
    else:
        sys.exit("Provide --ref-dataset or --ref-audio-dir + --ref-metadata.")

    # Pair texts with random refs
    random.shuffle(refs)
    pairs = []
    for i, t in enumerate(texts):
        ref = refs[i % len(refs)]
        pairs.append((t, ref[0], ref[1]))

    name = args.name or default_name
    out_dir = args.out or os.path.join("data", name)

    from tqdm.auto import tqdm
    target_seconds = round(args.hours * 3600)
    bar = tqdm(total=target_seconds, unit="s", unit_scale=False,
               desc="Synthesising ASR clips", file=sys.stderr)
    state = {"last": 0.0}

    def _on_clip(dur):
        delta = dur - state["last"]
        if delta > 0:
            bar.update(delta)
            state["last"] = dur

    summary = generator.generate_asr(
        out_dir=out_dir, pairs=pairs,
        min_duration=args.min_duration, max_duration=args.max_duration,
        min_samples=args.min_samples,
        target_seconds=target_seconds,
        sample_rate=args.sample_rate, precision=args.precision,
        cfg_value=args.cfg_value, steps=args.steps,
        model_id=args.model, token=token,
        on_clip=_on_clip,
        progress=lambda m: bar.set_description(m[:48]),
    )
    bar.close()

    print(f"\n✅ {summary['rows']} clips · {summary['hours']:.2f} h "
          f"({summary['skipped']} skipped, "
          f"{summary['duration_dropped']} dropped by duration)"
          f" → {summary['out_dir']}", file=sys.stderr)
    print("   wavs/  manifest.jsonl  metadata.jsonl", file=sys.stderr)

    if args.push is not None or args.dataset:
        push_repo = _push_repo(name, token, args.push, args.private)
        push_url = f"https://huggingface.co/datasets/{push_repo}"
        _upload(out_dir, push_repo, token, msg=f"asr data: {summary['rows']} clips / {summary['hours']:.2f}h")
        print(f"   pushed to {push_url}", file=sys.stderr)

    return 0


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_datasets:
        token = os.environ.get("HF_TOKEN") or ""
        from huggingface_hub import HfApi
        ids = sorted(d.id for d in HfApi(token=token).list_datasets(author=DATASET_ORG, limit=500))
        print("\n".join(ids) if ids else f"(no datasets found under {DATASET_ORG})")
        return 0

    if args.command == "tts":
        return _cmd_tts(args)
    elif args.command == "asr":
        return _cmd_asr(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

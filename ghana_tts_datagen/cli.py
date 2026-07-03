"""Command-line interface for the Ghana TTS VoxCPM synthetic-speech generator.

Generate synthetic TTS training data from a text dataset, locally. Examples:

    # Preview 5 clips before a big run (needs a GPU)
    ghana-tts-datagen --dataset ghananlpcommunity/some-text --text-column text --preview 5

    # Generate 5 hours, 50/50 male/female, into data/<name>
    ghana-tts-datagen --dataset ghananlpcommunity/some-text --text-column text \\
        --hours 5 --name twi-run

    # Data is auto-pushed to your HF account every 200 rows (no loss on crash)
    ghana-tts-datagen --dataset … --text-column text --hours 5 --name twi-run

The model is **private** — set HF_TOKEN in your environment or pass --token.
Data is automatically pushed to a dataset repo on your HF account as it's
generated (every --save-every rows). Use --push REPO_ID to override the
auto-generated repo name, or --private to make it private.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .generator import DEFAULT_SR, EXPORT_FORMATS, MODEL_ID, sanitize_name

DATASET_ORG = "ghananlpcommunity"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ghana-tts-datagen", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("source (use --dataset OR --text-file)")
    src.add_argument("--dataset", help="source text dataset id on the HF Hub")
    src.add_argument("--config", help="dataset config (optional)")
    src.add_argument("--split", default="train")
    src.add_argument("--text-column", help="column holding the text (with --dataset)")
    src.add_argument("--text-file", help="path to a .txt file with one sentence per line")
    src.add_argument("--max-chars", type=int, default=400, help="skip rows longer than this")

    gen = p.add_argument_group("generation")
    gen.add_argument("--hours", type=float, default=1.0, help="target hours of audio")
    gen.add_argument("--voices", choices=["custom", "male", "female"], default="custom")
    gen.add_argument("--male-pct", type=int, default=50, help="%% male in custom mode")
    gen.add_argument("--sample-rate", type=int, default=DEFAULT_SR,
                     help=f"output WAV sample rate in Hz (default {DEFAULT_SR}; "
                          "e.g. 24000/44100 for your TTS framework)")
    gen.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                     help="model precision. fp32 = safest (default). fp16 ≈ half VRAM & "
                          "faster, but may degrade quality / NaN on some TTS models. "
                           "bf16 ≈ half VRAM, more stable than fp16, but needs an Ampere+ "
                           "GPU (A100, L4, H100, H200…).")
    gen.add_argument("--instances", type=int, help="parallel model instances (default: auto by VRAM)")
    gen.add_argument("--cfg", type=float, default=2.0, dest="cfg_value", help="CFG value")
    gen.add_argument("--steps", type=int, default=10, help="inference timesteps")
    gen.add_argument("--model", default=MODEL_ID, help="VoxCPM model id")
    gen.add_argument("--max-samples", type=int,
                     help="randomly pick at most this many texts (randomised sub-sample)")
    gen.add_argument("--min-duration", type=float,
                     help="skip clips shorter than this many seconds")
    gen.add_argument("--max-duration", type=float,
                     help="skip clips longer than this many seconds")

    out = p.add_argument_group("output")
    out.add_argument("--out", help="output directory (default: data/<name>)")
    out.add_argument("--name", help="run name (folder under data/; enables resume)")
    out.add_argument("--format", default="ljspeech",
                     help=f"export format(s) (comma list): {','.join(EXPORT_FORMATS)}")
    out.add_argument("--save-every", type=int, default=200, help="write manifest every N rows")
    out.add_argument("--push", metavar="REPO_ID",
                     help="override auto-generated HF dataset repo (default: <user>/ghana-tts-synth-<name>)")
    out.add_argument("--private", action="store_true",
                     help="make the dataset repo private (default: public)")
    out.add_argument("--token", help="HF token (required; falls back to HF_TOKEN env)")

    spk = p.add_argument_group("speaker reference audio (default: bundled Twi male/female)")
    spk.add_argument("--speaker-dir",
                     help="directory with male.wav+txt and female.wav+txt (overrides all below)")
    spk.add_argument("--speaker-male", metavar="WAV",
                     help="custom male reference WAV (transcript: sibling .txt file)")
    spk.add_argument("--speaker-male-text",
                     help="male prompt transcript (skip if using sibling .txt)")
    spk.add_argument("--speaker-female", metavar="WAV",
                     help="custom female reference WAV (transcript: sibling .txt file)")
    spk.add_argument("--speaker-female-text",
                     help="female prompt transcript (skip if using sibling .txt)")

    misc = p.add_argument_group("misc")
    misc.add_argument("--preview", type=int, metavar="N",
                      help="generate N preview clips and exit (no full run)")
    misc.add_argument("--list-datasets", action="store_true",
                      help=f"list datasets under the {DATASET_ORG} org and exit")
    return p


def _build_speakers(args) -> dict | None:
    """Convert CLI speaker args to the speakers dict expected by generator."""
    overrides: dict = {}
    if args.speaker_dir:
        d = Path(args.speaker_dir)
        overrides["male"] = {"wav": str(d / "male.wav"), "txt": d / "male.txt"}
        overrides["female"] = {"wav": str(d / "female.wav"), "txt": d / "female.txt"}
        return overrides
    if args.speaker_male:
        m: dict = {"wav": args.speaker_male}
        if args.speaker_male_text:
            m["text"] = args.speaker_male_text
        overrides["male"] = m
    if args.speaker_female:
        f: dict = {"wav": args.speaker_female}
        if args.speaker_female_text:
            f["text"] = args.speaker_female_text
        overrides["female"] = f
    return overrides or None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not args.token:
        try:
            import getpass
            args.token = getpass.getpass(
                "HF Token (required — used to load the private model and push your\n"
                "              generated dataset to your HF account): "
            ).strip()
        except (EOFError, OSError):
            args.token = ""
        if not args.token:
            sys.exit("No token provided. Set --token or the HF_TOKEN env var.")

    os.environ["HF_TOKEN"] = args.token

    if args.list_datasets:
        from huggingface_hub import HfApi
        ids = sorted(d.id for d in HfApi(token=args.token).list_datasets(author=DATASET_ORG, limit=500))
        print("\n".join(ids) if ids else f"(no datasets found under {DATASET_ORG})")
        return 0

    texts = None
    if args.text_file:
        texts = [ln.strip() for ln in open(args.text_file, encoding="utf-8") if ln.strip()]
        default_name = sanitize_name(os.path.splitext(os.path.basename(args.text_file))[0])
    elif args.dataset and args.text_column:
        default_name = sanitize_name(args.dataset.split("/")[-1])
    else:
        sys.exit("Provide --text-file PATH, or --dataset ID with --text-column COL (see --help).")

    from . import generator

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
            print(f"  [{c['gender']}] {c['duration']}s  {c['file']}\n      {c['text'][:90]}")
        return 0

    from huggingface_hub import HfApi, create_repo

    # Resolve push repo — user override or auto-generate from their HF username
    if args.push:
        push_repo = args.push
    else:
        who = HfApi(token=args.token).whoami()
        push_repo = f"{who['name']}/ghana-tts-synth-{name}"
    create_repo(push_repo, repo_type="dataset", token=args.token,
                private=args.private, exist_ok=True)
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
        HfApi(token=args.token).upload_folder(
            folder_path=dir_path, path_in_repo=os.path.basename(dir_path.rstrip("/")),
            repo_id=push_repo, repo_type="dataset",
            commit_message=f"synth data: {bar.n:.0f}s / {bar.total:.0f}s",
        )

    summary = generator.generate(
        out_dir=out_dir, dataset=args.dataset, text_column=args.text_column, texts=texts,
        config=args.config, split=args.split, target_hours=args.hours,
        voices=args.voices, male_pct=args.male_pct, sample_rate=args.sample_rate,
        precision=args.precision, instances=args.instances,
        cfg_value=args.cfg_value, steps=args.steps, max_chars=args.max_chars,
        max_samples=args.max_samples,
        min_duration=args.min_duration, max_duration=args.max_duration,
        model_id=args.model, token=args.token, save_every=args.save_every,
        speakers=speakers,
        on_clip=_on_clip, on_save=_on_save,
        progress=lambda m: bar.set_description(m[:48]),
    )
    bar.close()

    fmts = [f.strip() for f in (args.format or "").split(",") if f.strip()]
    written = generator.export_formats(out_dir, fmts) if fmts else []

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


if __name__ == "__main__":
    raise SystemExit(main())

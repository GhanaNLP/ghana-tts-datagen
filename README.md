# Ghana Speech Datagen

Turn a **text dataset** into **synthetic speech training data** (TTS / ASR) — streamed through
the Ghana NLP Community VoxCPM model (`ghana-tts-36k`), voice-cloning built-in
male/female reference speakers. It writes WAVs (resampled to your target rate) +
a manifest locally, runs multiple model instances in parallel, and resumes where
it left off.

> **A GPU is required** for usable speed (VoxCPM is a neural TTS model; ~4.5 GB
> VRAM per instance). Instances are auto-sized from your GPU's VRAM.
>
> No GPU? Use the **Colab** or **Modal** runners below, or the `asr` subcommand
> (repackages existing audio — no GPU needed).

## Supported languages

The `ghana-tts-36k` model supports **41+ Ghanaian languages**. See the model card
at [hf.co/ghananlpcommunity/ghana-tts-36k](https://huggingface.co/ghananlpcommunity/ghana-tts-36k)
for the full list.

## Run in the cloud

**Colab** — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GhanaNLP/ghana-speech-datagen/blob/main/examples/ghana_speech_datagen.ipynb)

Pick a **GPU** runtime (`Runtime → Change runtime type → GPU`).

**Kaggle** — [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com)  
Open the notebook via Kaggle's GitHub import: `https://github.com/GhanaNLP/ghana-speech-datagen/blob/main/examples/ghana_speech_datagen.ipynb`

**Modal** (serverless GPU) — no local setup needed:

```bash
pip install modal
modal run examples/modal_run.py --dataset ghananlpcommunity/your-text-dataset --text text --hours 2
```

Requires a `hf-token` Modal secret with your Hugging Face token. See [`examples/modal_run.py`](examples/modal_run.py) for all options.

> Note: the Modal runner uses `ghana-speech-datagen tts` under the hood — it
> generates audio. It does not run the `asr` subcommand.

## Install (local)

> **Local use needs an NVIDIA GPU.** Without one, use the **Colab notebook** or
> **Modal** runner above instead.

```bash
git clone https://github.com/ghananlpcommunity/ghana-speech-datagen.git
cd ghana-speech-datagen
python3 -m venv .venv && source .venv/bin/activate
sudo apt-get install -y ffmpeg          # system dependency
pip install -e .                        # gives you the `ghana-speech-datagen` command
```

## Quickstart — TTS (generate audio from text)

Synthesise speech from text using VoxCPM. Output is **LJSpeech** format
(`metadata.csv`).

> `--text` names the column in the HF dataset that holds the sentences
> to be turned into speech. For example, `--text text` means the dataset's
> `"text"` column contains the prompts. You can point at any dataset on the Hub
> — just tell it which column has the text.

```bash
# Preview 5 clips first (hear it before a big run)
ghana-speech-datagen tts --dataset ghananlpcommunity/your-text-dataset --text text --preview 5

# From an HF dataset → 5 h, into data/<name>
ghana-speech-datagen tts --dataset ghananlpcommunity/your-text-dataset --text text \
    --hours 5 --name twi-run

# From your own sentences (one per line) → 2 h
ghana-speech-datagen tts --text-file sentences.txt --hours 2

# Randomly sample 5000 texts from a large dataset
ghana-speech-datagen tts --dataset org/big-text --text text --max-samples 5000 \
    --hours 3

# Custom speaker reference audio (optional, up to 10 speakers)
# Point to a dir with <id>.wav + optional <id>.txt sidecars.
# Speakers cycle round-robin: row 0 → speaker0, row 1 → speaker1, ...
ghana-speech-datagen tts --dataset org/ds --text text --hours 5 \
    --speaker-dir /path/to/speakers/
# Without .txt sidecars, use --ref-text as shared prompt:
ghana-speech-datagen tts --dataset org/ds --text text --hours 5 \
    --speaker-dir /path/to/speakers/ --ref-text "my prompt"

# Resume: re-run the same command (finished rows are skipped)
```

## Quickstart — ASR (repackage existing audio — no GPU)

Validate, filter, and repackage an existing audio-text dataset into ASR format
(`metadata.jsonl`). No generation happens — just audio validation, copying, and
manifest writing. Requires ≥50 valid samples by default.

```bash
# From an HF dataset with audio + text columns
ghana-speech-datagen asr --dataset org/audio-text-ds --audio-column audio --text-column text

# From a local directory of audio files + metadata
ghana-speech-datagen asr --audio-dir my_clips/ --metadata transcripts.csv

# Apply duration filtering and sub-sampling
ghana-speech-datagen asr --dataset org/audio-text-ds --audio-column audio \
    --text-column text --min-duration 2.0 --max-duration 25.0 --max-samples 2000

# Push result to a new HF dataset repo
ghana-speech-datagen asr --dataset org/audio-text-ds --audio-column audio \
    --text-column text --push my-asr-repo
```

## Output — TTS (`tts` subcommand)

Everything lands in `data/<name>/` (override with `--out`):

```
data/twi-run/
  wavs/<id>.wav            mono, silence-trimmed, at --sample-rate (default 22050)
  manifest.jsonl           full info: id, file, text, gender, speaker, duration
  progress.json            resume state (re-run to continue)
  metadata.csv             ljspeech format:  id|text|text
```

## Output — ASR (`asr` subcommand)

```
data/my-repo/
  wavs/<id>.<ext>          copied from source, same format as original
  manifest.jsonl           full info: id, file, text, duration
  metadata.jsonl           asr manifest:  {"audio":"...","text":"..."}
```

## Options — TTS (`tts` subcommand)

| flag | meaning |
|------|---------|
| `--dataset ID` / `--text COL` | source: an HF dataset column |
| `--text-file PATH` | source: a .txt file, one sentence per line |
| `--config` / `--split` | dataset config / split (default split `train`) |
| `--hours H` | target hours of audio to generate |
| `--voices custom\|male\|female` | speaker selection (default `custom`) |
| `--male-pct N` | %% male in `custom` mode (deterministic per row) |
| `--max-chars N` | skip rows longer than this (default 400) |
| `--sample-rate HZ` | output WAV rate (default 22050) |
| `--precision fp32\|fp16\|bf16` | model precision (default fp32) — see Performance |
| `--instances N` | parallel model instances (default: auto by VRAM) |
| `--cfg` / `--steps` | CFG value / inference timesteps |
| `--max-samples N` | randomly pick at most this many texts (sub-sample) |
| `--min-duration` / `--max-duration` | skip clips shorter/longer than these (seconds) |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO [--private]` | upload the finished run to an HF dataset repo |
| `--token` | HF token — for gated datasets/models |
| `--preview N` | generate N preview clips and exit |
| `--list-datasets` | list datasets under the `ghananlpcommunity` org |
| `--speaker-dir DIR` | dir with `<id>.wav` + optional `<id>.txt` (up to 10 speakers) |
| `--ref-text TEXT` | fallback prompt text for speakers missing a `.txt` sidecar |

Resuming is automatic: point `--name`/`--out` at an existing run folder (or just
re-run the same command) and it reads `progress.json` and skips finished rows.

## Options — ASR (`asr` subcommand)

| flag | meaning |
|------|---------|
| `--dataset ID` | source: an HF dataset with audio+text columns |
| `--audio-column COL` | column with audio (default `audio`) |
| `--text-column COL` | column with transcripts (default `text`) |
| `--config` / `--split` | dataset config / split (default split `train`) |
| `--audio-dir DIR` | local dir with audio files (use with `--metadata`) |
| `--metadata PATH` | CSV/JSONL mapping audio filenames to transcripts |
| `--min-duration` / `--max-duration` | drop clips outside this range (seconds) |
| `--max-samples N` | randomly pick at most this many rows from source |
| `--min-samples N` | minimum valid samples required (default 50) |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO [--private]` | upload the finished run to an HF dataset repo |
| `--token` | HF token — for pushing / gated datasets |

## Performance & GPU

- **Parallel instances.** Several model copies pull rows off a shared queue
  (~4.5 GB VRAM each in fp32). The number of instances is auto-detected from
  your GPU's VRAM. Override with `--instances N`.
- **Precision** (`--precision`):
  - `fp32` — default, safest, highest quality.
  - `fp16` — ~half the VRAM (so ~2× the instances) and faster on most GPUs, **but
    may degrade quality or NaN** on TTS models; preview before committing.
  - `bf16` — ~half the VRAM, more numerically stable than fp16, **but needs an
    Ampere+ GPU (A100, L4, H100, H200, etc.)**.
- **Sample rate.** The model synthesises at **16 kHz**; output is resampled to
  `--sample-rate` so files match your framework, but true bandwidth stays ~8 kHz
  (upsampling doesn't add detail).

## Use as a library

```python
from ghana_speech_datagen import generate, export_formats

summary = generate(out_dir="data/run", dataset="org/ds", text_column="text",
                   target_hours=5, voices="custom", male_pct=50,
                   max_samples=10000)
export_formats("data/run", ["ljspeech"])
# {'rows': ..., 'hours': ..., 'errors': ..., 'duration_dropped': ...}
```

## Tests

```bash
pip install pytest
pytest tests/
```

## Project layout

```
ghana_speech_datagen/
  cli.py             the `ghana-speech-datagen` command
  generator.py       voice-clone, silence-trim, parallel run, resume, format export
  speakers/          built-in male/female reference wav + text
examples/
  ghana_speech_datagen.ipynb   Colab (GPU) runner
  modal_run.py                Modal (serverless GPU) runner
tests/
```

## License

CC-BY-4.0

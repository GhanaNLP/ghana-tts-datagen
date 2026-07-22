# Ghana Speech Datagen

Generate synthetic speech training data against a **vLLM-Omni VoxCPM2 TTS
server**. The tool itself is a lightweight HTTP client (no PyTorch, no local
model) — deploy the [`ghananlpcommunity/VoxCPM2-Ghana`](https://huggingface.co/ghananlpcommunity/VoxCPM2-Ghana)
model once on a GPU (see [`deploy/`](deploy/README.md)), then generate as many
datasets as you want against its API, voice-cloning from reference audio.

**You don't have to bring your own text.** Just pass `--lang` and the tool pulls
default text (and reference voices) for that language automatically:

```bash
ghana-speech-datagen tts --lang ewe --hours 5   # → TTS dataset (LJSpeech)
ghana-speech-datagen asr --lang ewe --hours 5   # → ASR dataset (JSONL manifest)
```

`--lang` selects the built-in text and default reference pool for that language,
**and** prepends the model's `<|lang:CODE|>` tag to each line at synthesis time —
exactly how VoxCPM2-Ghana was trained (the tag is learned as plain text and is
needed for correct per-language pronunciation). The manifests still store the
clean transcript. You can still bring your own text (`--dataset`/`--text-file`)
and reference audio when you want to.

## Two modes

Both modes synthesise speech from text; they differ in the **reference voices**
they use and the **output format** they write — each ready for its use case.

| | `tts` | `asr` |
|---|---|---|
| **Voices** | a small speaker set (the packaged male/female voices by default, or your own) | a large, diverse pool of reference audio (min. `--min-samples`, default 50) |
| **Best for** | building a TTS voice/dataset with consistent speakers | building ASR training data with many speakers for robustness |
| **Reference** | `--voices`, `--speaker-dir`, `--speaker` | `--ref-dataset`, `--ref-audio-dir`, or in-language default pool |
| **Output** | LJSpeech: `wavs/` + `metadata.csv` (`id\|text\|text`) | `wavs/` + `metadata.jsonl` (`{"audio","text"}`) |
| **Default rate** | 24000 Hz | 24000 Hz |

Every clip is also recorded in `manifest.jsonl` (full record, including
`speaker` for `tts`).

> The heavy lifting happens on the **TTS server** (a GPU box running vLLM-Omni).
> The datagen client is I/O-bound and can even run from your laptop against a
> remote server. vLLM-Omni batches concurrent requests, so throughput scales far
> beyond one-clip-at-a-time.

## Supported languages

The `VoxCPM2-Ghana` model supports **40+ Ghanaian languages** (plus English). Every
language ships with a built-in default text source, so `--lang <code>` is all you
need. List them with:

```bash
ghana-speech-datagen asr --list-langs
```

Codes are e.g. `ewe`, `fat`, `dag`, `twi-asante`, `twi-akuapem`, `en`. `--lang`
also accepts a full config name (`Ewe_ewe`) or display name (`Asante Twi`).

### Adding more text sources

Default text comes from [`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech).
To give a language extra text, add one line to `_EXTRA_SOURCES` in
[`ghana_speech_datagen/text_sources.py`](ghana_speech_datagen/text_sources.py).
Twi, for example, also draws from a 500-hour health corpus:

```python
_EXTRA_SOURCES = {
    "twi-asante": [
        TextSource("ghananlpcommunity/twi-health-asr-gemini-500hrs",
                   text_column="transcription"),
    ],
    # "ewe": [TextSource("your-org/your-ewe-text", text_column="text")],
}
```

## Setup

There are two pieces: the **TTS server** (on a GPU) and the **datagen client**
(anywhere that can reach the server — even your laptop).

### 1. Deploy the TTS server

Follow [`deploy/README.md`](deploy/README.md) — in short, on a GPU box:

```bash
# install vLLM-Omni (once), then:
cd deploy
API_KEY=my-secret bash serve.sh          # pulls VoxCPM2-Ghana from HF, serves :8000
```

### 2. Install the datagen client

```bash
git clone https://github.com/GhanaNLP/ghana-speech-datagen.git
cd ghana-speech-datagen
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Point it at your server (per-command flags also work):

```bash
export TTS_SERVER_URL=http://your-gpu-host:8000
export TTS_API_KEY=my-secret     # only if the server was started with API_KEY
```

## Quickstart — TTS

Synthesise a TTS dataset voiced by a small, consistent speaker set. By default it
uses the packaged male/female voices — no reference audio needed.

```bash
# Simplest: default text for a language, packaged male + female voices
ghana-speech-datagen tts --lang ewe --hours 5

# One voice only
ghana-speech-datagen tts --lang ewe --voices female --hours 5

# Your own speakers: a dir of NAME.wav + NAME.txt (prompt) pairs
ghana-speech-datagen tts --lang ewe --speaker-dir my_voices/ --hours 5

# A single custom speaker
ghana-speech-datagen tts --lang ewe \
    --speaker ref.wav --speaker-text "the reference transcript" --hours 2

# Your own text file, tagged with a language for correct pronunciation
ghana-speech-datagen tts --text-file sentences.txt --lang ewe --hours 2
```

Output is **LJSpeech** format (`wavs/` + `metadata.csv`), ready for most TTS
trainers (Coqui TTS, VITS, Tacotron, …).

## Quickstart — ASR

The model speaks each text in the voice of a randomly-selected reference clip
from a large pool. Both the text and the reference audio have sensible defaults
per language.

```bash
# Simplest: default text + in-language reference voices for a language
ghana-speech-datagen asr --lang ewe --hours 5

# Default text for Twi (ghana-speech + health corpus), your own reference voices
ghana-speech-datagen asr --lang twi-asante \
    --ref-dataset org/ref-audio-ds --ref-text-column text --hours 5

# List every supported language and its text sources
ghana-speech-datagen asr --list-langs
```

You can also bring your own text and/or reference audio:

```bash
# Texts from HF dataset, ref audio from another HF dataset
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --hours 5

# Your own text file, but tag it with a language so pronunciation is correct
ghana-speech-datagen asr --text-file sentences.txt --lang ewe \
    --ref-dataset org/ref-audio-ds --hours 2

# Texts from a .txt file, ref audio from local dir + metadata
ghana-speech-datagen asr --text-file sentences.txt \
    --ref-audio-dir my_refs/ --ref-metadata refs.csv \
    --hours 2

# Sub-sample texts, point at a remote TTS server
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --max-samples 2000 --server-url http://gpu-host:8000

# Send it to a specific HF dataset repo (instead of the auto-named one)
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --hours 10 --push my-asr-repo
```

## Uploading to Hugging Face

**Both modes auto-push to the Hub by default**, incrementally, as clips are
generated — so a long run keeps a live copy on HF even if it's interrupted.

- The repo is auto-named `you/ghana-speech-synth-<name>`; override it with `--push REPO_ID`.
- `--save-every N` controls how often the partial dataset is flushed and pushed (default 200 clips).
- `--private` makes the repo private.
- **`--no-push` disables uploading** — generate locally only (no HF token needed).

```bash
# Local only, nothing uploaded
ghana-speech-datagen tts --lang ewe --hours 5 --no-push

# Push to a private repo, uploading every 500 clips
ghana-speech-datagen asr --lang ewe --hours 20 --private --save-every 500
```

## Output

```
data/<name>/
  wavs/<id>.wav            mono 16-bit PCM, at --sample-rate (default 24000)
  manifest.jsonl           full record per clip: id, file, text, duration (+ speaker for tts)

  # tts writes (LJSpeech, the standard TTS layout):
  metadata.csv             id|text|normalized_text

  # asr writes:
  metadata.jsonl           {"audio":"wavs/...","text":"..."}
```

The transcript in the manifests is the **clean spoken text**.

## Options

| flag | meaning |
|------|---------|
| `--lang CODE` | use built-in default text for a language, default the reference pool to in-language audio, and prepend the model's `<|lang:CODE|>` training tag at synthesis |
| `--list-langs` | list supported languages and their default text sources |
| `--dataset ID` / `--text COL` | source: an HF dataset with text to synthesise |
| `--text-file PATH` | source: a .txt file with text to synthesise |
| `--config` / `--split` | dataset config / split (default `train`) |
| `--ref-dataset ID` | HF dataset with reference audio+transcript columns |
| `--audio-column COL` | column with reference audio (default `audio`) |
| `--ref-text-column COL` | column with reference transcripts (default `text`) |
| `--ref-config` / `--ref-split` | ref dataset config / split |
| `--ref-audio-dir DIR` | local dir with ref audio (use with `--ref-metadata`) |
| `--ref-metadata PATH` | CSV/JSONL mapping ref audio filenames to transcripts |
| `--hours H` | target hours of audio to generate |
| `--min-duration` / `--max-duration` | drop generated clips outside this range (seconds) |
| `--max-samples N` | randomly pick at most this many texts |
| `--min-samples N` | minimum valid samples required (default 50) |
| `--sample-rate HZ` | output WAV rate (default 24000) |
| `--cfg` | CFG value passed to the server (default 2.0) |
| `--server-url URL` | TTS server base URL (env `TTS_SERVER_URL`; default `http://127.0.0.1:8000`) |
| `--api-key KEY` | API key for the TTS server, if required (env `TTS_API_KEY`) |
| `--model NAME` | served model name on the TTS server (default `voxcpm2`) |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO` | override the auto-named HF dataset repo to push to |
| `--no-push` | disable the default auto-push; generate locally only |
| `--save-every N` | flush + push every N clips as they're generated (default 200) |
| `--private` | make the pushed repo private instead |
| `--token` | HF token — for gated datasets/models and pushing |

## Use as a library

```python
from ghana_speech_datagen.generator import generate_asr

pairs = [
    ("Hello world", "/refs/spk1.wav", "Prompt text one"),
    ("How are you", "/refs/spk2.wav", "Prompt text two"),
]

summary = generate_asr(
    out_dir="data/my-run",
    pairs=pairs,
    target_seconds=7200,
    sample_rate=24000,
    server_url="http://gpu-host:8000",
    api_key="my-secret",           # if the server requires one
    on_clip=lambda dur: print(f"Generated {dur:.1f}s"),
)
# {'rows': ..., 'hours': ..., 'skipped': ..., 'duration_dropped': ...}
```

For direct control of the TTS client:

```python
from ghana_speech_datagen.tts_client import VoxCPM2Client

with VoxCPM2Client(base_url="http://gpu-host:8000", api_key="my-secret") as tts:
    tts.wait_until_ready()
    tts.register_voice("spk1", "/refs/spk1.wav", "Prompt text")
    wav_bytes = tts.synthesize("spk1", "Text to speak", response_format="wav")
```

## Performance

- **The TTS server does the work.** Throughput comes from vLLM-Omni batching
  concurrent requests on the GPU — see the tuning knobs in
  [`deploy/voxcpm2.yaml`](deploy/voxcpm2.yaml) (KV-cache budget, batched CFM /
  VAE decode). The datagen client is I/O-bound.
- **Sample rate.** VoxCPM2 synthesises at a high rate and the output is
  resampled to `--sample-rate` (default 24 kHz). The client reads the true rate
  from each returned WAV rather than assuming one.
- **Remote-friendly.** Because the client only speaks HTTP, you can run long
  generation jobs from a small machine against a shared GPU server.

## Tests

```bash
pip install pytest
pytest tests/
```

## Project layout

```
ghana_speech_datagen/
  cli.py             the `ghana-speech-datagen` command
  generator.py       generation loop (generate_asr / generate_tts)
  tts_client.py      VoxCPM2Client — HTTP client for the vLLM-Omni TTS server
  speakers/          built-in male/female reference wav + text
deploy/
  serve.sh           launch the VoxCPM2 TTS server with vLLM-Omni
  voxcpm2.yaml       tuned vLLM-Omni deploy config
  README.md          how to deploy the server
tests/
```

## License

CC-BY-4.0

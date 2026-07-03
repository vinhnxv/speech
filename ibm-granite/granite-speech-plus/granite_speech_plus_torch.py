"""Granite Speech 4.1 2B Plus experiment script.

Supports four modes:
  asr        — plain transcript (no punctuation/capitalization)
  saa        — speaker-attributed ASR with [Speaker N]: tags
  timestamps — word-level end times via [T:N] centisecond tags (mod-1000 rollover)
  combined   — runs SAA + timestamps separately, aligns by text matching

Outputs structured JSON with all extractable metadata to stdout (compact) and
writes a pretty-printed .json file alongside the audio file.

Usage:
  python granite_speech_plus_torch.py [audio_path] [mode] [keywords...]
  python granite_speech_plus_torch.py voices/ENG_UK_M_DaveB.mp3 asr
  python granite_speech_plus_torch.py voices/ENG_UK_M_DaveB.mp3 saa
  python granite_speech_plus_torch.py voices/ENG_UK_M_DaveB.mp3 timestamps
  python granite_speech_plus_torch.py voices/ENG_UK_M_DaveB.mp3 combined
  python granite_speech_plus_torch.py voices/ENG_UK_M_DaveB.mp3 asr IBM Granite

Requires transformers>=5.8 (or a source install from
https://github.com/huggingface/transformers).
"""

import json
import os
import sys
import time

import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Ensure the script's directory is on the path for parsing.py import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parsing import parse_asr, parse_saa, parse_timestamps, parse_combined

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "ibm-granite/granite-speech-4.1-2b-plus"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bfloat16: same memory as float16, larger exponent range → fewer overflows with LLMs
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32

SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)

# Mode-specific prompts (from the model card)
PROMPTS = {
    "asr": "<|audio|> can you transcribe the speech into a written format?",
    "saa": (
        "<|audio|> Speaker attribution: Transcribe and denote who is speaking "
        "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
    ),
    "timestamps": (
        "<|audio|> Timestamps: Transcribe the speech. After each word, add a "
        "timestamp tag showing the end time in centiseconds, "
        "e.g. hello [T:45] world [T:82]"
    ),
}

# Timestamps produce far more tokens — one [T:N] tag per word
MAX_NEW_TOKENS = {
    "asr": 2000,
    "saa": 2000,
    "timestamps": 10000,
}

VALID_MODES = ("asr", "saa", "timestamps", "combined")


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return processor, model


def transcribe(processor, model, audio_tensor, prompt, max_new_tokens=2000, keywords=None):
    """Run a single inference pass and return the raw decoded text."""
    if keywords:
        prompt = f"{prompt} Keywords: {', '.join(keywords)}"

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text, audio=audio_tensor, return_tensors="pt")
    inputs = {
        k: v.to(DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE)
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    result = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path):
    """Load audio file, convert to mono, resample to 16 kHz. Returns (tensor, sr, duration)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    if sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
        sr = 16000
    audio_tensor = audio_tensor.squeeze(0)  # (samples,)

    duration = audio_tensor.shape[0] / sr
    return audio_tensor, sr, duration


# ---------------------------------------------------------------------------
# JSON assembly
# ---------------------------------------------------------------------------

def build_json(mode, audio_path, sr, duration, raw_outputs, parsed, keywords, proc_time):
    """Construct the output JSON dict with all metadata fields."""
    result = {
        "audio": {
            "path": audio_path,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "channels": 1,  # always mono after our conversion
        },
        "session": {
            "model": MODEL_ID,
            "mode": mode,
            "processing_time": round(proc_time, 2),
        },
    }

    if keywords:
        result["session"]["keywords"] = keywords

    # Merge mode-specific parsed fields
    result.update(parsed)

    # Include raw model output for debugging/transparency
    result["raw_output"] = raw_outputs

    return result


def write_json(result, audio_path):
    """Print compact JSON to stdout and write pretty .json file alongside audio."""
    compact = json.dumps(result, ensure_ascii=False)
    print(compact)

    json_path = os.path.splitext(audio_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[json] written to {json_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "voices/ENG_UK_M_DaveB.mp3"
    mode = sys.argv[2] if len(sys.argv) > 2 else "asr"
    keywords = sys.argv[3:] if len(sys.argv) > 3 else None

    if mode not in VALID_MODES:
        print(
            f"Error: invalid mode '{mode}'. Valid modes: {', '.join(VALID_MODES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(audio_path):
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model on {DEVICE}...", file=sys.stderr)
    processor, model = load_model()

    print(f"Loading audio: {audio_path}", file=sys.stderr)
    audio_tensor, sr, duration = load_audio(audio_path)
    print(f"Audio: {duration:.1f}s @ {sr} Hz", file=sys.stderr)

    start_time = time.time()

    raw_outputs = {}
    parsed = {}

    if mode == "combined":
        # Run SAA and timestamps separately, then align
        print("Running SAA inference...", file=sys.stderr)
        saa_raw = transcribe(
            processor, model, audio_tensor, PROMPTS["saa"],
            max_new_tokens=MAX_NEW_TOKENS["saa"], keywords=keywords,
        )
        print("Running timestamps inference...", file=sys.stderr)
        ts_raw = transcribe(
            processor, model, audio_tensor, PROMPTS["timestamps"],
            max_new_tokens=MAX_NEW_TOKENS["timestamps"], keywords=keywords,
        )

        saa_parsed = parse_saa(saa_raw)
        ts_parsed = parse_timestamps(ts_raw, duration=duration)
        parsed = parse_combined(saa_parsed, ts_parsed)
        raw_outputs = {"saa": saa_raw, "timestamps": ts_raw}

    elif mode == "asr":
        print("Running ASR inference...", file=sys.stderr)
        raw = transcribe(
            processor, model, audio_tensor, PROMPTS["asr"],
            max_new_tokens=MAX_NEW_TOKENS["asr"], keywords=keywords,
        )
        parsed = parse_asr(raw)
        raw_outputs = raw

    elif mode == "saa":
        print("Running SAA inference...", file=sys.stderr)
        raw = transcribe(
            processor, model, audio_tensor, PROMPTS["saa"],
            max_new_tokens=MAX_NEW_TOKENS["saa"], keywords=keywords,
        )
        parsed = parse_saa(raw)
        raw_outputs = raw

    elif mode == "timestamps":
        print("Running timestamps inference...", file=sys.stderr)
        raw = transcribe(
            processor, model, audio_tensor, PROMPTS["timestamps"],
            max_new_tokens=MAX_NEW_TOKENS["timestamps"], keywords=keywords,
        )
        parsed = parse_timestamps(raw, duration=duration)
        raw_outputs = raw

    proc_time = time.time() - start_time

    result = build_json(
        mode, audio_path, sr, duration, raw_outputs, parsed, keywords, proc_time,
    )
    write_json(result, audio_path)


if __name__ == "__main__":
    main()
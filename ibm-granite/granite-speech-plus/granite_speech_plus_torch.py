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
import re
import sys
import time

import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
# Output parsers
# ---------------------------------------------------------------------------

def parse_asr(text):
    """Parse plain ASR output — just a transcript string."""
    return {"transcript": text.strip()}


def parse_saa(text):
    """Parse speaker-attributed ASR output into speaker turns.

    Output format: [Speaker 1]: text [Speaker 2]: text ...
    """
    # Split on speaker tags, keeping the tags
    parts = re.split(r"(\[Speaker \d+\]:)", text)
    speakers = []
    speaker_ids = set()
    current_speaker = None
    current_text = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"\[Speaker (\d+)\]:", part)
        if match:
            if current_speaker is not None:
                speakers.append({
                    "speaker_id": current_speaker,
                    "text": current_text.strip(),
                    "turn_order": len(speakers),
                })
            current_speaker = int(match.group(1))
            speaker_ids.add(current_speaker)
            current_text = ""
        else:
            current_text = (current_text + " " + part).strip() if current_text else part

    if current_speaker is not None:
        speakers.append({
            "speaker_id": current_speaker,
            "text": current_text.strip(),
            "turn_order": len(speakers),
        })

    return {
        "speakers": speakers,
        "speaker_count": len(speaker_ids),
        "raw": text.strip(),
    }


def parse_timestamps(text):
    """Parse word-level timestamp output into per-word records.

    Timestamps are in centiseconds mod 1000 (10s rollover).
    N = round(t * 100) mod 1000; recover with t = N/100 + 10*R.
    Silence is transcribed as `_`.
    """
    # re.split with capture group gives: [word0, tag0, word1, tag1, ..., trailing_word]
    ts_parts = re.split(r"\[T:(\d+)\]", text)
    words = []
    last_end = 0.0
    offset = 0.0

    # ts_parts: even indices = word text, odd indices = timestamp numbers
    word_parts = ts_parts[::2]
    tag_parts = ts_parts[1::2]

    for word_text, ts in zip(word_parts, tag_parts):
        word_text = word_text.strip()
        if not word_text:
            continue

        raw_time = float(ts) / 100.0
        while raw_time + offset < last_end:
            offset += 10.0
        abs_time = raw_time + offset
        last_end = abs_time

        # Words can include silence markers `_`
        is_silence = word_text == "_"

        words.append({
            "text": word_text,
            "end_time": round(abs_time, 2),
            "is_silence": is_silence,
        })

    return {"words": words}


def parse_combined(saa_result, ts_result):
    """Align speaker turns with word timings by text matching.

    Runs SAA and timestamps separately, then aligns by normalizing word sequences.
    """
    saa_turns = saa_result["speakers"]
    ts_words = ts_result["words"]

    # Flatten SAA turns into a normalized word list (lowercase, no tags)
    saa_words_flat = []
    saa_word_to_turn = []  # maps flat index → turn index
    for turn_idx, turn in enumerate(saa_turns):
        for w in turn["text"].split():
            saa_words_flat.append(w.lower().strip(".,!?;:\"'()[]"))
            saa_word_to_turn.append(turn["speaker_id"])

    # Normalize timestamp words (strip silence markers for alignment count)
    ts_words_norm = []
    for w in ts_words:
        ts_words_norm.append(w["text"].lower().strip(".,!?;:\"'()[]"))

    # Remove silence words from ts for alignment (they don't correspond to SAA words)
    ts_non_silence = [(i, w) for i, w in enumerate(ts_words) if not w["is_silence"]]
    ts_norm_non_silence = [ts_words_norm[i] for i, _ in ts_non_silence]

    alignment_method = None
    combined_words = []

    if len(saa_words_flat) == len(ts_norm_non_silence):
        # Exact word count match — map non-silence ts words to SAA words by index,
        # and assign silence words the speaker of the preceding non-silence word.
        alignment_method = "exact_match"
        saa_idx = 0
        last_speaker = None
        for ts_w in ts_words:
            if ts_w["is_silence"]:
                speaker_id = last_speaker
            else:
                speaker_id = saa_word_to_turn[saa_idx]
                saa_idx += 1
                last_speaker = speaker_id
            combined_words.append({
                "text": ts_w["text"],
                "end_time": ts_w["end_time"],
                "is_silence": ts_w["is_silence"],
                "speaker_id": speaker_id,
            })
    else:
        # Proportional allocation: distribute ts words across saa turns by turn length ratio
        alignment_method = "proportional_fallback"
        total_saa_words = len(saa_words_flat)
        total_ts_words = len(ts_norm_non_silence)

        if total_saa_words == 0 or total_ts_words == 0:
            # Can't align — return unaligned per-mode output
            alignment_method = "unaligned_fallback"
            for ts_w in ts_words:
                combined_words.append({
                    "text": ts_w["text"],
                    "end_time": ts_w["end_time"],
                    "is_silence": ts_w["is_silence"],
                    "speaker_id": None,
                })
        else:
            # Allocate ts words to turns proportionally
            ts_idx = 0
            for turn_idx, turn in enumerate(saa_turns):
                turn_word_count = len(turn["text"].split())
                # Proportional allocation
                alloc = round(turn_word_count * total_ts_words / total_saa_words)
                alloc = max(1, alloc)  # at least 1 word per turn if possible

                allocated = 0
                # Include any silence markers that fall within this turn's ts range
                while ts_idx < len(ts_words) and allocated < alloc:
                    ts_w = ts_words[ts_idx]
                    combined_words.append({
                        "text": ts_w["text"],
                        "end_time": ts_w["end_time"],
                        "is_silence": ts_w["is_silence"],
                        "speaker_id": turn["speaker_id"],
                    })
                    ts_idx += 1
                    if not ts_w["is_silence"]:
                        allocated += 1

            # Assign remaining ts words to last speaker
            while ts_idx < len(ts_words):
                ts_w = ts_words[ts_idx]
                combined_words.append({
                    "text": ts_w["text"],
                    "end_time": ts_w["end_time"],
                    "is_silence": ts_w["is_silence"],
                    "speaker_id": saa_turns[-1]["speaker_id"] if saa_turns else None,
                })
                ts_idx += 1

    return {
        "words": combined_words,
        "alignment_method": alignment_method,
        "saa_word_count": len(saa_words_flat),
        "ts_word_count": len(ts_norm_non_silence),
    }


# ---------------------------------------------------------------------------
# JSON assembly
# ---------------------------------------------------------------------------

def build_json(mode, audio_path, sr, duration, raw_outputs, parsed, keywords, proc_time):
    """Construct the output JSON dict with all metadata fields."""
    # Count channels from the loaded audio (always 1 after our mono conversion)
    channels = 1

    result = {
        "audio": {
            "path": audio_path,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "channels": channels,
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
        ts_parsed = parse_timestamps(ts_raw)
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
        parsed = parse_timestamps(raw)
        raw_outputs = raw

    proc_time = time.time() - start_time

    result = build_json(
        mode, audio_path, sr, duration, raw_outputs, parsed, keywords, proc_time,
    )
    write_json(result, audio_path)


if __name__ == "__main__":
    main()
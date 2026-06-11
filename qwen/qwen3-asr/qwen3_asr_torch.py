import os
import sys
import torch
from qwen_asr import Qwen3ASRModel

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bfloat16: same memory as float16, larger exponent range → fewer overflows with LLMs
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32


def load_model():
    return Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
        max_new_tokens=256,
        forced_aligner=ALIGNER_ID,
        forced_aligner_kwargs=dict(dtype=DTYPE, device_map=DEVICE),
    )


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_srt(words, max_words: int = 10, max_gap: float = 1.0) -> str:
    """Group word timestamps into subtitle cues: split on long pauses or cue length."""
    cues, current = [], []
    for w in words:
        if current and (
            len(current) >= max_words or w.start_time - current[-1].end_time > max_gap
        ):
            cues.append(current)
            current = []
        current.append(w)
    if current:
        cues.append(current)

    blocks = []
    for i, cue in enumerate(cues, 1):
        text = " ".join(w.text for w in cue)
        blocks.append(
            f"{i}\n{srt_time(cue[0].start_time)} --> {srt_time(cue[-1].end_time)}\n{text}\n"
        )
    return "\n".join(blocks)


def transcribe(audio_path: str, model, language: str | None = None):
    results = model.transcribe(
        audio=audio_path,
        language=language,  # None = auto-detect; otherwise e.g. "English", "Vietnamese"
        return_time_stamps=True,
    )
    return results[0]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "voices/ENG_UK_M_DaveB.mp3"
    language = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Loading model on {DEVICE}...")
    mdl = load_model()
    print("Transcribing...")
    result = transcribe(path, mdl, language)
    print(f"[language] {result.language}")
    print(f"[text] {result.text}")

    if result.time_stamps:
        srt = words_to_srt(result.time_stamps)
        srt_path = os.path.splitext(path)[0] + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt)
        print(f"[srt] written to {srt_path}")
        print(srt)

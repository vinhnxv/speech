"""On-device Vietnamese TTS via VieNeu-TTS v3 Turbo (torch-free ONNX path).

Runs entirely locally. The first call downloads the ~0.1B model from Hugging Face
(pnnbao-ump/VieNeu-TTS-v3-Turbo) and caches it. Output is 48 kHz. Supports the 10
built-in preset voices and instant voice cloning from a 3-5s reference clip.
Inline emotion / non-verbal cues are supported in the text: [cười] [thở dài]
[hắng giọng].

Install (separate venv — see requirements.txt):
    pyenv virtualenv 3.14.5 vieneu && pyenv shell vieneu && pip install vieneu

Usage:
    # default voice (Ngọc Lan), built-in sample text -> voices/vieneu_sample.wav
    python pnnbao-ump/vieneu-tts/vieneu_tts_onnx.py

    # list the built-in preset voices and exit
    python pnnbao-ump/vieneu-tts/vieneu_tts_onnx.py --list

    # preset voice + custom text
    python pnnbao-ump/vieneu-tts/vieneu_tts_onnx.py "Xin chào mọi người" --voice "Xuân Vĩnh" -o out.wav

    # text from a file
    python pnnbao-ump/vieneu-tts/vieneu_tts_onnx.py voices/script.txt --voice "Ngọc Linh"

    # instant voice cloning from a reference clip (3-5s)
    python pnnbao-ump/vieneu-tts/vieneu_tts_onnx.py "Đây là giọng nhân bản" --clone ref.wav
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOICES_DIR = ROOT / "voices"

DEFAULT_TEXT = (
    "Xin chào, đây là giọng nói tiếng Việt được tạo bởi mô hình VieNeu-TTS, "
    "chạy hoàn toàn trên máy."
)
DEFAULT_OUTPUT = VOICES_DIR / "vieneu_sample.wav"


def resolve_text(value: str) -> str:
    """Return literal text, or the contents of `value` if it is an existing file."""
    path = Path(value)
    if path.is_file():
        return path.read_text().strip()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-device Vietnamese TTS (VieNeu-TTS v3 Turbo, ONNX)."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=DEFAULT_TEXT,
        help="Text to synthesize, or a path to a .txt file. Defaults to a sample.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Preset voice name (e.g. 'Xuân Vĩnh'). Omit for the default (Ngọc Lan).",
    )
    parser.add_argument(
        "--clone",
        metavar="REF_AUDIO",
        default=None,
        help="Reference clip (3-5s) for instant voice cloning. Overrides --voice.",
    )
    parser.add_argument(
        "--emotion",
        default="natural",
        help="Emotion preset for built-in voices (default: natural).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8; ~0.8 recommended for stability).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output WAV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the built-in preset voices and exit.",
    )
    args = parser.parse_args()

    from vieneu import Vieneu  # heavy import; defer until after arg parsing

    t0 = time.time()
    tts = Vieneu()  # v3 Turbo, torch-free ONNX on CPU
    print(f"Model ready in {time.time() - t0:.1f}s", file=sys.stderr)

    if args.list:
        for label, voice_id in tts.list_preset_voices():
            print(f"- {label} ({voice_id})")
        return

    text = resolve_text(args.text)
    preview = text if len(text) <= 80 else text[:77] + "..."
    if args.clone:
        print(f"Cloning voice from {args.clone}: {preview}", file=sys.stderr)
    else:
        print(f"Voice {args.voice or '(default)'}: {preview}", file=sys.stderr)

    t0 = time.time()
    audio = tts.infer(
        text,
        ref_audio=args.clone,
        voice=args.voice,
        emotion=args.emotion,
        temperature=args.temperature,
    )
    dur = len(audio) / tts.sample_rate
    print(
        f"Synthesized {dur:.1f}s of audio in {time.time() - t0:.1f}s "
        f"(RTF {(time.time() - t0) / dur:.2f})",
        file=sys.stderr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tts.save(audio, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

import os
import sys
import tempfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio
import soundfile as sf
import nemo.collections.asr as nemo_asr

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
    model = model.to(DEVICE)
    model.eval()
    return model


def to_wav_16k_mono(audio_path: str) -> str:
    """NeMo expects mono WAV; convert arbitrary input to a temp 16kHz mono wav."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    if sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_tensor.squeeze(0).numpy(), 16000)
    return tmp.name


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{seg['segment']}\n"
        )
    return "\n".join(blocks)


def transcribe(audio_path: str, model):
    wav_path = to_wav_16k_mono(audio_path)
    try:
        outputs = model.transcribe([wav_path], timestamps=True)
    finally:
        os.unlink(wav_path)
    return outputs[0]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "voices/ENG_UK_M_DaveB.mp3"
    print(f"Loading model on {DEVICE}...")
    mdl = load_model()
    print("Transcribing...")
    result = transcribe(path, mdl)
    print(f"[text] {result.text}")

    segments = result.timestamp.get("segment") if result.timestamp else None
    if segments:
        srt = segments_to_srt(segments)
        srt_path = os.path.splitext(path)[0] + ".nemotron.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt)
        print(f"[srt] written to {srt_path}")
        print(srt)

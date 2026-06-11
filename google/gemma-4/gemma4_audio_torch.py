import os
import sys
import tempfile

import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForMultimodalLM, AutoProcessor

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# 12B needs ~24GB in bf16 — too big for a 24GB Mac; E4B/E2B also support audio
MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-4-E4B-it")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bfloat16: same memory as float16, larger exponent range → fewer overflows with LLMs
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32
SAMPLE_RATE = 16000  # Gemma 4 audio encoder expects 16kHz mono
CHUNK_SECONDS = 30  # Gemma 4 caps audio input at 30s per clip


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return processor, model


def load_audio(audio_path: str):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # from_numpy avoids a data copy
    if sr != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, SAMPLE_RATE)
    return audio_tensor.squeeze(0).numpy()


def generate(audio_chunk, processor, model, prompt: str) -> str:
    # apply_chat_template loads audio from a path, so stage the chunk as a temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_chunk, SAMPLE_RATE)
        chunk_path = tmp.name
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": chunk_path},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE)
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

        result = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        return result
    finally:
        os.unlink(chunk_path)


def transcribe(audio_path: str, processor, model, language: str | None = None) -> str:
    if language:
        prompt = f"Transcribe this audio clip into text in {language}."
    else:
        prompt = "Transcribe this audio clip into text in its original language."

    audio = load_audio(audio_path)
    chunk_len = CHUNK_SECONDS * SAMPLE_RATE
    chunks = [audio[i : i + chunk_len] for i in range(0, len(audio), chunk_len)]

    texts = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"  chunk {i}/{len(chunks)} ({len(chunk) / SAMPLE_RATE:.1f}s)...")
        texts.append(generate(chunk, processor, model, prompt))
    return " ".join(texts)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "voices/ENG_UK_M_DaveB.mp3"
    language = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Loading model on {DEVICE}...")
    proc, mdl = load_model()
    print("Transcribing...")
    print(transcribe(path, proc, mdl, language))

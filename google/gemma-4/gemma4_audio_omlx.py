import base64
import io
import os
import sys

import librosa
import soundfile as sf
from openai import OpenAI

# oMLX OpenAI-compatible server (also works with LM Studio / llama-server)
HOST = os.environ.get("OMLX_HOST", "http://127.0.0.1:1234/v1")
MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-12B-it-8bit")
SAMPLE_RATE = 16000  # Gemma 4 audio encoder expects 16kHz mono
CHUNK_SECONDS = 30  # Gemma 4 caps audio input at 30s per clip


def chunk_to_wav_b64(chunk) -> str:
    buf = io.BytesIO()
    sf.write(buf, chunk, SAMPLE_RATE, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def transcribe(audio_path: str, client, language: str | None = None) -> str:
    if language:
        prompt = f"Transcribe this audio clip into text in {language}."
    else:
        prompt = "Transcribe this audio clip into text in its original language."

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_len = CHUNK_SECONDS * SAMPLE_RATE
    chunks = [audio[i : i + chunk_len] for i in range(0, len(audio), chunk_len)]

    texts = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"  chunk {i}/{len(chunks)} ({len(chunk) / SAMPLE_RATE:.1f}s)...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": chunk_to_wav_b64(chunk),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=512,
        )
        texts.append(response.choices[0].message.content.strip())
    return " ".join(texts)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "voices/ENG_UK_M_DaveB.mp3"
    language = sys.argv[2] if len(sys.argv) > 2 else None
    client = OpenAI(base_url=HOST, api_key=os.environ.get("OMLX_API_KEY", "omlx"))
    print(f"Transcribing via {HOST} ({MODEL})...")
    print(transcribe(path, client, language))

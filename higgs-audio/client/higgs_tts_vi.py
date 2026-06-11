"""Voice-cloned Vietnamese TTS via a remote Higgs Audio v3 (sglang-omni) server.

The server only accepts reference audio as a server-local path or an HTTP URL,
so this client serves the reference voice over a temporary HTTP server that the
TTS host fetches from (works across Tailscale).

Usage:
    python higgs-audio/client/higgs_tts_vi.py [text_file] [ref_audio] [ref_text_file] [output_wav]
"""

import functools
import os
import socket
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

HOST = os.environ.get("HIGGS_HOST", "http://localhost:8000")
MODEL = "bosonai/higgs-audio-v3-tts-4b"
TIMEOUT_SECONDS = 600

ROOT = Path(__file__).resolve().parents[2]
VOICES_DIR = ROOT / "voices"

DEFAULT_TEXT = VOICES_DIR / "ENG_UK_M_DaveB.vi.txt"
DEFAULT_REF_AUDIO = VOICES_DIR / "ENG_UK_M_DaveB.mp3"
DEFAULT_REF_TEXT = VOICES_DIR / "ENG_UK_M_DaveB.txt"
DEFAULT_OUTPUT = VOICES_DIR / "ENG_UK_M_DaveB.vi.wav"


def local_ip_towards(host_url: str) -> str:
    """Source IP this machine uses to reach the TTS host (Tailscale-aware)."""
    server_host = host_url.split("//", 1)[-1].split(":")[0].split("/")[0]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((server_host, 80))
        return s.getsockname()[0]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve_directory(directory: Path) -> tuple[ThreadingHTTPServer, int]:
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("0.0.0.0", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def synthesize(text: str, ref_audio: Path, ref_text: str, output: Path) -> None:
    server, port = serve_directory(ref_audio.parent)
    try:
        ref_url = f"http://{local_ip_towards(HOST)}:{port}/{ref_audio.name}"
        response = requests.post(
            f"{HOST}/v1/audio/speech",
            json={
                "model": MODEL,
                "input": text,
                "response_format": "wav",
                "references": [{"audio_path": ref_url, "text": ref_text}],
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    finally:
        server.shutdown()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    print(f"Saved {output} ({len(response.content) / 1024:.1f} KB)")


if __name__ == "__main__":
    args = sys.argv[1:]
    text_file = Path(args[0]) if len(args) > 0 else DEFAULT_TEXT
    ref_audio = Path(args[1]) if len(args) > 1 else DEFAULT_REF_AUDIO
    ref_text_file = Path(args[2]) if len(args) > 2 else DEFAULT_REF_TEXT
    output = Path(args[3]) if len(args) > 3 else DEFAULT_OUTPUT

    text = text_file.read_text().strip()
    preview = text if len(text) <= 80 else text[:77] + "..."
    print(f"Synthesizing with voice {ref_audio.name}: {preview}")
    synthesize(text, ref_audio, ref_text_file.read_text().strip(), output)

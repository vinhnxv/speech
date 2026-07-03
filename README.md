# Speech

## Setup

```bash
pyenv virtualenv 3.14.5 speech

pyenv shell speech

pip install -r requirements.txt
```

## Run

```bash
# IBM Granite Speech (en, fr, de, es, pt, ja)
python ibm-granite/granite-speech/granite_speech_torch.py [audio_path]

# IBM Granite Speech 4.1 2B Plus (en, fr, de, es, pt) — speaker-attributed ASR, word timestamps, combined mode; outputs JSON metadata
# requires transformers>=5.8 (may need: pip install git+https://github.com/huggingface/transformers)
python ibm-granite/granite-speech-plus/granite_speech_plus_torch.py [audio_path] [asr|saa|timestamps|combined] [keywords...]

# Cohere Transcribe (14 languages incl. vi) — text only, no timestamps
python cohere-labs/cohere-transcribe/cohere_transcribe_torch.py [audio_path] [language]

# Qwen3-ASR (30 languages incl. vi) — word timestamps via forced aligner, writes .srt
python qwen/qwen3-asr/qwen3_asr_torch.py [audio_path] [language]

# MOSS-Transcribe-preview-2B (en only) — tests enable_time_marker timestamp support (result: input-only, no output timestamps)
# requires separate venv (model built for transformers 4.57.x, incompatible with 5.x internals):
#   pyenv virtualenv 3.14.5 moss && pip install "transformers==4.57.6" librosa torch
python openmoss/moss-transcribe/moss_transcribe_torch.py [audio_path]

# NVIDIA Nemotron Speech Streaming (en only) — segment/word timestamps, writes .srt
# requires: pip install "nemo_toolkit[asr]"
python nvidia/nemotron-speech-streaming/nemotron_streaming_torch.py [audio_path]

# Gemma 4 audio understanding (transcription/translation, 30s max per chunk — auto-chunked)
# via oMLX OpenAI-compatible server (default http://127.0.0.1:1234/v1, model gemma-4-12B-it-8bit)
# override with OMLX_HOST / OMLX_API_KEY / GEMMA_MODEL
python google/gemma-4/gemma4_audio_omlx.py [audio_path] [language]

# local transformers variant: default google/gemma-4-E4B-it (better Vietnamese than the oMLX path)
# override with GEMMA_MODEL=google/gemma-4-12B-it (needs ~24GB+ free RAM)
python google/gemma-4/gemma4_audio_torch.py [audio_path] [language]

# Higgs Audio v3 TTS (sglang-omni server, default http://localhost:8000) — voice-cloned Vietnamese speech
# serves the reference voice over a temp HTTP server so the TTS host can fetch it
# set HIGGS_HOST to point at a remote server, e.g. HIGGS_HOST=http://<server-ip>:8000
python higgs-audio/client/higgs_tts_vi.py [text_file] [ref_audio] [ref_text_file] [output_wav]

# VieNeu-TTS v3 Turbo (on-device Vietnamese TTS, 48 kHz, torch-free ONNX on CPU)
# runs locally — first call downloads the ~0.1B model from HF and caches it
# needs its own venv (see requirements.txt): pyenv virtualenv 3.14.5 vieneu && pip install vieneu
python pnnbao-ump/vieneu-tts/vieneu_tts_onnx.py [text_or_txt_file] [--voice NAME | --clone ref.wav] [-o out.wav]
python pnnbao-ump/vieneu-tts/vieneu_tts_onnx.py --list   # list the 10 built-in preset voices
```

> ⚠️ Dependency conflict: cohere/gemma need `transformers>=5.4`, granite-speech-plus needs `>=5.8`,
> qwen-asr pins `==4.57.6`, nemo_toolkit wants `4.53.x`, moss-transcribe may need `==4.57.6` if API signatures differ
> from the model's build version — `requirements.txt` covers the `>=5.4` family;
> install qwen-asr, nemo_toolkit, and moss-transcribe in their own virtualenvs (see comments in requirements.txt).
> Granite Speech Plus may need `pip install git+https://github.com/huggingface/transformers` until PyPI reaches 5.8.
> VieNeu-TTS is torch-free but pulls its own stack (onnxruntime, gradio, sea-g2p),
> so install it in a separate `vieneu` virtualenv too.

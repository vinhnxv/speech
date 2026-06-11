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

# Cohere Transcribe (14 languages incl. vi) — text only, no timestamps
python cohere-labs/cohere-transcribe/cohere_transcribe_torch.py [audio_path] [language]

# Qwen3-ASR (30 languages incl. vi) — word timestamps via forced aligner, writes .srt
python qwen/qwen3-asr/qwen3_asr_torch.py [audio_path] [language]

# NVIDIA Nemotron Speech Streaming (en only) — segment/word timestamps, writes .srt
# requires: pip install "nemo_toolkit[asr]"
python nvidia/nemotron-speech-streaming/nemotron_streaming_torch.py [audio_path]
```

> ⚠️ Dependency conflict: cohere needs `transformers>=5.4`, qwen-asr pins `==4.57.6`,
> nemo_toolkit wants `4.53.x` — use a separate virtualenv per model family.

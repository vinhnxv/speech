import sys
import torch
import torchaudio
import soundfile as sf
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bfloat16: same memory as float16, larger exponent range → fewer overflows with LLMs
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return processor, model


def transcribe(audio_path: str, processor, model, language: str = "en") -> str:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # from_numpy avoids a data copy
    if sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
    audio_tensor = audio_tensor.squeeze(0)

    inputs = processor(
        audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt", language=language
    )
    inputs = inputs.to(DEVICE, dtype=DTYPE)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    result = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "voices/ENG_UK_M_DaveB.mp3"
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    print(f"Loading model on {DEVICE}...")
    proc, mdl = load_model()
    print("Transcribing...")
    print(transcribe(path, proc, mdl, language))

import os
import sys
import torch
import torchaudio
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "ibm-granite/granite-speech-4.1-2b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bfloat16: same memory as float16, larger exponent range → fewer overflows with LLMs
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return processor, model


def transcribe(
    audio_path: str,
    processor,
    model,
    prompt: str = "transcribe the speech with proper punctuation and capitalization.",
) -> str:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # from_numpy avoids a data copy
    if sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
    audio_tensor = audio_tensor.squeeze(0)

    messages = [{"role": "user", "content": f"<|audio|> {prompt}"}]
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

    result = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "voices/ENG_UK_M_DaveB.mp3"
    print(f"Loading model on {DEVICE}...")
    proc, mdl = load_model()
    print("Transcribing...")
    print(transcribe(path, proc, mdl))

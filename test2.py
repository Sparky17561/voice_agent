#!/usr/bin/env python3
"""
Baymax Offline Voice Agent (Optimized Fast Version)

Fixes included:
✔ No early-cut sentences
✔ No dtype warnings (using dtype=)
✔ Faster generation (streamlined settings)
✔ Safe prompt cleaner (no truncation)
✔ Real-time Kokoro chunked TTS
✔ Preloaded voices
✔ Male/female voices supported
"""

import os
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------
# Standard Imports
# ---------------------------------
import sys
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ---------------------------------
# Audio Configuration
# ---------------------------------
RECORD_SECONDS = 4
WHISPER_SR = 16000
KOKORO_SR = 24000

VOICE = "am_michael"    # Male voice (clean + natural)
# Available clean voices: af_bella, af_sarah, am_michael, am_james


# ---------------------------------
# MODEL SELECTION (SWITCH EASILY)
# ---------------------------------

# MODEL_NAME = "microsoft/phi-2"
# LOAD_4BIT = False

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LOAD_4BIT = False

# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# LOAD_4BIT = False

# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# LOAD_4BIT = True


# ---------------------------------
# Load Whisper (CPU)
# ---------------------------------
print("🧠 Loading Whisper (CPU)…")
stt_model = WhisperModel("small", device="cpu", compute_type="int8")
print("✔ Whisper ready.")


# ---------------------------------
# Load LLM (GPU if available)
# ---------------------------------
def load_llm():
    print(f"🤖 Loading LLM: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if LOAD_4BIT:
        print("→ Loading 4-bit quantized model (fast + light)…")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("→ Loading model with dtype…")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            dtype=dtype,
            trust_remote_code=True
        )

    print(f"✔ Loaded: {MODEL_NAME}")
    return tokenizer, model


tokenizer, model = load_llm()


# ---------------------------------
# Low-level Kokoro TTS (preloaded)
# ---------------------------------
from kokoro import KPipeline
print("🔊 Loading Kokoro TTS… (CPU-only)")
tts = KPipeline(lang_code="a")   # 'a' = English voices
print("✔ Kokoro ready.")


def speak(text):
    print("🔊 Speaking…")
    chunks = tts(text, voice=VOICE)

    # Real-time chunked playback
    for _, _, audio_chunk in chunks:
        sd.play(audio_chunk, samplerate=KOKORO_SR)
        sd.wait()


# ---------------------------------
# Record Microphone
# ---------------------------------
def record_audio():
    print(f"\n🎤 Recording {RECORD_SECONDS}s — speak now…")
    audio = sd.rec(
        int(RECORD_SECONDS * WHISPER_SR),
        samplerate=WHISPER_SR,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return (audio[:, 0] * 32767).astype(np.int16)


# ---------------------------------
# Speech-to-Text (Whisper)
# ---------------------------------
def transcribe(int16_audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, int16_audio, WHISPER_SR)
        wavpath = tmp.name

    segments, _ = stt_model.transcribe(wavpath)
    os.remove(wavpath)

    text = "".join([seg.text for seg in segments]).strip()
    print("📝 Transcript:", text)
    return text


# ---------------------------------
# Baymax Prompt (Updated)
# ---------------------------------
def build_prompt(user_text):
    return (
        "You are Baymax, a warm, friendly companion who talks in simple natural sentences.\n"
        "Rules:\n"
        "1. Never use formatting or symbols.\n"
        "2. Always speak like a calm human.\n"
        "3. Keep replies short (one or two sentences).\n"
        "4. Never give medical advice.\n"
        "5. No bullet points or structure.\n"
        "6. Always reply in plain speech.\n\n"
        f"User: {user_text}\n"
        "Assistant:"
    )


# ---------------------------------
# Generate Reply
# ---------------------------------
def generate_reply(user_text):
    prompt = build_prompt(user_text)

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **encoded,
        max_new_tokens=200,
        min_new_tokens=20,      # prevents early-cutting
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None       # disable early stop
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Safe prompt removal
    if "Assistant:" in decoded:
        decoded = decoded.split("Assistant:")[-1].strip()

    return decoded


# ---------------------------------
# Main Loop
# ---------------------------------
def main():
    print("\n=== Baymax Multi-LLM Voice Agent (Optimized) ===")
    print("Loaded Model:", MODEL_NAME, "\n")

    while True:
        input("Press Enter to speak…")
        audio = record_audio()
        text = transcribe(audio)
        reply = generate_reply(text)
        print("\nBaymax:", reply)
        speak(reply)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Baymax Voice Agent — Gemma-2B-IT (CUDA) version

Flow:
  Record -> Whisper (CPU) -> Gemma-2b-it (CUDA, bfloat16) -> Kokoro (CPU) TTS -> Play

Requirements (install in your venv before running):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers accelerate sentencepiece
  pip install faster-whisper
  pip install kokoro
  pip install sounddevice soundfile numpy
"""

import os
# Use GPU 0 for the LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import tempfile
import traceback
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Kokoro TTS (we force CPU for stability)
from kokoro import KPipeline


# ---------------------------
# Config
# ---------------------------
WHISPER_MODEL = "small"
WHISPER_SR = 16000
RECORD_SECONDS = 4

LLM_ID = "google/gemma-2b-it"   # gemma-2b instruction-tuned (Italian variants often work well; change if needed)

KOKORO_SR = 24000
KOKORO_VOICE = "af_bella"
KOKORO_LANG = "a"

# Baymax persona (short & human)
BAYMAX_PROMPT = (
    "You are Baymax, a warm and gentle companion. Speak in short, natural sentences (1–2 sentences). "
    "No headings, no lists, no special formatting. Be kind, calm, and human."
)

# Generation settings
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.1
DO_SAMPLE = True

# ---------------------------
# Globals
# ---------------------------
_whisper = None
_tokenizer = None
_model = None
_kokoro = None


# ---------------------------
# Whisper (CPU)
# ---------------------------
def load_whisper():
    global _whisper
    if _whisper:
        return _whisper
    print("🧠 Loading Whisper (CPU, int8)...")
    _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("✔ Whisper loaded.")
    return _whisper


def transcribe_wav(int16_audio, sr):
    whisper = load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        sf.write(path, int16_audio, sr)
        print("🧠 Transcribing...")
        segments, info = whisper.transcribe(path)
        text = "".join([s.text for s in segments]).strip()
        print("📝 Transcript:", text)
        return text
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ---------------------------
# Audio helper
# ---------------------------
def record_mic(seconds=RECORD_SECONDS, samplerate=WHISPER_SR):
    print(f"\n🎤 Recording {seconds}s — speak now...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    return (audio * 32767).astype(np.int16), samplerate


def play_audio_float32(arr, sr=KOKORO_SR):
    sd.play(arr, sr)
    sd.wait()


# ---------------------------
# Gemma-2b-it (CUDA BF16)
# ---------------------------
def load_gemma():
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    print("🤖 Loading Gemma-2B-IT (CUDA) ...")
    _tokenizer = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True)

    # Try loading in bfloat16 on device_map=auto (works well on Ampere+ / RTX 30xx)
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True  # gemma models may require trust_remote_code
        )
        print("✔ Gemma loaded with torch.bfloat16 (BF16) on GPU.")
    except Exception as e_bf:
        print("⚠ BF16 load failed, falling back to float16. Error:", e_bf)
        try:
            _model = AutoModelForCausalLM.from_pretrained(
                LLM_ID,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            print("✔ Gemma loaded with float16 on GPU.")
        except Exception as e_f16:
            print("⚠ float16 load failed, falling back to dtype=float32 on GPU. Errors:", e_f16)
            _model = AutoModelForCausalLM.from_pretrained(
                LLM_ID,
                device_map="auto",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            print("✔ Gemma loaded with float32 on GPU.")

    # Ensure model uses tokenizer.pad_token_id if absent
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    return _tokenizer, _model


def run_gemma(user_text: str) -> str:
    tokenizer, model = load_gemma()

    # Build prompt
    prompt = f"{BAYMAX_PROMPT}\nUser: {user_text}\nBaymax:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the assistant part after "Baymax:"
    if "Baymax:" in decoded:
        reply = decoded.split("Baymax:")[-1].strip()
    else:
        # Fallback: remove prompt portion
        reply = decoded[len(prompt) :].strip()
    # Safety fallback if empty
    if not reply or reply.isspace():
        reply = "I'm here to help. Tell me what you'd like to log."

    # Keep reply short: if model outputs long text, truncate politely to two sentences
    # (split by punctuation)
    # This ensures crisp human-like answers
    sentences = []
    for sep in (".", "?", "!"):
        if sep in reply:
            sentences = [s.strip() for s in reply.split(sep) if s.strip()]
            # restore punctuation on first two
            if sentences:
                reply = (sentences[0] + sep) + ((" " + sentences[1] + sep) if len(sentences) > 1 else "")
            break
    # as final guard, limit length to ~200 chars
    if len(reply) > 200:
        reply = reply[:200].rsplit(".", 1)[0] + "."
    return reply.strip()


# ---------------------------
# Kokoro TTS (force CPU)
# ---------------------------
def load_kokoro():
    global _kokoro
    if _kokoro:
        return _kokoro
    print("🔊 Loading Kokoro (CPU only)...")
    # Force device="cpu" to avoid CUDA issues
    _kokoro = KPipeline(lang_code=KOKORO_LANG, device="cpu")
    return _kokoro


def kokoro_speak(text: str):
    pipe = load_kokoro()
    chunks = []
    print("🔊 Generating speech...")
    for _, _, audio in pipe(text, voice=KOKORO_VOICE):
        chunks.append(audio)
    if chunks:
        audio = np.concatenate(chunks)
        play_audio_float32(audio, KOKORO_SR)


# ---------------------------
# Main loop
# ---------------------------
def run_turn():
    try:
        int16, sr = record_mic()
        user_text = transcribe_wav(int16, sr)
    except Exception as e:
        print("STT Error:", e)
        traceback.print_exc()
        return

    if not user_text or not user_text.strip():
        print("No speech detected.")
        return

    try:
        reply = run_gemma(user_text)
        print("\nBaymax:", reply)
    except Exception as e:
        print("LLM Error:", e)
        traceback.print_exc()
        reply = "I’m Baymax. I had trouble understanding that."

    try:
        kokoro_speak(reply)
    except Exception as e:
        print("TTS Error:", e)
        traceback.print_exc()


def main():
    print("\n=== Baymax Voice Agent (Gemma-2B-IT on CUDA) ===")
    print("Press Enter to record. Ctrl+C to quit.\n")

    load_whisper()
    # Delay heavy LLM load until ready; user will wait when first used
    load_gemma()

    while True:
        try:
            input("Press Enter to start recording...")
            run_turn()
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

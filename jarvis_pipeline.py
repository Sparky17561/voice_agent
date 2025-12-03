#!/usr/bin/env python3
"""
Baymax Voice Agent
Record → Whisper (CPU) → Baymax PromptTemplate (LangChain Runnable)
→ Groq Answer → Kokoro TTS → Play audio.
"""

import os

# Disable GPU everywhere to avoid cuDNN issues
os.environ["CT2_USE_ONNX"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_DISABLE_GPU"] = "1"
os.environ["ORT_DISABLE_CUDA"] = "1"
os.environ["ORT_DISABLE_DML"] = "1"
os.environ["ORT_DISABLE_OPENCL"] = "1"

# ------------------------------
# Standard imports
# ------------------------------
import sys
import tempfile
import traceback
import numpy as np
import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel

# ------------------------------
# LangChain (latest API)
# ------------------------------
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ------------------------------
# CONFIG
# ------------------------------

GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.1-8b-instant"

WHISPER_MODEL = "small"
WHISPER_SR = 16000
RECORD_SECONDS = 4

KOKORO_SR = 24000
KOKORO_VOICE = "af_bella"
KOKORO_LANG = "a"

# ------------------------------
# Globals
# ------------------------------
_whisper = None
groq_client = None
_kokoro = None
BAYMAX_PROMPT = ""

# ============================================================
#               LOAD PROMPT.TXT
# ============================================================

def load_system_prompt():
    global BAYMAX_PROMPT
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            BAYMAX_PROMPT = f.read().strip()
            print("✔ Loaded Baymax persona from prompt.txt")
    except Exception:
        BAYMAX_PROMPT = "You are Baymax, a gentle medical logging assistant."
        print("⚠ prompt.txt missing — using fallback prompt.")

# ============================================================
#                   AUDIO I/O
# ============================================================

def record_mic(seconds=RECORD_SECONDS, samplerate=WHISPER_SR):
    print(f"\n🎤 Recording {seconds}s — speak now...")
    audio = sd.rec(int(seconds * samplerate),
                   samplerate=samplerate,
                   channels=1,
                   dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    int16 = (audio * 32767).astype(np.int16)
    return int16, samplerate

def play_audio_float32(audio_float32, samplerate=KOKORO_SR):
    sd.play(audio_float32, samplerate)
    sd.wait()

# ============================================================
#                 WHISPER CPU
# ============================================================

def load_whisper():
    global _whisper
    if _whisper:
        return _whisper

    print("🧠 Loading Whisper (CPU)...")
    _whisper = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="float32"
    )
    print("✔ Whisper loaded (CPU)")
    return _whisper

def transcribe_tmp_wav(int16_audio, samplerate):
    whisper = load_whisper()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmpname = tmp.name

    try:
        sf.write(tmpname, int16_audio, samplerate)
        print("🧠 Transcribing...")
        segments, info = whisper.transcribe(tmpname)

        text = "".join([seg.text for seg in segments]).strip()
        print(f"📝 Transcript: {text}")
        return text
    finally:
        os.remove(tmpname)

# ============================================================
#             BAYMAX ANSWER (LangChain Runnable)
# ============================================================

def groq_answer(user_text):
    global BAYMAX_PROMPT, groq_client

    template = """
{system_prompt}

User said:
{user_input}

As Baymax, respond kindly in a short, natural, human way.
"""

    prompt = PromptTemplate(
        input_variables=["system_prompt", "user_input"],
        template=template
    )

    chain = (
        prompt
        | groq_client
        | StrOutputParser()
    )

    return chain.invoke({
        "system_prompt": BAYMAX_PROMPT,
        "user_input": user_text
    }).strip()

# ============================================================
#              KOKORO TTS (lazy load)
# ============================================================

def load_kokoro():
    global _kokoro
    if _kokoro:
        return _kokoro

    from kokoro import KPipeline
    print("🔊 Loading Kokoro CPU mode...")
    _kokoro = KPipeline(lang_code=KOKORO_LANG)
    return _kokoro

def kokoro_speak(text):
    pipeline = load_kokoro()
    out = []

    print("🔊 Generating speech...")
    for _, _, audio in pipeline(text, voice=KOKORO_VOICE):
        out.append(audio)

    if out:
        audio = np.concatenate(out)
        play_audio_float32(audio, samplerate=KOKORO_SR)

# ============================================================
#                     MAIN FLOW
# ============================================================

def run_turn():
    try:
        int16, sr = record_mic()
        text = transcribe_tmp_wav(int16, sr)
    except Exception as e:
        print("Whisper error:", e)
        traceback.print_exc()
        return

    if not text.strip():
        print("No speech detected.")
        return

    try:
        answer = groq_answer(text)
        print("\nBaymax:", answer)
    except Exception as e:
        print("Groq error:", e)
        answer = "I am Baymax. I could not process your request."

    kokoro_speak(answer)

# ============================================================
#                     ENTRY POINT
# ============================================================

def main():
    global groq_client
    load_system_prompt()
    load_whisper()

    # Start Groq LLM client (LangChain wrapper)
    groq_client = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.7,
        max_tokens=512
    )

    print("\n=== Baymax Voice Agent ===")
    print("Press Enter to record. Ctrl+C to quit.\n")

    while True:
        try:
            input("Press Enter to start recording...")
            run_turn()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()

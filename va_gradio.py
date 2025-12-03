#!/usr/bin/env python3
"""
Baymax Voice Agent - COMPLETE REWRITE FOR ZERO HISS

NEW APPROACH:
- Accumulate ALL audio first, then play as single buffer
- No streaming = no buffer gaps = no hiss
- Simple, clean audio pipeline
- Fast and reliable
"""

import io
import os
import tempfile
import threading

import numpy as np
import torch
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
from kokoro import KPipeline

# ========================= CONFIG =========================
WHISPER_MODEL = "tiny.en"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LOAD_4BIT = False
KOKORO_SR = 24000
KOKORO_LANG = "a"
KOKORO_VOICE = "af_bella"

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.7

# ========================= INTERRUPT FLAG =========================
_interrupt_lock = threading.Lock()
_interrupt_flag = False

def set_interrupt(v: bool):
    global _interrupt_flag
    with _interrupt_lock:
        _interrupt_flag = v

def get_interrupt() -> bool:
    with _interrupt_lock:
        return _interrupt_flag

# ========================= FASTAPI APP =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# ========================= LOAD MODELS =========================
print("🎤 Loading Whisper...")
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("✅ Whisper ready")

print("🧠 Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

if LOAD_4BIT:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        load_in_4bit=True, 
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=dtype
    )

model.eval()
print(f"✅ LLM ready on {next(model.parameters()).device}")

print("🔊 Loading Kokoro TTS...")
kokoro = KPipeline(lang_code=KOKORO_LANG, device="cpu")
print("✅ Kokoro ready\n")

# ========================= CLEAN AUDIO PROCESSING =========================
def generate_complete_audio(text: str) -> np.ndarray:
    """
    Generate complete audio buffer at once - NO STREAMING.
    This eliminates all hiss from buffer gaps.
    """
    all_chunks = []
    
    try:
        # Collect ALL audio chunks first
        for _, _, chunk in kokoro(text, voice=KOKORO_VOICE):
            if get_interrupt():
                break
            
            # Convert to numpy
            if torch.is_tensor(chunk):
                chunk = chunk.detach().cpu().numpy()
            
            chunk = np.asarray(chunk, dtype=np.float32).flatten()
            
            if len(chunk) > 0:
                all_chunks.append(chunk)
        
        if not all_chunks:
            print("⚠️ No audio generated")
            return np.array([], dtype=np.float32)
        
        # Concatenate into single continuous buffer
        full_audio = np.concatenate(all_chunks)
        
        # Simple normalization
        peak = np.abs(full_audio).max()
        if peak > 0:
            full_audio = full_audio * (0.9 / peak)
        
        return full_audio
        
    except Exception as e:
        print(f"❌ Audio generation error: {e}")
        return np.array([], dtype=np.float32)

def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """
    Convert float32 audio to complete WAV file bytes.
    """
    if len(audio) == 0:
        # Return silent audio if empty
        audio = np.zeros(sample_rate, dtype=np.float32)
    
    # Convert to int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    
    # Create WAV file
    byte_io = io.BytesIO()
    
    # WAV header
    num_samples = len(audio_int16)
    datasize = num_samples * 2  # 2 bytes per sample
    
    byte_io.write(b'RIFF')
    byte_io.write((datasize + 36).to_bytes(4, 'little'))
    byte_io.write(b'WAVE')
    byte_io.write(b'fmt ')
    byte_io.write((16).to_bytes(4, 'little'))
    byte_io.write((1).to_bytes(2, 'little'))   # PCM
    byte_io.write((1).to_bytes(2, 'little'))   # Mono
    byte_io.write((sample_rate).to_bytes(4, 'little'))
    byte_io.write((sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
    byte_io.write((2).to_bytes(2, 'little'))   # Block align
    byte_io.write((16).to_bytes(2, 'little'))  # Bits per sample
    byte_io.write(b'data')
    byte_io.write((datasize).to_bytes(4, 'little'))
    
    # Audio data
    byte_io.write(audio_int16.tobytes())
    
    return byte_io.getvalue()

# ========================= LLM PROMPT =========================
SYSTEM_PROMPT = (
    "You are Baymax, a warm healthcare companion. "
    "Be concise and friendly. Keep responses to 2-3 sentences. "
    "Never diagnose or prescribe."
)

def build_prompt(user_text: str) -> str:
    """Build chat prompt with proper formatting."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True
    )

# ========================= API ENDPOINTS =========================
@app.post("/interrupt")
async def interrupt_endpoint():
    """Handle user interruption."""
    set_interrupt(True)
    print("⚠️ User interrupted")
    return JSONResponse({"ok": True})

@app.post("/process")
async def process_audio(file: UploadFile):
    """
    Main audio processing pipeline - returns complete audio file.
    NO STREAMING = NO HISS!
    """
    set_interrupt(False)
    
    # Save uploaded audio
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Transcribe with Whisper
    user_text = ""
    try:
        segments, _ = whisper.transcribe(tmp_path, language="en")
        user_text = " ".join([s.text for s in segments]).strip()
    except Exception as e:
        print(f"❌ Whisper error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    # Handle empty transcription
    if not user_text:
        print("❌ No speech detected")
        audio = generate_complete_audio("I didn't catch that. Please try again.")
        wav_bytes = audio_to_wav_bytes(audio)
        return Response(content=wav_bytes, media_type="audio/wav")
    
    print(f"🎤 User: {user_text}")
    
    # Generate LLM response
    reply_text = ""
    try:
        prompt = build_prompt(user_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_len:]
        
        if generated_ids.numel() > 0:
            reply_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        if not reply_text:
            reply_text = "I'm here to help. What's on your mind?"
            
    except Exception as e:
        print(f"❌ LLM error: {e}")
        reply_text = "I'm having trouble processing that. Could you try again?"
    
    print(f"🤖 Baymax: {reply_text}")
    
    # Generate complete audio at once
    audio = generate_complete_audio(reply_text)
    wav_bytes = audio_to_wav_bytes(audio)
    
    return Response(content=wav_bytes, media_type="audio/wav")

# ========================= WEB UI =========================
HTML_UI = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Baymax Voice Agent</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    padding: 20px;
  }
  .container {
    background: white;
    border-radius: 20px;
    padding: 40px;
    max-width: 500px;
    width: 100%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    text-align: center;
  }
  h1 {
    font-size: 36px;
    margin-bottom: 10px;
    color: #333;
  }
  .subtitle {
    color: #666;
    margin-bottom: 30px;
    font-size: 16px;
  }
  #status {
    font-size: 18px;
    color: #667eea;
    margin: 20px 0;
    font-weight: 600;
    min-height: 27px;
  }
  .spacebar-icon {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px 40px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 700;
    margin: 30px auto;
    display: inline-block;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    transition: transform 0.1s;
  }
  .spacebar-icon.active {
    transform: scale(0.95);
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 0 0 8px rgba(245, 87, 108, 0.2);
  }
  .instructions {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    text-align: left;
    font-size: 14px;
    line-height: 1.8;
    color: #555;
    margin-top: 30px;
  }
  .instructions strong {
    color: #333;
  }
  .volume-indicator {
    width: 100%;
    height: 6px;
    background: #e0e0e0;
    border-radius: 3px;
    margin-top: 15px;
    overflow: hidden;
  }
  .volume-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    width: 0%;
    transition: width 0.1s;
  }
</style>
</head>
<body>
  <div class="container">
    <h1>🤖 Baymax</h1>
    <div class="subtitle">Your Healthcare Companion</div>
    
    <div class="spacebar-icon" id="spacebarIcon">⎵ SPACEBAR</div>
    
    <div id="status">Hold SPACEBAR to speak</div>
    
    <div class="volume-indicator">
      <div class="volume-bar" id="volumeBar"></div>
    </div>
    
    <div class="instructions">
      <strong>How to use:</strong><br>
      • Hold SPACEBAR to record your voice<br>
      • Release to send and get response<br>
      • Speak during playback to interrupt<br>
      • Keep questions brief for best results
    </div>
  </div>

<script>
let audioCtx = null;
let micStream = null;
let analyser = null;
let mediaRecorder = null;
let chunks = [];
let isPlaying = false;
let spaceDown = false;
let monitorInterval = null;
let currentAudioSource = null;

const statusEl = document.getElementById('status');
const spacebarIcon = document.getElementById('spacebarIcon');
const volumeBar = document.getElementById('volumeBar');

async function initAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (!micStream) {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      const source = audioCtx.createMediaStreamSource(micStream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      console.log('✅ Audio initialized');
    } catch (e) {
      console.error('❌ Microphone error:', e);
      statusEl.textContent = 'Microphone access denied';
    }
  }
}

function getMicLevel() {
  if (!analyser) return 0;
  const dataArray = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(dataArray);
  
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const normalized = (dataArray[i] - 128) / 128;
    sum += normalized * normalized;
  }
  return Math.sqrt(sum / dataArray.length);
}

function updateVolume() {
  const level = getMicLevel();
  volumeBar.style.width = Math.min(level * 500, 100) + '%';
}

async function startRecording() {
  await initAudio();
  chunks = [];
  mediaRecorder = new MediaRecorder(micStream);
  mediaRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  mediaRecorder.start();
  
  spacebarIcon.classList.add('active');
  statusEl.textContent = '🎤 Listening...';
  
  monitorInterval = setInterval(updateVolume, 50);
}

function stopRecording() {
  return new Promise(resolve => {
    if (monitorInterval) {
      clearInterval(monitorInterval);
      monitorInterval = null;
    }
    volumeBar.style.width = '0%';
    
    mediaRecorder.onstop = () => {
      resolve(new Blob(chunks, { type: 'audio/webm' }));
    };
    mediaRecorder.stop();
  });
}

async function sendAudio(blob) {
  spacebarIcon.classList.remove('active');
  statusEl.textContent = '⚙️ Processing...';
  
  const formData = new FormData();
  formData.append('file', blob, 'audio.webm');
  
  try {
    const response = await fetch('/process', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) throw new Error('Server error');
    
    // Get complete audio file
    const audioBlob = await response.blob();
    await playCompleteAudio(audioBlob);
    
  } catch (e) {
    console.error('❌ Error:', e);
    statusEl.textContent = '❌ Error - try again';
    setTimeout(() => {
      statusEl.textContent = 'Hold SPACEBAR to speak';
    }, 2000);
  }
}

async function playCompleteAudio(blob) {
  await initAudio();
  
  // Decode complete audio file
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  
  isPlaying = true;
  statusEl.textContent = '🔊 Speaking...';
  
  // Create audio source
  currentAudioSource = audioCtx.createBufferSource();
  currentAudioSource.buffer = audioBuffer;
  currentAudioSource.connect(audioCtx.destination);
  
  // Monitor for interruption
  const interruptMonitor = setInterval(() => {
    if (!isPlaying) {
      clearInterval(interruptMonitor);
      return;
    }
    const level = getMicLevel();
    if (level > 0.08) {
      console.log('⚠️ Interrupting...');
      if (currentAudioSource) {
        currentAudioSource.stop();
        currentAudioSource = null;
      }
      fetch('/interrupt', { method: 'POST' }).catch(() => {});
      isPlaying = false;
      clearInterval(interruptMonitor);
      statusEl.textContent = 'Hold SPACEBAR to speak';
    }
  }, 100);
  
  // Handle playback end
  currentAudioSource.onended = () => {
    isPlaying = false;
    clearInterval(interruptMonitor);
    currentAudioSource = null;
    statusEl.textContent = 'Hold SPACEBAR to speak';
  };
  
  // Start playback
  currentAudioSource.start(0);
}

window.addEventListener('keydown', async (e) => {
  if (e.code === 'Space' && !spaceDown && !isPlaying) {
    e.preventDefault();
    spaceDown = true;
    try {
      await startRecording();
    } catch (err) {
      console.error('❌ Recording error:', err);
      statusEl.textContent = '❌ Microphone error';
      spaceDown = false;
    }
  }
});

window.addEventListener('keyup', async (e) => {
  if (e.code === 'Space' && spaceDown) {
    e.preventDefault();
    spaceDown = false;
    try {
      const blob = await stopRecording();
      await sendAudio(blob);
    } catch (err) {
      console.error('❌ Send error:', err);
      statusEl.textContent = '❌ Error sending audio';
    }
  }
});

window.addEventListener('load', () => {
  initAudio().catch(() => {});
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HTMLResponse(HTML_UI)

# ========================= RUN SERVER =========================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Baymax Voice Agent (Zero Hiss - Complete Audio)")
    print("=" * 60)
    print(f"\n📍 Server: http://127.0.0.1:7860")
    print(f"🧠 LLM: {MODEL_NAME}")
    print(f"🎤 Whisper: {WHISPER_MODEL}")
    print(f"🔊 TTS: Kokoro-82M")
    print("\n💡 Hold SPACEBAR to speak in the browser")
    print("✨ NEW: Complete audio buffer = ZERO hiss!\n")
    
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
# 🤖 Baymax Voice Agent

A fully **offline, GPU-accelerated voice assistant** that listens, understands, and responds naturally—all running locally on your machine.

Built with state-of-the-art AI models:
- 🧠 **Gemma 2B-IT** (GPU) — Fast, intelligent language model
- 🎤 **Whisper** (CPU) — Accurate speech-to-text
- 🔊 **Kokoro-82M** (CPU) — Natural-sounding text-to-speech

---

## ✨ Features

- **100% Offline** — No internet required, complete privacy
- **GPU Accelerated** — Fast response times with CUDA support
- **Natural Conversations** — Short, friendly responses like a real assistant
- **Easy Setup** — Simple installation with clear instructions
- **Cross-Platform** — Works on Windows, Linux, and macOS (with CUDA GPU)

---

## 🎯 Demo

```
=== Baymax Voice Agent (Gemma-2B-IT on CUDA) ===
Press Enter to start recording...

[Recording for 4 seconds...]
You: "What's the weather like today?"
Baymax: "I don't have access to real-time weather data, but I can help you find that information!"
[Speaking response...]
```

---

## 📋 Prerequisites

- **Python 3.9 - 3.11** (recommended: 3.10)
- **NVIDIA GPU** with CUDA 12.1+ support
- **8GB+ VRAM** (16GB recommended)
- **Working microphone**
- **Windows 10/11** (or Linux with CUDA drivers)

---

## 🚀 Quick Start

### 1️⃣ Clone and Setup

```bash
git clone <your-repo-url>
cd baymax-voice-agent
python -m venv venv
```

**Activate environment:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

---

### 2️⃣ Install PyTorch with CUDA (CRITICAL)

**You MUST install PyTorch with CUDA support first:**

```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU is detected:**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
CUDA Available: True
CUDA Version: 12.1
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `transformers` — LLM runtime
- `accelerate` — GPU optimization
- `faster-whisper` — Speech recognition
- `kokoro-onnx` — Text-to-speech
- `sounddevice` — Audio recording
- `scipy` — Audio processing

---

### 4️⃣ Configure Microphone (Windows)

If recording fails:

1. Open **Settings** → **Privacy & Security** → **Microphone**
2. Enable **"Let apps access your microphone"**
3. Ensure **Python** is allowed

---

### 5️⃣ Run Baymax

```bash
python new.py
```

**First run will:**
- Download Gemma-2B-IT (~5GB)
- Download Whisper-Small (~500MB)
- Download Kokoro-82M (~100MB)

**Then you'll see:**
```
=== Baymax Voice Agent (Gemma-2B-IT on CUDA) ===
Press Enter to start recording...
```

Press **Enter**, speak for **4 seconds**, and Baymax will respond!

---

## 📁 Project Structure

```
baymax-voice-agent/
│
├── new.py              # Main voice agent script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── prompt.txt          # (Optional) Custom system prompt
│
└── venv/               # Virtual environment (created by you)
```

---

## 🔧 Configuration

### Adjust Recording Duration

In `new.py`, modify:
```python
duration = 4  # Change to 5, 6, etc. for longer recordings
```

### Change Response Length

In `new.py`, find:
```python
max_new_tokens=50  # Increase for longer responses
```

### Switch Models

**Smaller/Faster LLM:**
```python
model_name = "google/gemma-2b-it"  # Current
# Try: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Better Speech Recognition:**
```python
model_size = "small"  # Current
# Options: "tiny", "base", "small", "medium", "large"
```

---

## 🐛 Troubleshooting

### ❌ `CUDA out of memory`
**Solution:** Reduce model precision or use smaller models
```python
torch_dtype=torch.float16  # Instead of bfloat16
```

### ❌ `No CUDA devices available`
**Check:**
1. NVIDIA drivers installed? (`nvidia-smi`)
2. PyTorch installed with CUDA? (See Step 2)
3. GPU supports CUDA 11.8+?

### ❌ Microphone not working
**Windows:** Check Privacy settings (see Step 4)  
**Linux:** Install `portaudio19-dev`
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

### ❌ Slow first response
**Normal!** First inference loads models into GPU memory (~30s)  
Subsequent responses are fast (~2-3s)

### ❌ `RuntimeError: device-side assert triggered`
**Solution:** Switch to FP16
```python
torch_dtype=torch.float16
```

---

## 🎛️ Advanced Configuration

### Use Custom System Prompt

Create `prompt.txt`:
```
You are Baymax, a helpful healthcare companion. Be concise and caring.
```

Then in `new.py`:
```python
with open("prompt.txt", "r") as f:
    system_prompt = f.read()
```

### Enable Conversation Memory

Add conversation history to context:
```python
conversation_history = []
# In main loop:
conversation_history.append({"role": "user", "content": text})
conversation_history.append({"role": "assistant", "content": response})
```

---

## 📊 Performance Benchmarks

| Component | Hardware | Time |
|-----------|----------|------|
| Speech-to-Text (4s audio) | CPU | ~1-2s |
| LLM Response (50 tokens) | RTX 3060 | ~2-3s |
| Text-to-Speech | CPU | ~1-2s |
| **Total (first time)** | - | ~30-40s |
| **Total (subsequent)** | - | ~5-7s |

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

- **Google** — [Gemma Language Models](https://huggingface.co/google/gemma-2b-it)
- **OpenAI** — [Whisper Speech Recognition](https://github.com/openai/whisper)
- **Systran** — [Faster-Whisper Implementation](https://github.com/SYSTRAN/faster-whisper)
- **Kokoro** — [Natural TTS System](https://huggingface.co/hexgrad/Kokoro-82M)
- **HuggingFace** — [Transformers Library](https://huggingface.co/docs/transformers)
- **PyTorch** — [Deep Learning Framework](https://pytorch.org/)

---

## 🔮 Roadmap

- [ ] Add wake word detection ("Hey Baymax")
- [ ] Implement conversation memory
- [ ] Support multiple languages
- [ ] Add voice activity detection (stop on silence)
- [ ] Create web interface
- [ ] Docker containerization
- [ ] Add emotion detection in voice
- [ ] Integrate function calling (weather, calculations, etc.)

---

## 💬 Support

Having issues? 

- 📖 Check the [Troubleshooting](#-troubleshooting) section
- 🐛 Open an [Issue](https://github.com/yourusername/baymax-voice-agent/issues)
- 💡 Start a [Discussion](https://github.com/yourusername/baymax-voice-agent/discussions)

---

## ⭐ Star History

If this project helped you, consider giving it a star! ⭐

---

**Built with ❤️ for the AI community**
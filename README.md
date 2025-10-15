Realtime Voice Q&A (OpenRouter + DeepSeek)
A local, privacy‑friendly voice assistant that turns your microphone input into text answers:

WebRTC mic capture with turn detection (VAD + energy fallback)

Local transcription using faster‑whisper (CPU)

LLM answer via OpenRouter (default model: deepseek/deepseek-chat)

Clean Gradio UI with progress messages

Demo flow
Press Record and speak.

The app detects end of speech automatically.

You’ll see “Transcribing…”, then your transcript, then “Calling OpenRouter…”, and finally the model’s text reply.

Tech stack
Python 3.10, FastAPI, Gradio

VAD (WebRTC) + energy fallback

faster‑whisper for ASR (tiny/base/small/medium on CPU)

OpenRouter Chat Completions (DeepSeek)

1. Prerequisites
Python 3.10 (Windows/Mac/Linux)

A working microphone and browser mic permission

An OpenRouter API key (free to start)

Optional but recommended:

Git for cloning the repo

2. Setup
Clone and enter the project:

text
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
Create and activate a virtual environment.

Windows (CMD):

text
py -3.10 -m venv venv
venv\Scripts\activate
PowerShell:

text
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
Linux/macOS:

text
python3.10 -m venv venv
source venv/bin/activate
Install dependencies:

text
python -m pip install --upgrade pip setuptools wheel
pip install fastrtc fastapi uvicorn gradio numpy python-dotenv faster-whisper scipy webrtcvad requests
Create a .env file in the project root (exactly as below; no quotes):

text
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENROUTER_MODEL=deepseek/deepseek-chat
MODE=UI
USE_STREAM=0
REPLY_LANG=en
WHISPER_SIZE=tiny
Notes:

Keep USE_STREAM=0 initially for reliability.

Set REPLY_LANG=mr to always answer in Marathi.

Start with WHISPER_SIZE=tiny for speed; switch to base or small for accuracy.

3. Run
text
python app.py
Open the printed local URL (e.g., http://127.0.0.1:7860), allow the mic, ask a question, then pause briefly to end your turn.

4. Configuration
Environment variables (in .env):

OPENROUTER_API_KEY: your key.

OPENROUTER_MODEL: model id, default deepseek/deepseek-chat (you can try deepseek/deepseek-r1 later).

MODE: UI to launch the built‑in Gradio interface.

USE_STREAM: 0 keeps non‑streaming replies (recommended); 1 enables SSE streaming (not enabled in this app version).

REPLY_LANG: auto, en, or mr.

WHISPER_SIZE: tiny, base, small, or medium.

ASR_TIMEOUT_S: optional, default 8 seconds for transcription.

DEBUG_SAVE: set to 1 to keep the last recorded WAV for debugging.

5. How it works
Browser mic sends 24 kHz PCM frames to the backend.

Audio resampled to 16 kHz, VAD + energy fallback detects speech turns.

Each turn is saved to a temp WAV and transcribed by faster‑whisper on CPU.

Transcript is sent to OpenRouter’s chat completion endpoint using the configured model.

Assistant’s text answer is appended to the chat.

6. Troubleshooting
.env parsing error

Ensure no quotes or stray characters; first line should be exactly:

OPENROUTER_API_KEY=sk-or-xxxx

401 Unauthorized

Wrong or missing API key; update .env, restart terminal, and rerun.

404 model not found

Use a precise id like deepseek/deepseek-chat (avoid “-latest”).

“[Unintelligible]” transcripts

Increase your OS mic input level, use a quiet environment, or set WHISPER_SIZE=base.

No mic waveform / no prompt to allow mic

Check browser permissions; try a fresh tab or different browser.

High CPU

Keep WHISPER_SIZE=tiny or base; ask shorter questions.

7. Quick key verification (optional)
Run this in the same venv:

text
python - << "PY"
import requests, os
api_key = os.getenv("OPENROUTER_API_KEY")
r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": "deepseek/deepseek-chat",
        "messages": [{"role":"user","content":"Say hello in one short sentence."}]
    },
    timeout=30
)
print(r.status_code, r.text[:200])
PY
8. Roadmap
Optional SSE partial-token streaming

UI selector for model and language

Hotkeys (press-to-talk)

Dockerfile and simple deploy guide

9. Security
Never commit .env or keys to git.

Use separate keys for development and demo.

Rotate keys periodically.

10. License
Add a LICENSE file (e.g., MIT) and update this section accordingly.

11. Acknowledgements
OpenRouter for routing to many models (DeepSeek by default)

faster‑whisper for efficient CPU transcription

WebRTC VAD for robust speech detection

Gradio for the simple UI


import asyncio
import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    Stream,
    get_twilio_turn_credentials,
    wait_for_item,
)
from gradio.utils import get_space

# Audio / DSP
import webrtcvad
from scipy.signal import resample_poly
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel

# ----------------------- Config -----------------------
load_dotenv()

# Keys and model
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing. Put it in .env as OPENROUTER_API_KEY=YOUR_KEY")

# Free/low‑cost default that avoids “-latest”: change to deepseek/deepseek-r1 later if you want
MODEL_ID = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Controls
USE_STREAM = os.getenv("USE_STREAM", "0") != "0"     # keep off first; SSE is optional
REPLY_LANG = os.getenv("REPLY_LANG", "auto").lower() # auto|en|mr
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "tiny")
ASR_TIMEOUT_S = float(os.getenv("ASR_TIMEOUT_S", "8"))
DEBUG_SAVE = os.getenv("DEBUG_SAVE", "0") == "1"

# I/O and ASR params
INPUT_SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 24000
ASR_SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES_16K = int(ASR_SAMPLE_RATE * FRAME_MS / 1000)

# VAD thresholds (permissive)
START_VOICE_FRAMES = 1
END_SILENCE_FRAMES = 8

# Load Whisper (CPU)
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# OpenRouter endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    # These two are recommended by OpenRouter; customize if you host elsewhere
    "HTTP-Referer": "http://localhost:7860",
    "X-Title": "Realtime Mic Pipeline",
}

def call_openrouter(messages, temperature=0.3, stream=False, timeout=45):
    """
    Minimal non-streaming call to OpenRouter chat completions.
    messages: [{"role":"system"/"user","content":"..."}]
    """
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "stream": bool(stream),
    }
    resp = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=payload, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

class OpenRouterRealtimeHandler(AsyncStreamHandler):
    """
    Mic in (24 kHz) -> 16 kHz -> VAD + energy fallback -> Whisper transcription ->
    OpenRouter DeepSeek answer (text) -> chat UI.
    """
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )
        self.output_queue = asyncio.Queue()

        self.vad = webrtcvad.Vad(1)  # permissive
        self.frame_residual = np.zeros(0, dtype=np.int16)

        self.in_speech = False
        self.voiced_frames = 0
        self.silence_frames = 0
        self.current_speech = []

        self.turn_lock = asyncio.Lock()

        # Energy fallback
        self.noise_floor = 300.0
        self.min_energy_thr = 200.0
        self.energy_multiplier = 2.0

        self.max_frames_per_turn = int((15_000 / FRAME_MS))
        self.debug = True

    def copy(self):
        return OpenRouterRealtimeHandler()

    async def start_up(self):
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": f"Using OpenRouter model: {MODEL_ID}"})
        )

    def _to_16k_mono_int16(self, pcm24k: np.ndarray) -> np.ndarray:
        if pcm24k.ndim > 1:
            pcm24k = pcm24k.squeeze()
        x = pcm24k.astype(np.float32) / 32768.0
        y = resample_poly(x, up=2, down=3)  # 24k -> 16k
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16)

    def _frames_30ms(self, samples16k: np.ndarray):
        data = samples16k
        idx = 0
        total = len(data)
        while idx + FRAME_SAMPLES_16K <= total:
            frame = data[idx: idx + FRAME_SAMPLES_16K]
            yield frame.tobytes(), frame
            idx += FRAME_SAMPLES_16K
        self.frame_residual = data[idx:]

    def _reset_turn(self):
        self.in_speech = False
        self.voiced_frames = 0
        self.silence_frames = 0
        self.current_speech = []

    def _rms(self, x: np.ndarray) -> float:
        x = x.astype(np.float32)
        return float(np.sqrt(np.mean(x * x)) + 1e-6)

    def _is_speech(self, frame_bytes: bytes, frame_i16: np.ndarray) -> bool:
        vad_flag = self.vad.is_speech(frame_bytes, ASR_SAMPLE_RATE)
        rms = self._rms(frame_i16)
        alpha = 0.98 if not self.in_speech else 0.995
        self.noise_floor = alpha * self.noise_floor + (1 - alpha) * rms
        dynamic_thr = max(self.min_energy_thr, self.noise_floor * self.energy_multiplier)
        energy_flag = rms > dynamic_thr
        if self.debug:
            print(f"VAD={vad_flag} RMS={rms:.1f} floor={self.noise_floor:.1f} thr={dynamic_thr:.1f}")
        return vad_flag or energy_flag

    def _blocking_transcribe(self, wav_path: str) -> str:
        segments, _ = whisper_model.transcribe(
            wav_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250),
        )
        parts = []
        for seg in segments:
            if seg.text:
                parts.append(seg.text.strip())
        return " ".join(parts).strip()

    async def _finalize_turn_and_respond(self, speech16k: np.ndarray):
        if self.debug:
            print(f"Finalizing turn with {len(speech16k)} samples")

        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": "Transcribing..."})
        )

        # Write temp wav and transcribe with timeout
        keep_file = DEBUG_SAVE
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            wav_write(wav_path, ASR_SAMPLE_RATE, speech16k)
            try:
                user_text = await asyncio.wait_for(
                    asyncio.to_thread(self._blocking_transcribe, wav_path),
                    timeout=ASR_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                user_text = ""
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": "Transcribe timeout. Please speak a bit longer or closer to the mic."})
                )
            except Exception as e:
                user_text = ""
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": f"Transcribe error: {e}"})
                )
        finally:
            if not keep_file:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            else:
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": f"Saved last audio to: {wav_path}"})
                )

        # Always show what was heard
        await self.output_queue.put(
            AdditionalOutputs({"role": "user", "content": user_text or "[Unintelligible]"})
        )

        # Build prompt and system instruction
        if REPLY_LANG == "en":
            instr = "Answer concisely in English."
        elif REPLY_LANG == "mr":
            instr = "Answer concisely in Marathi."
        else:
            instr = "Answer concisely in the same language as the user."
        prompt = user_text or "No recognizable speech."
        system_msg = instr

        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": "Calling OpenRouter..."})
        )

        # Non-streaming call (more robust across networks)
        final_text = ""
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            final_text = call_openrouter(messages, temperature=0.3, stream=False, timeout=45) or "No response text."
        except Exception as e:
            final_text = f"OpenRouter error: {e}"

        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": final_text})
        )

    async def _process_turn(self, speech16k: np.ndarray):
        async with self.turn_lock:
            if speech16k.size < FRAME_SAMPLES_16K * 4:
                if self.debug:
                    print("Turn discarded: too short")
                return
            await self._finalize_turn_and_respond(speech16k)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, array = frame
        pcm24k = array.squeeze().astype(np.int16)
        pcm16k = self._to_16k_mono_int16(pcm24k)

        if self.frame_residual.size:
            pcm16k = np.concatenate([self.frame_residual, pcm16k])
            self.frame_residual = np.zeros(0, dtype=np.int16)

        for frame_bytes, frame_i16 in self._frames_30ms(pcm16k):
            is_speech = self._is_speech(frame_bytes, frame_i16)

            if is_speech:
                self.voiced_frames += 1
                self.silence_frames = 0
                self.current_speech.append(frame_i16)
                if not self.in_speech and self.voiced_frames >= START_VOICE_FRAMES:
                    self.in_speech = True
                    if self.debug:
                        print("Speech START")
            else:
                if self.in_speech:
                    self.silence_frames += 1
                    self.current_speech.append(frame_i16)
                    if self.silence_frames >= END_SILENCE_FRAMES or len(self.current_speech) >= self.max_frames_per_turn:
                        if self.debug:
                            print("Speech END")
                        speech = np.concatenate(self.current_speech) if self.current_speech else np.zeros(0, dtype=np.int16)
                        self._reset_turn()
                        asyncio.create_task(self._process_turn(speech))
                else:
                    self.voiced_frames = 0
                    self.silence_frames = 0

    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        pass


def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


chatbot = gr.Chatbot(type="messages")
latest_message = gr.Textbox(type="text", visible=False)

stream = Stream(
    OpenRouterRealtimeHandler(),
    mode="send-receive",
    modality="audio",
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

app = FastAPI()
stream.mount(app)


@app.get("/")
async def _():
    html = "<html><body><h3>Start with MODE=UI and open the Gradio link in console.</h3></body></html>"
    return HTMLResponse(content=html)


@app.get("/outputs")
def _(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            s = json.dumps(output.args[0])
            yield f"event: output\ndata: {s}\n\n"
    return StreamingResponse(output_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    mode = os.getenv("MODE", "UI")
    if mode == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)

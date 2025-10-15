import asyncio
import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
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

# Gemini
import google.generativeai as genai

# -------------------------------------------------
# Config
# -------------------------------------------------
load_dotenv()

# Use .env if set; your inline key works for quick tests but rotate it later
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyACxdgQK9bGCVJcP7s7Ge-3aaikrx_K-R8"
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing.")
genai.configure(api_key=GEMINI_API_KEY)

def pick_gemini_model():
    preferred = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
    candidates = [
        preferred,
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
    ]
    try:
        models = list(genai.list_models())
        supported = {
            m.name.split("/")[-1]
            for m in models
            if getattr(m, "supported_generation_methods", None)
            and "generateContent" in m.supported_generation_methods
        }
        for c in candidates:
            if c in supported:
                return c
    except Exception:
        pass
    return candidates[0]

GEMINI_MODEL_NAME = pick_gemini_model()
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

cur_dir = Path(__file__).parent

# WebRTC I/O sample rates
INPUT_SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 24000  # no audio out; text-only

# ASR config
ASR_SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES_16K = int(ASR_SAMPLE_RATE * FRAME_MS / 1000)  # 480

# VAD thresholds (more permissive)
START_VOICE_FRAMES = 1      # ~30 ms to start speech
END_SILENCE_FRAMES = 8      # ~240 ms of silence to end speech

# Whisper model (CPU). Options: "tiny", "base", "small", "medium"
WHISPER_SIZE = "small"
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


class GeminiRealtimeTextHandler(AsyncStreamHandler):
    """
    Mic in (24 kHz) -> resample to 16 kHz -> VAD + energy fallback -> Whisper transcription ->
    Gemini answer (text) -> push to chat UI.
    """
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )
        self.output_queue = asyncio.Queue()

        # More permissive VAD
        self.vad = webrtcvad.Vad(1)  # 0..3 (lower = less strict)
        self.frame_residual = np.zeros(0, dtype=np.int16)

        self.in_speech = False
        self.voiced_frames = 0
        self.silence_frames = 0
        self.current_speech = []  # list of int16 frames @16kHz

        self.turn_lock = asyncio.Lock()

        # Energy fallback state
        self.noise_floor = 300.0  # conservative initial RMS
        self.min_energy_thr = 300.0  # absolute minimum RMS threshold
        self.energy_multiplier = 2.5  # how far above noise_floor counts as speech

        # Optional: cap each turn (e.g., 15 s) to ensure a response
        self.max_frames_per_turn = int((15_000 / FRAME_MS))  # 15s at 30ms/frame

        # Debug
        self.debug = True

    def copy(self):
        return GeminiRealtimeTextHandler()

    async def start_up(self):
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": f"Using Gemini model: {GEMINI_MODEL_NAME}"})
        )

    def _to_16k_mono_int16(self, pcm24k: np.ndarray) -> np.ndarray:
        if pcm24k.ndim > 1:
            pcm24k = pcm24k.squeeze()
        x = pcm24k.astype(np.float32) / 32768.0
        # 24k -> 16k (2/3)
        y = resample_poly(x, up=2, down=3)
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
        # WebRTC VAD decision
        vad_flag = self.vad.is_speech(frame_bytes, ASR_SAMPLE_RATE)

        # Energy fallback with adaptive noise floor
        rms = self._rms(frame_i16)
        # Exponential moving average for noise floor
        alpha = 0.98 if not self.in_speech else 0.995  # track slower during speech
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

        # Write temp WAV for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            wav_write(wav_path, ASR_SAMPLE_RATE, speech16k)
            user_text = await asyncio.to_thread(self._blocking_transcribe, wav_path)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

        await self.output_queue.put(
            AdditionalOutputs({"role": "user", "content": user_text or "[Unintelligible]"})
        )

        # Build instruction
        lang = os.getenv("REPLY_LANG", "auto").lower()  # auto|en|mr
        if lang == "en":
            instr = "Answer concisely in English."
        elif lang == "mr":
            instr = "Answer concisely in Marathi."
        else:
            instr = "Answer concisely in the same language as the user."
        prompt = f"{instr}\n\nUser said:\n{user_text or 'No recognizable speech.'}"

        # Stream from Gemini (iterator)
        try:
            stream = gemini_model.generate_content(prompt, stream=True)
            buf = []
            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    buf.append(chunk.text)
            final_text = "".join(buf).strip() if buf else "No response text."
        except Exception as e:
            final_text = f"Gemini error: {e}"

        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": final_text})
        )

    async def _process_turn(self, speech16k: np.ndarray):
        async with self.turn_lock:
            if speech16k.size < FRAME_SAMPLES_16K * 4:  # ignore very short utterances
                if self.debug:
                    print("Turn discarded: too short")
                return
            await self._finalize_turn_and_respond(speech16k)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        # frame: (rate, np.ndarray shape (1, N) int16 @24kHz)
        _, array = frame
        pcm24k = array.squeeze().astype(np.int16)
        pcm16k = self._to_16k_mono_int16(pcm24k)

        # stitch residual
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
                        speech = (
                            np.concatenate(self.current_speech)
                            if self.current_speech
                            else np.zeros(0, dtype=np.int16)
                        )
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


# Build UI/server
chatbot = gr.Chatbot(type="messages")
latest_message = gr.Textbox(type="text", visible=False)

stream = Stream(
    GeminiRealtimeTextHandler(),
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

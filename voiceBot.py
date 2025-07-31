import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import torch
import sys
import requests
import json
import io
from dotenv import load_dotenv
import os
from kokoro import KPipeline
import scipy.signal
import nltk

# Download NLTK sentence tokenizer
#nltk.download('punkt_tab')

# Load environment variables
load_dotenv()

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SILENCE_THRESHOLD = 500  # Amplitude threshold for silence detection
SILENCE_DURATION = 2.0  # Seconds of silence to stop recording

# Initialize Kokoro TTS pipeline
print(f"Loading Kokoro TTS model...")
try:
    tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')  # 'a' for American English
except Exception as e:
    print(f"Error initializing Kokoro TTS: {e}")
    sys.exit(1)

# Initialize PyAudio
p = pyaudio.PyAudio()

def get_gemini_response(text):
    """Get response from Gemini 2.5 Flash"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{
                    "text": text
                }]
            }]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return "Sorry, I couldn't process that. Please try again."

def text_to_speech(text):
    """Convert text to speech using Kokoro TTS"""
    try:
        # Split text into sentences to manage memory
        sentences = nltk.sent_tokenize(text)
        audio_chunks = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            # Generate audio for each sentence
            generator = tts_pipeline(sentence, voice='af_bella')  # Use supported voice
            for _, _, audio in generator:
                # Audio is a numpy array at 24kHz, resample to 16kHz
                audio = scipy.signal.resample_poly(audio, 16000, 24000)
                # Normalize to int16 range
                audio = (audio * 32767 / np.max(np.abs(audio))).astype(np.int16)
                audio_chunks.append(audio)

        # Concatenate all audio chunks
        if not audio_chunks:
            return None
        waveform = np.concatenate(audio_chunks)

        # Create BytesIO buffer for PyAudio
        audio_buffer = io.BytesIO()
        audio_buffer.write(waveform.tobytes())
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        print(f"Error in Kokoro TTS: {e}")
        return None

def play_audio(audio_data):
    """Play audio using PyAudio"""
    stream_out = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True
    )
    stream_out.write(audio_data)
    stream_out.stop_stream()
    stream_out.close()

def is_silent(audio_data, threshold=SILENCE_THRESHOLD):
    """Check if audio chunk is silent based on amplitude threshold"""
    return np.abs(audio_data).max() < threshold

# === LOAD MODEL ===
print(f"Loading Whisper model on {DEVICE}...")
model = WhisperModel("base", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")

try:
    # Use default input device
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    print(f"Stream active: {stream.is_active()}")

    # Main conversation loop
    while True:
        print("Recording... (Speak, and it will stop automatically after silence)")
        frames = []
        silent_chunks = 0
        max_silent_chunks = int(SILENCE_DURATION * RATE / CHUNK)

        # Record until silence is detected
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            frames.append(audio_data)

            # Check for silence
            if is_silent(audio_data):
                silent_chunks += 1
                if silent_chunks >= max_silent_chunks:
                    break
            else:
                silent_chunks = 0

        # Process recorded audio
        if len(frames) == 0:
            print("No audio recorded, please try again.")
            continue

        full_audio = np.concatenate(frames).astype(np.float32) / 32768.0
        print("Transcribing...")
        segments, _ = model.transcribe(full_audio)

        # Process transcript
        transcript = " ".join(segment.text for segment in segments)
        print(f"Transcript: {transcript}")

        if transcript.lower().strip() in ["exit", "quit", "stop"]:
            print("Stopping conversation.")
            break

        # Get LLM response
        print("Generating response...")
        llm_response = get_gemini_response(transcript)
        print(f"LLM Response: {llm_response}")

        # Convert response to speech
        print("Converting to speech...")
        audio_content = text_to_speech(llm_response)
        if audio_content:
            print("🎵 Playing response...")
            play_audio(audio_content)

        print("-" * 40)

except KeyboardInterrupt:
    print("Stopped by user.")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
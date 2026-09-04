Voice AI Assistant

A simple voice assistant using Whisper, Gemini, and Kokoro TTS.

You speak into the microphone, Whisper converts your voice to text, Gemini generates a response, and Kokoro converts the response back to speech.

Setup

Create a .env file:

GEMINI_API_KEY=your_api_key_here


Run the project:

python app.py

Tech Used
Python
Faster-Whisper
Gemini 2.5 Flash
Kokoro TTS
PyAudio
PyTorch

The assistant automatically stops recording after a few seconds of silence.

Say "exit", "quit", or "stop" to end the conversation.

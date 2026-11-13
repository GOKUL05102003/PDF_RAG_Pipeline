import os
import sys
from pathlib import Path
from typing import Optional
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr

def get_recognizer() -> sr.Recognizer:
    """Initialize and configure SpeechRecognition Recognizer."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    return recognizer

def record_audio_via_sounddevice(sample_rate: int = 16000, max_duration: int = 10, silence_timeout: float = 1.5) -> Optional[sr.AudioData]:
    """
    Captures live audio from microphone using sounddevice.
    Bypasses PyAudio dependency entirely on Windows.
    """
    chunk_duration = 0.2
    chunk_samples = int(sample_rate * chunk_duration)
    chunks = []
    has_spoken = False
    silence_start = None

    print(f"\n[STT - Microphone] Listening (up to {max_duration}s)... Speak your question now.")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        start_time = time.time()
        while time.time() - start_time < max_duration:
            data, _ = stream.read(chunk_samples)
            chunks.append(data)
            energy = np.abs(data).mean()
            if energy > 250:
                has_spoken = True
                silence_start = None
            elif has_spoken:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_timeout:
                    print("[STT] Silence detected, finishing recording.")
                    break

    if not chunks:
        return None

    audio_bytes = np.concatenate(chunks, axis=0).tobytes()
    return sr.AudioData(audio_bytes, sample_rate=sample_rate, sample_width=2)

def transcribe_from_microphone(timeout: int = 8, phrase_time_limit: int = 15) -> Optional[str]:
    """
    Captures live audio from the microphone and converts it to text (STT).
    Uses sounddevice to ensure zero PyAudio dependency on Windows.
    """
    recognizer = get_recognizer()
    audio = None

    # Primary recorder: sounddevice
    try:
        audio = record_audio_via_sounddevice(max_duration=phrase_time_limit)
    except Exception as sd_err:
        # Fallback recorder: sr.Microphone if available
        try:
            with sr.Microphone() as source:
                print("\n[STT] Calibrating microphone for ambient noise... (1 sec)")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                print(f"[STT] Listening (limit {phrase_time_limit}s)... Speak now.")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except Exception:
            print(f"[STT] Microphone device error: {sd_err}")
            return None

    if not audio:
        return None

    try:
        print("[STT] Audio captured, recognizing speech with Google STT...")
        query = recognizer.recognize_google(audio)
        print(f"[STT] Recognized Question: \"{query}\"")
        return query.strip()
    except sr.UnknownValueError:
        print("[STT] Could not understand the audio. Please speak clearly.")
        return None
    except sr.RequestError as e:
        print(f"[STT] Speech recognition service error: {e}")
        return None
    except Exception as e:
        print(f"[STT] Unexpected error during recognition: {e}")
        return None

def transcribe_from_file(audio_path: str) -> Optional[str]:
    """Transcribes speech from a given audio file (WAV, AIFF, FLAC)."""
    file_path = Path(audio_path)
    if not file_path.exists():
        print(f"[STT] Audio file not found: {audio_path}")
        return None

    recognizer = get_recognizer()
    try:
        with sr.AudioFile(str(file_path)) as source:
            audio = recognizer.record(source)
            print(f"[STT] Processing audio file '{file_path.name}'...")
            query = recognizer.recognize_google(audio)
            print(f"[STT] Recognized text from file: \"{query}\"")
            return query.strip()
    except Exception as e:
        print(f"[STT] Failed to transcribe audio file: {e}")
        return None

def get_user_query(prefer_voice: bool = True) -> str:
    """
    High-level query prompt for STT -> LLM.
    If prefer_voice is True, attempts STT first; falls back gracefully to keyboard input.
    """
    if prefer_voice:
        print("\n" + "=" * 55)
        print(" [STT] VOICE INPUT MODE")
        print(" Speak your question into the microphone.")
        print(" (Press Enter without speaking or choose fallback if needed)")
        print("=" * 55)
        spoken_text = transcribe_from_microphone()
        if spoken_text:
            return spoken_text
        print("[STT] Voice input failed or was empty. Switching to text input fallback.")

    print("\n[Input] Type your question below:")
    user_input = input(">> ").strip()
    return user_input


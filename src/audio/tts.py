import os
import re
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional
import pyttsx3
from src import config

def clean_for_speech(text: str) -> str:
    """
    Cleans markdown formatting, links, and code blocks so that the text
    is pronounced naturally without verbalizing symbols like asterisks or hashtags.
    """
    if not text:
        return ""

    # Remove markdown links [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove raw URLs
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    # Remove code blocks and inline backticks
    cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    # Remove bold, italics, headers, bullets, blockquotes
    cleaned = re.sub(r'[*_~#>-]', ' ', cleaned)
    # Remove multiple spaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_tts_engine(rate: int = config.TTS_RATE, volume: float = config.TTS_VOLUME):
    """Initializes and configures the local pyttsx3 engine."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        return engine
    except Exception as e:
        print(f"[TTS Warning] Could not initialize pyttsx3 engine: {e}")
        return None

def speak_text(text: str, save_to_file: Optional[str] = None, blocking: bool = True) -> bool:
    """
    Speaks the provided answer text out loud (TTS).
    Optionally saves the audio to an output audio file (e.g. .wav).
    """
    speech_content = clean_for_speech(text)
    if not speech_content:
        return False

    engine = get_tts_engine()
    if engine is None:
        print("[TTS] Engine unavailable, skipping voice playback.")
        return False

    try:
        if save_to_file:
            out_path = Path(save_to_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            engine.save_to_file(speech_content, str(out_path))

        engine.say(speech_content)
        if blocking:
            engine.runAndWait()
        return True
    except Exception as e:
        print(f"[TTS Error] Playback failed: {e}")
        return False

async def _edge_tts_synthesize(text: str, output_path: str, voice: str = "en-US-GuyNeural"):
    """Internal helper to generate high quality neural audio using edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def synthesize_audio_file(text: str, output_path: str, prefer_neural: bool = True) -> Optional[str]:
    """
    Synthesizes speech to an audio file.
    Runs edge-tts in a separate worker thread to avoid event loop conflicts in Streamlit,
    with an offline pyttsx3 WAV fallback. Returns the actual saved file path, or None.
    """
    speech_content = clean_for_speech(text)
    if not speech_content:
        return None

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try high quality neural edge-tts in a clean background thread
    if prefer_neural:
        try:
            mp3_path = str(out_file.with_suffix(".mp3"))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, _edge_tts_synthesize(speech_content, mp3_path)).result(timeout=10)
            if Path(mp3_path).exists() and Path(mp3_path).stat().st_size > 0:
                return mp3_path
        except Exception as e:
            print(f"[TTS Neural Notice] edge-tts unavailable ({e}), falling back to pyttsx3.")

    # 2. Offline fallback via pyttsx3 (SAPI5 produces WAV files natively on Windows)
    try:
        wav_path = str(out_file.with_suffix(".wav"))
        engine = get_tts_engine()
        if engine:
            engine.save_to_file(speech_content, wav_path)
            engine.runAndWait()
            if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
                return wav_path
    except Exception as e:
        print(f"[TTS Offline Fallback Error]: {e}")

    return None

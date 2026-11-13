import re
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
        print("\n[TTS] Speaking final answer out loud...")
        if save_to_file:
            out_path = Path(save_to_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            engine.save_to_file(speech_content, str(out_path))
            print(f"[TTS] Saved audio file to: {out_path}")

        engine.say(speech_content)
        if blocking:
            engine.runAndWait()
        return True
    except Exception as e:
        print(f"[TTS Error] Playback failed: {e}")
        return False


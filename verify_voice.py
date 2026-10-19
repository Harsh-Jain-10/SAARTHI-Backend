#!/usr/bin/env python3
"""
Saarthi AI — Pre-Flight Voice Hardware & API Verification Script
=================================================================
Validates:
1. Python runtime environment & dependencies
2. Local microphone input driver (PyAudio)
3. Local speaker output driver (PyAudio)
4. .env secrets configuration (DEEPGRAM_API_KEY, GROQ_API_KEY)
5. Live Groq API connectivity & latency (Llama 3.1 8B Instant)
6. Live Deepgram API connectivity & latency (Nova-2 & Aura)
"""

import os
import sys
import time
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# ANSI terminal styling
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")


def check_python_environment() -> bool:
    print_header("1. PYTHON RUNTIME & DEPENDENCIES")
    py_ver = sys.version_info
    print(f"[*] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    if py_ver < (3, 10):
        print(f" {RED}[FAIL]{RESET} Python 3.10 or higher is required.")
        return False
    print(f" {GREEN}[PASS]{RESET} Python version is supported.")

    required_pkgs = [
        ("pipecat", "pipecat-ai"),
        ("pyaudio", "PyAudio"),
        ("onnxruntime", "onnxruntime"),
        ("groq", "groq"),
        ("deepgram", "deepgram-sdk"),
        ("loguru", "loguru"),
    ]

    missing = []
    for mod_name, pkg_name in required_pkgs:
        try:
            __import__(mod_name)
            print(f" {GREEN}[PASS]{RESET} Package installed: {BOLD}{pkg_name}{RESET}")
        except ImportError:
            print(f" {RED}[FAIL]{RESET} Missing package: {BOLD}{pkg_name}{RESET}")
            missing.append(pkg_name)

    if missing:
        print(f"\n{YELLOW}[!] Run: pip install -r requirements.txt{RESET}")
        return False
    return True


def check_audio_hardware(play_tone: bool = True) -> bool:
    print_header("2. LOCAL AUDIO DRIVERS & HARDWARE (PYAUDIO)")
    try:
        import pyaudio
        import numpy as np

        pa = pyaudio.PyAudio()
    except Exception as e:
        print(f" {RED}[FAIL]{RESET} Failed to initialize PyAudio: {e}")
        return False

    success = True
    input_dev = None
    output_dev = None

    # Check Default Microphone
    try:
        input_dev = pa.get_default_input_device_info()
        print(f" {GREEN}[PASS]{RESET} Default Microphone: {BOLD}{input_dev.get('name')}{RESET}")
        print(f"        Channels: {input_dev.get('maxInputChannels')}, Sample Rate: {int(input_dev.get('defaultSampleRate'))} Hz")
    except Exception as e:
        print(f" {RED}[FAIL]{RESET} No default microphone detected: {e}")
        success = False

    # Check Default Speakers
    try:
        output_dev = pa.get_default_output_device_info()
        print(f" {GREEN}[PASS]{RESET} Default Speaker/Headphones: {BOLD}{output_dev.get('name')}{RESET}")
        print(f"        Channels: {output_dev.get('maxOutputChannels')}, Sample Rate: {int(output_dev.get('defaultSampleRate'))} Hz")
    except Exception as e:
        print(f" {RED}[FAIL]{RESET} No default speaker detected: {e}")
        success = False

    # Test Recording 0.5s from Microphone
    if input_dev:
        try:
            print("[*] Testing microphone capture (recording 0.5s)...")
            stream_in = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
            )
            frames = []
            for _ in range(int(16000 / 1024 * 0.5)):
                frames.append(stream_in.read(1024, exception_on_overflow=False))
            stream_in.stop_stream()
            stream_in.close()

            total_bytes = sum(len(f) for f in frames)
            print(f" {GREEN}[PASS]{RESET} Microphone stream active ({total_bytes} bytes captured).")
        except Exception as e:
            print(f" {RED}[FAIL]{RESET} Microphone capture stream failed: {e}")
            success = False

    # Test Speaker Audio Output Tone
    if output_dev and play_tone:
        try:
            print("[*] Testing speaker output (gentle 440 Hz test tone)...")
            stream_out = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True,
            )
            duration = 0.25  # seconds
            t = np.linspace(0, duration, int(16000 * duration), False)
            tone = (np.sin(2 * np.pi * 440 * t) * 800).astype(np.int16)
            stream_out.write(tone.tobytes())
            stream_out.stop_stream()
            stream_out.close()
            print(f" {GREEN}[PASS]{RESET} Speaker playback stream validated successfully.")
        except Exception as e:
            print(f" {YELLOW}[WARN]{RESET} Speaker playback test failed: {e}")

    pa.terminate()
    return success


def check_api_credentials() -> tuple[bool, str | None, str | None]:
    print_header("3. ENVIRONMENT SECRETS (.ENV)")
    if not ENV_FILE.exists():
        print(f" {RED}[FAIL]{RESET} .env file not found at {ENV_FILE}")
        return False, None, None

    print(f" {GREEN}[PASS]{RESET} Found .env file at: {ENV_FILE.name}")

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    has_dg = bool(deepgram_key)
    has_groq = bool(groq_key)

    if has_dg:
        masked_dg = deepgram_key[:4] + "..." + deepgram_key[-4:] if len(deepgram_key) > 8 else "***"
        print(f" {GREEN}[PASS]{RESET} DEEPGRAM_API_KEY configured ({masked_dg})")
    else:
        print(f" {YELLOW}[WAIT]{RESET} DEEPGRAM_API_KEY is empty in .env")

    if has_groq:
        masked_groq = groq_key[:4] + "..." + groq_key[-4:] if len(groq_key) > 8 else "***"
        print(f" {GREEN}[PASS]{RESET} GROQ_API_KEY configured ({masked_groq})")
    else:
        print(f" {YELLOW}[WAIT]{RESET} GROQ_API_KEY is empty in .env")

    return (has_dg and has_groq), deepgram_key or None, groq_key or None


def check_api_connectivity(deepgram_key: str | None, groq_key: str | None) -> bool:
    print_header("4. CLOUD API CONNECTIVITY & AUTHENTICATION")
    all_ok = True

    # 1. Groq Connectivity
    if groq_key:
        print("[*] Testing Groq API connection...")
        try:
            from groq import Groq

            t0 = time.time()
            client = Groq(api_key=groq_key)
            models = client.models.list()
            latency_ms = int((time.time() - t0) * 1000)

            available_ids = [m.id for m in models.data]
            target_model = "llama-3.1-8b-instant"
            if target_model in available_ids:
                print(f" {GREEN}[PASS]{RESET} Groq API authenticated ({latency_ms}ms latency). Model '{target_model}' is available.")
            else:
                print(f" {YELLOW}[WARN]{RESET} Groq API authenticated ({latency_ms}ms), but '{target_model}' not found in model list.")
        except Exception as e:
            print(f" {RED}[FAIL]{RESET} Groq API authentication failed: {e}")
            all_ok = False
    else:
        print(f" {YELLOW}[SKIP]{RESET} Groq API test skipped (GROQ_API_KEY missing in .env)")
        all_ok = False

    # 2. Deepgram Connectivity
    if deepgram_key:
        print("[*] Testing Deepgram API connection...")
        try:
            import httpx

            t0 = time.time()
            resp = httpx.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {deepgram_key}"},
                timeout=8.0,
            )
            latency_ms = int((time.time() - t0) * 1000)

            if resp.status_code == 200:
                print(f" {GREEN}[PASS]{RESET} Deepgram API authenticated ({latency_ms}ms latency).")
            else:
                print(f" {RED}[FAIL]{RESET} Deepgram API returned status {resp.status_code}: {resp.text}")
                all_ok = False
        except Exception as e:
            print(f" {RED}[FAIL]{RESET} Deepgram API connection error: {e}")
            all_ok = False
    else:
        print(f" {YELLOW}[SKIP]{RESET} Deepgram API test skipped (DEEPGRAM_API_KEY missing in .env)")
        all_ok = False

    return all_ok


def main():
    print(f"\n{BOLD}==================================================================={RESET}")
    print(f"{BOLD}  SAARTHI AI — PRE-FLIGHT VOICE PIPELINE VERIFICATION SUITE       {RESET}")
    print(f"{BOLD}==================================================================={RESET}")

    env_ok = check_python_environment()
    audio_ok = check_audio_hardware(play_tone=True)
    keys_configured, dg_key, groq_key = check_api_credentials()
    api_ok = check_api_connectivity(dg_key, groq_key) if keys_configured else False

    print_header("SYSTEM READINESS SUMMARY")
    print(f" 1. Dependencies & Environment: {'[OK]' if env_ok else '[FAIL]'}")
    print(f" 2. Audio Hardware & Drivers:   {'[OK]' if audio_ok else '[FAIL]'}")
    print(f" 3. API Keys Configured:        {'[OK]' if keys_configured else '[WAITING FOR KEYS]'}")
    print(f" 4. API Live Connectivity:      {'[OK]' if api_ok else '[PENDING KEYS / FAILED]'}")

    if env_ok and audio_ok and keys_configured and api_ok:
        print(f"\n{GREEN}{BOLD}[READY] All pre-flight checks passed! You can run:{RESET}")
        print(f"        {CYAN}python src/voice_agent.py{RESET}\n")
        sys.exit(0)
    elif not keys_configured:
        print(f"\n{YELLOW}{BOLD}[ACTION REQUIRED] Add your API keys to .env:{RESET}")
        print(f"        Edit {BOLD}.env{RESET} and provide {BOLD}DEEPGRAM_API_KEY{RESET} and {BOLD}GROQ_API_KEY{RESET}")
        print(f"        Then re-run: {CYAN}python verify_voice.py{RESET}\n")
        sys.exit(2)
    else:
        print(f"\n{RED}{BOLD}[ERROR] Some pre-flight checks failed. Review details above.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

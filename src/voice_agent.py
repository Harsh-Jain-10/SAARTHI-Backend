#!/usr/bin/env python3
"""
Saarthi AI — Local Real-Time Voice Agent Pipeline
==================================================
A terminal-based voice pipeline with sub-250ms barge-in interruption.

Pipeline Architecture:
    System Microphone (PyAudio)
       -> Silero VAD (Speech Activity Detection & Sub-250ms Barge-in)
       -> Deepgram Nova-2 STT (WebSocket Streaming Transcription)
       -> Groq Qwen 3.8 27B (Ultra-fast LLM Response Generation via Groq)
       -> Deepgram Aura TTS (Aura-Asteria Natural Voice Synthesis)
       -> System Speaker (PyAudio Output)

Logged Lifecycle Events:
    [LISTENING]         - Waiting for user speech
    [USER TRANSCRIPT]   - Real-time speech transcription from STT
    [BOT THINKING]      - Turn completion & LLM inference triggered
    [BOT SPEAKING]      - Audio playback started through speakers
    [BARGE-IN DETECTED] - User spoke over bot speech; playback interrupted
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Pipecat Core Components
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

# -----------------------------------------------------------------------------
# Configuration & Environment
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# System prompt configured strictly for short, conversational voice interaction
SYSTEM_PROMPT = (
    "You are Saarthi, a friendly, ultra-fast campus voice AI assistant. "
    "Your responses are spoken aloud via speech synthesis in real time. "
    "Strict rules for all responses:\n"
    "1. Keep answers concise, natural, and direct (strictly 1 to 2 short sentences max).\n"
    "2. Never use bullet points, numbered lists, markdown asterisks, emojis, URLs, or symbols that cannot be pronounced clearly.\n"
    "3. Be warm, helpful, and conversational like a real human voice assistant."
)


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
def configure_logger():
    """Configures clean, informative terminal logging.

    Uses a patcher to ensure every log record has a default 'tag' key so
    that Pipecat's internal loggers (which don't bind 'tag') don't trigger
    a KeyError in our custom loguru format string.
    """
    logger.remove()

    # Patch: inject a default empty tag on every record that lacks one.
    # This prevents KeyError when Pipecat internals emit log records without
    # the 'tag' extra key that our format string expects.
    logger.configure(
        patcher=lambda record: record["extra"].setdefault("tag", "")
    )

    log_format = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{extra[tag]}</cyan> {message}"
    )
    logger.add(sys.stderr, format=log_format, level="INFO")


# -----------------------------------------------------------------------------
# Lifecycle Event Coordination & Frame Observers
# -----------------------------------------------------------------------------
class VoiceStateCoordinator:
    """Tracks speaking lifecycle state to accurately log transitions and barge-in events."""

    def __init__(self):
        self.is_bot_speaking = False

    def on_user_started(self):
        """Called when user starts speaking."""
        if self.is_bot_speaking:
            self.is_bot_speaking = False
            logger.bind(tag="[BARGE-IN DETECTED]").warning(
                "User spoke while assistant was speaking! Immediate interruption triggered."
            )

    def on_transcript(self, text: str):
        """Called when a finalized transcript is received from Deepgram STT."""
        logger.bind(tag="[USER TRANSCRIPT]").info(f'"{text}"')

    def on_user_stopped(self):
        """Called when user stops speaking (end-of-turn detected)."""
        logger.bind(tag="[BOT THINKING]").info(
            "Generating concise voice response via Groq qwen3.8-27b..."
        )

    def on_bot_started(self):
        """Called when TTS begins streaming audio to speakers."""
        if not self.is_bot_speaking:
            self.is_bot_speaking = True
            logger.bind(tag="[BOT SPEAKING]").info(
                "Assistant started speaking response through speakers..."
            )

    def on_bot_stopped(self):
        """Called when TTS audio playback finishes."""
        if self.is_bot_speaking:
            self.is_bot_speaking = False
            logger.bind(tag="[LISTENING]").info(
                "Ready for user input. Speak into your microphone..."
            )

    def on_interruption(self):
        """Called when an InterruptionFrame is pushed through the pipeline."""
        if self.is_bot_speaking:
            self.is_bot_speaking = False
            logger.bind(tag="[BARGE-IN DETECTED]").warning(
                "Pipeline interruption signal received. Halting audio output."
            )


class UserStreamObserver(FrameProcessor):
    """Monitors incoming user speech events, transcripts, and end-of-turn."""

    def __init__(self, coordinator: VoiceStateCoordinator):
        super().__init__()
        self._coord = coordinator

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._coord.on_user_started()

        elif isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                self._coord.on_transcript(frame.text.strip())

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._coord.on_user_stopped()

        elif isinstance(frame, InterruptionFrame):
            self._coord.on_interruption()

        await self.push_frame(frame, direction)


class BotStreamObserver(FrameProcessor):
    """Monitors TTS playback states and barge-in interruptions on output stream."""

    def __init__(self, coordinator: VoiceStateCoordinator):
        super().__init__()
        self._coord = coordinator

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (BotStartedSpeakingFrame, TTSStartedFrame)):
            self._coord.on_bot_started()

        elif isinstance(frame, (BotStoppedSpeakingFrame, TTSStoppedFrame)):
            self._coord.on_bot_stopped()

        elif isinstance(frame, InterruptionFrame):
            self._coord.on_interruption()

        await self.push_frame(frame, direction)


# -----------------------------------------------------------------------------
# Pipeline Builder
# -----------------------------------------------------------------------------
def build_voice_pipeline(
    deepgram_key: str,
    groq_key: str,
    coordinator: VoiceStateCoordinator,
) -> tuple[Pipeline, LLMContext]:
    """
    Constructs the modular Pipecat voice pipeline:
    Audio In -> Deepgram STT -> User Aggregator (Silero VAD) -> Observer
             -> Groq LLM -> Deepgram TTS -> Observer -> Audio Out -> Asst Aggregator
    """
    # 1. Local Audio Transport (Microphone & Speaker via PyAudio)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # 2. Silero VAD Analyzer for sub-250ms barge-in interruption
    # stop_secs=0.2 is Pipecat's recommended default for accurate turn detection
    # with the built-in TurnAnalyzer (Smart Turn v3 ONNX model).
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,  # 200ms speech onset (sub-250ms barge-in)
            stop_secs=0.2,   # 200ms silence threshold — Pipecat recommended default
        )
    )

    # 3. Deepgram Nova-2 STT via WebSocket Streaming
    stt = DeepgramSTTService(
        api_key=deepgram_key,
        settings=DeepgramSTTService.Settings(
            model="nova-2",
            language="en",
        ),
    )

    # 4. Groq LLM — qwen/qwen3.8-27b
    # Note: llama-3.1-8b-instant has been deprecated on this Groq account.
    # qwen/qwen3.8-27b is the fastest confirmed-working chat model available.
    # System prompt is passed via system_instruction (replaces deprecated LLMContext pattern).
    llm = GroqLLMService(
        api_key=groq_key,
        settings=GroqLLMService.Settings(
            model="qwen/qwen3.8-27b",
            system_instruction=SYSTEM_PROMPT,
            temperature=0.6,
        ),
    )

    # 5. Deepgram Aura TTS (Aura-Asteria English voice)
    tts = DeepgramTTSService(
        api_key=deepgram_key,
        settings=DeepgramTTSService.Settings(
            voice="aura-asteria-en",
        ),
    )

    # 6. Conversation Context & Turn Aggregators
    # System prompt is now managed via system_instruction in GroqLLMService.Settings
    # (the LLMContext(messages=[{"role": "system", ...}]) pattern is deprecated since 1.9.0).
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer),
    )

    # 7. Event Observers
    user_observer = UserStreamObserver(coordinator)
    bot_observer = BotStreamObserver(coordinator)

    # 8. Assemble Pipeline
    pipeline = Pipeline(
        [
            transport.input(),      # System Microphone input
            stt,                    # Deepgram STT streaming
            user_aggregator,        # Turn aggregation & VAD
            user_observer,          # Log [USER TRANSCRIPT], [BOT THINKING]
            llm,                    # Groq qwen3.8-27b LLM
            tts,                    # Deepgram Aura TTS
            bot_observer,           # Log [BOT SPEAKING], [LISTENING], [BARGE-IN DETECTED]
            transport.output(),     # System Speaker output
            assistant_aggregator,   # Assistant context tracking
        ]
    )

    return pipeline, context


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
async def run_voice_agent():
    configure_logger()

    print("\n" + "=" * 68)
    print("        SAARTHI AI -- REAL-TIME VOICE PIPELINE (PIPECAT)")
    print("   STT: Deepgram Nova-2 | LLM: Groq qwen3.8-27b | TTS: Deepgram Aura")
    print("   VAD: Silero ONNX (Sub-250ms Barge-In) | Hardware: Local PyAudio")
    print("=" * 68 + "\n")

    # Validate API Keys
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not deepgram_key or not groq_key:
        logger.bind(tag="[CONFIG ERROR]").error(
            "Missing required API credentials in .env file!"
        )
        print("\nPlease configure both keys in .env:")
        if not deepgram_key:
            print("  - DEEPGRAM_API_KEY (from https://console.deepgram.com/)")
        if not groq_key:
            print("  - GROQ_API_KEY (from https://console.groq.com/keys)")
        print("\nTip: Run 'python verify_voice.py' to test your environment and credentials.\n")
        sys.exit(1)

    coordinator = VoiceStateCoordinator()
    pipeline, context = build_voice_pipeline(deepgram_key, groq_key, coordinator)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)

    # Trigger initial assistant introduction
    context.add_message(
        {
            "role": "user",
            "content": "Please greet the user warmly in one short sentence and introduce yourself as Saarthi.",
        }
    )
    await worker.queue_frames([LLMRunFrame()])

    logger.bind(tag="[LISTENING]").info(
        "Saarthi Voice Agent initialized. Speak into your microphone anytime (barge-in enabled)..."
    )

    try:
        await runner.run()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        logger.bind(tag="[SHUTDOWN]").info("Received keyboard interrupt. Stopping voice agent...")
    finally:
        logger.bind(tag="[SHUTDOWN]").info("Voice agent pipeline terminated cleanly.")


def main():
    try:
        asyncio.run(run_voice_agent())
    except KeyboardInterrupt:
        print("\n[+] Exited successfully.")


if __name__ == "__main__":
    main()

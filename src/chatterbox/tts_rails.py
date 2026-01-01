"""
TTSRails - Parallel Streaming TTS with Voice Cloning

A high-level API for managing multiple concurrent TTS streams ("rails"),
each with its own voice, running in parallel on a shared GPU.

Example:
    tts = TTSRails(device="cuda")
    tts.register_voice("narrator", "voices/narrator.wav")

    rail = tts.rail("main", voice="narrator")
    rail.push("Hello world!")

    while chunk := rail.read(timeout=1.0):
        play_audio(chunk)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue, Empty
from typing import Dict, Optional, Callable

import torch

from .tts_turbo import ChatterboxTurboTTS, Conditionals, normalize_text

logger = logging.getLogger(__name__)

MAX_RAILS = 6


class RailState(Enum):
    """Rail state machine states."""
    IDLE = auto()
    GENERATING = auto()
    STOPPED = auto()


@dataclass
class RailConfig:
    """Configuration for a rail."""
    voice: str
    exaggeration: float = 0.5
    temperature: float = 0.3
    top_k: int = 1000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    # Buffering: wait for sentence end OR this many ms of silence before generating
    flush_timeout_ms: float = 300.0


class Rail:
    """
    A single TTS streaming channel.

    Push text in, read audio out. Text pushed during generation
    is appended to the current utterance.
    """

    def __init__(
        self,
        name: str,
        tts: 'TTSRails',
        config: RailConfig,
    ):
        self.name = name
        self.tts = tts
        self.config = config
        self.conds: Optional[Conditionals] = None

        # Queues
        self.text_queue: Queue[str] = Queue()
        self.audio_queue: Queue[torch.Tensor] = Queue()

        # State
        self.state = RailState.IDLE
        self.pending_text = ""
        self.last_push_time = 0.0
        self._lock = threading.Lock()

        # Control flags
        self.interrupt_flag = threading.Event()
        self.running = True

        # Worker thread
        self.worker = threading.Thread(target=self._run, daemon=True, name=f"rail-{name}")
        self.worker.start()

        logger.info(f"Rail '{name}' created with voice '{config.voice}'")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def push(self, text: str):
        """
        Push text to be spoken. Non-blocking.

        Text is buffered until a sentence boundary or timeout, then
        generated as a single utterance for optimal quality ramping.
        """
        if not text:
            return
        self.text_queue.put(text)
        self.last_push_time = time.time()

    def read(self, timeout: Optional[float] = None) -> Optional[torch.Tensor]:
        """
        Read next audio chunk. Blocks until available or timeout.

        Args:
            timeout: Max seconds to wait. None = wait forever.

        Returns:
            Audio tensor or None if timeout/stopped.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except Empty:
            return None

    def read_nowait(self) -> Optional[torch.Tensor]:
        """Read next audio chunk without blocking. Returns None if empty."""
        try:
            return self.audio_queue.get_nowait()
        except Empty:
            return None

    def interrupt(self):
        """
        Immediately stop generation and clear all queues.

        The rail returns to IDLE and is ready for new input.
        """
        logger.info(f"Rail '{self.name}' interrupted")
        self.interrupt_flag.set()

        # Clear queues
        self._clear_queue(self.text_queue)
        self._clear_queue(self.audio_queue)

        with self._lock:
            self.pending_text = ""

        # Brief wait for worker to notice
        time.sleep(0.01)
        self.interrupt_flag.clear()

    def flush(self):
        """Force generation of any buffered text immediately."""
        # Set last_push_time to 0 to trigger timeout condition
        self.last_push_time = 0

    def close(self):
        """Shutdown this rail."""
        self.running = False
        self.interrupt()
        self.worker.join(timeout=1.0)
        logger.info(f"Rail '{self.name}' closed")

    @property
    def is_idle(self) -> bool:
        """True if rail is not generating and has no pending text."""
        return (
            self.state == RailState.IDLE
            and self.text_queue.empty()
            and not self.pending_text
        )

    @property
    def has_audio(self) -> bool:
        """True if there's audio ready to read."""
        return not self.audio_queue.empty()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _run(self):
        """Worker thread main loop."""
        while self.running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Rail '{self.name}' error: {e}")
                time.sleep(0.1)

    def _tick(self):
        """Single iteration of the worker loop."""
        # Collect any pending text
        self._collect_text()

        with self._lock:
            text = self.pending_text
            if not text:
                time.sleep(0.01)
                return

            # Check if we should generate now:
            # 1. Sentence boundary reached (. ! ?)
            # 2. OR timeout since last push
            sentence_end = text.rstrip().endswith(('.', '!', '?'))
            time_since_push = (time.time() - self.last_push_time) * 1000  # ms
            timeout_reached = time_since_push >= self.config.flush_timeout_ms

            if sentence_end or timeout_reached:
                self.pending_text = ""
            else:
                # Keep buffering
                time.sleep(0.01)
                return

        self._generate(text)

    def _collect_text(self):
        """Drain text queue into pending buffer."""
        while True:
            try:
                text = self.text_queue.get_nowait()
                with self._lock:
                    self.pending_text += text
            except Empty:
                break

    def _generate(self, text: str):
        """
        Run TTS generation with GPU lock.

        Note: GPU lock is held for entire utterance because:
        1. Generator holds KV cache state between yields
        2. model.conds must not be changed mid-generation

        For parallel rails, they will queue and run sequentially.
        Each rail's sentences are buffered, so the interleaving happens
        at sentence boundaries rather than chunk boundaries.
        """
        self.state = RailState.GENERATING

        # Load voice conditionals
        if self.config.voice in self.tts.voices:
            self.conds = self.tts.voices[self.config.voice]
        else:
            logger.warning(f"Voice '{self.config.voice}' not found, using default")
            self.conds = None

        try:
            with self.tts.gpu_lock:
                if self.interrupt_flag.is_set():
                    return

                if self.conds is not None:
                    self.tts.model.conds = self.conds

                for chunk, _ in self.tts.model.generate_stream(
                    text=text,
                    audio_prompt_path=None,
                    exaggeration=self.config.exaggeration,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                ):
                    if self.interrupt_flag.is_set():
                        logger.debug(f"Rail '{self.name}' generation interrupted")
                        return

                    self.audio_queue.put(chunk)

        finally:
            self.state = RailState.IDLE

    @staticmethod
    def _clear_queue(q: Queue):
        """Clear all items from a queue."""
        while True:
            try:
                q.get_nowait()
            except Empty:
                break


class TTSRails:
    """
    TTS orchestrator managing multiple parallel rails.

    Compiles models once, keeps them hot, and manages voice registration
    and rail allocation.

    Example:
        tts = TTSRails(device="cuda")
        tts.register_voice("narrator", "voices/deep_male.wav")

        narrator = tts.rail("narrator", voice="narrator")
        narrator.push("Hello, world!")

        while chunk := narrator.read(timeout=1.0):
            play(chunk)
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize TTSRails.

        Args:
            device: Device for inference ("cuda", "cpu", "mps")
        """
        self.device = device
        self.voices: Dict[str, Conditionals] = {}
        self.rails: Dict[str, Rail] = {}
        self.gpu_lock = threading.Lock()

        logger.info(f"Loading TTS model on {device}...")
        self.model = ChatterboxTurboTTS.from_pretrained(device=device)

        logger.info("Compiling model...")
        self.model.compile()

        # Warmup
        logger.info("Warming up...")
        for _ in self.model.generate_stream("Hello."):
            pass

        logger.info("TTSRails ready")

    # -------------------------------------------------------------------------
    # Voice Management
    # -------------------------------------------------------------------------

    def register_voice(self, name: str, wav_path: str, exaggeration: float = 0.5):
        """
        Register a voice from a reference audio file.

        Args:
            name: Voice identifier
            wav_path: Path to reference audio (must be >5 seconds)
            exaggeration: Emotion exaggeration factor
        """
        logger.info(f"Registering voice '{name}' from {wav_path}")

        with self.gpu_lock:
            self.model.prepare_conditionals(wav_path, exaggeration=exaggeration)
            self.voices[name] = self.model.conds
            self.model.conds = None

        logger.info(f"Voice '{name}' registered")

    def list_voices(self) -> list[str]:
        """List registered voice names."""
        return list(self.voices.keys())

    # -------------------------------------------------------------------------
    # Rail Management
    # -------------------------------------------------------------------------

    def rail(
        self,
        name: str,
        voice: str,
        exaggeration: float = 0.5,
        temperature: float = 0.3,
    ) -> Rail:
        """
        Allocate a new rail or return existing one.

        Args:
            name: Rail identifier
            voice: Voice name (must be registered)
            exaggeration: Emotion exaggeration
            temperature: Sampling temperature

        Returns:
            Rail instance
        """
        if name in self.rails:
            return self.rails[name]

        if len(self.rails) >= MAX_RAILS:
            raise RuntimeError(f"Maximum {MAX_RAILS} rails allowed")

        if voice not in self.voices:
            raise ValueError(f"Voice '{voice}' not registered. Available: {self.list_voices()}")

        config = RailConfig(
            voice=voice,
            exaggeration=exaggeration,
            temperature=temperature,
        )

        rail = Rail(name, self, config)
        self.rails[name] = rail
        return rail

    def get_rail(self, name: str) -> Optional[Rail]:
        """Get a rail by name, or None if not found."""
        return self.rails.get(name)

    def list_rails(self) -> list[str]:
        """List allocated rail names."""
        return list(self.rails.keys())

    def close_rail(self, name: str):
        """Close and remove a rail."""
        if name in self.rails:
            self.rails[name].close()
            del self.rails[name]

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def interrupt_all(self):
        """Interrupt all rails."""
        for rail in self.rails.values():
            rail.interrupt()

    def shutdown(self):
        """Shutdown all rails and cleanup."""
        logger.info("Shutting down TTSRails...")
        for name in list(self.rails.keys()):
            self.close_rail(name)
        logger.info("TTSRails shutdown complete")

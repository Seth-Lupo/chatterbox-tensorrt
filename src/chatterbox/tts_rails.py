"""
TTSRails - Parallel Streaming TTS with Voice Cloning

A high-level API for managing multiple concurrent TTS streams ("rails"),
each with its own voice, running in TRUE PARALLEL on a shared GPU.

Example:
    tts = TTSRails(device="cuda")
    tts.register_voice("narrator", "voices/narrator.wav")

    rail = tts.rail("main", voice="narrator")
    rail.push("Hello world!")

    while chunk := rail.read(timeout=1.0):
        play_audio(chunk)

Configuration:
    tts = TTSRails(
        device="cuda",
        max_rails=6,           # Max concurrent streams
        batch_wait_ms=50,      # Time to collect rails before batching
    )
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue, Empty
from typing import Dict, Optional, List

import torch

from .tts_turbo import (
    ChatterboxTurboTTS,
    Conditionals,
    normalize_text,
    DEFAULT_RAMP_SCHEDULE,
    RampSchedule,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_RAILS = 6
DEFAULT_BATCH_WAIT_MS = 50


class RailState(Enum):
    """Rail state machine states."""
    IDLE = auto()
    PENDING = auto()
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
    flush_timeout_ms: float = 300.0


@dataclass
class PendingUtterance:
    """An utterance ready for batched generation."""
    rail_name: str
    text: str
    conds: Conditionals
    config: RailConfig


class Rail:
    """
    A single TTS streaming channel.

    Push text in, read audio out. Text is buffered until sentence
    boundaries, then submitted to the BatchCoordinator for parallel
    generation with other rails.

    Properties:
        is_idle: True if no pending text and not generating
        has_audio: True if audio chunks are ready to read

    Methods:
        push(text): Add text to be spoken (non-blocking)
        read(timeout): Get next audio chunk (blocking)
        read_nowait(): Get next audio chunk (non-blocking)
        interrupt(): Stop generation and clear queues
        flush(): Force generation of buffered text
        close(): Shutdown this rail
    """

    def __init__(self, name: str, tts: 'TTSRails', config: RailConfig):
        self.name = name
        self.tts = tts
        self.config = config
        self.conds: Optional[Conditionals] = None

        self.text_queue: Queue[str] = Queue()
        self.audio_queue: Queue[torch.Tensor] = Queue()

        self.state = RailState.IDLE
        self.pending_text = ""
        self.last_push_time = 0.0
        self._lock = threading.Lock()

        self.interrupt_flag = threading.Event()
        self.running = True

        self.worker = threading.Thread(target=self._run, daemon=True, name=f"rail-{name}")
        self.worker.start()

        logger.debug(f"Rail '{name}' created with voice '{config.voice}'")

    def push(self, text: str):
        """Push text to be spoken. Non-blocking."""
        if not text:
            return
        self.text_queue.put(text)
        self.last_push_time = time.time()

    def read(self, timeout: Optional[float] = None) -> Optional[torch.Tensor]:
        """Read next audio chunk. Blocks until available or timeout."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except Empty:
            return None

    def read_nowait(self) -> Optional[torch.Tensor]:
        """Read next audio chunk without blocking."""
        try:
            return self.audio_queue.get_nowait()
        except Empty:
            return None

    def interrupt(self):
        """Stop generation and clear all queues."""
        logger.debug(f"Rail '{self.name}' interrupted")
        self.interrupt_flag.set()
        self._clear_queue(self.text_queue)
        self._clear_queue(self.audio_queue)
        with self._lock:
            self.pending_text = ""
        time.sleep(0.01)
        self.interrupt_flag.clear()
        self.state = RailState.IDLE

    def flush(self):
        """Force generation of any buffered text immediately."""
        self.last_push_time = 0

    def close(self):
        """Shutdown this rail."""
        self.running = False
        self.interrupt()
        self.worker.join(timeout=1.0)
        logger.debug(f"Rail '{self.name}' closed")

    @property
    def is_idle(self) -> bool:
        """True if rail is not generating and has no pending text."""
        return (
            self.state == RailState.IDLE
            and self.text_queue.empty()
            and not self.pending_text
            and self.audio_queue.empty()
        )

    @property
    def has_audio(self) -> bool:
        """True if there's audio ready to read."""
        return not self.audio_queue.empty()

    # Internal methods

    def _run(self):
        while self.running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Rail '{self.name}' error: {e}")
                time.sleep(0.1)

    def _tick(self):
        self._collect_text()

        with self._lock:
            text = self.pending_text
            if not text:
                time.sleep(0.01)
                return

            sentence_end = text.rstrip().endswith(('.', '!', '?'))
            time_since_push = (time.time() - self.last_push_time) * 1000
            timeout_reached = time_since_push >= self.config.flush_timeout_ms

            if sentence_end or timeout_reached:
                self.pending_text = ""
            else:
                time.sleep(0.01)
                return

        self._submit_for_generation(text)

    def _collect_text(self):
        while True:
            try:
                text = self.text_queue.get_nowait()
                with self._lock:
                    self.pending_text += text
            except Empty:
                break

    def _submit_for_generation(self, text: str):
        conds = self.tts.voices.get(self.config.voice, self.tts.model.conds)
        utterance = PendingUtterance(
            rail_name=self.name,
            text=text,
            conds=conds,
            config=self.config,
        )
        self.state = RailState.PENDING
        self.tts.coordinator.submit(utterance)

    def _receive_audio(self, chunk: torch.Tensor):
        if not self.interrupt_flag.is_set():
            self.audio_queue.put(chunk)

    def _generation_complete(self):
        self.state = RailState.IDLE

    @staticmethod
    def _clear_queue(q: Queue):
        while True:
            try:
                q.get_nowait()
            except Empty:
                break


class BatchCoordinator:
    """Coordinates batched generation across multiple rails."""

    def __init__(self, tts: 'TTSRails', batch_wait_ms: float, max_batch_size: int):
        self.tts = tts
        self.pending: Queue[PendingUtterance] = Queue()
        self.running = True
        self.batch_wait_ms = batch_wait_ms
        self.max_batch_size = max_batch_size

        self.worker = threading.Thread(target=self._run, daemon=True, name="batch-coordinator")
        self.worker.start()

    def submit(self, utterance: PendingUtterance):
        self.pending.put(utterance)

    def shutdown(self):
        self.running = False
        self.worker.join(timeout=2.0)

    def _run(self):
        while self.running:
            try:
                self._process_batch()
            except Exception as e:
                logger.error(f"BatchCoordinator error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _process_batch(self):
        try:
            first = self.pending.get(timeout=0.1)
        except Empty:
            return

        batch = [first]
        deadline = time.time() + self.batch_wait_ms / 1000

        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                utterance = self.pending.get(timeout=0.001)
                batch.append(utterance)
            except Empty:
                break

        self._generate_batch(batch)

    def _generate_batch(self, batch: List[PendingUtterance]):
        if not batch:
            return

        for utterance in batch:
            rail = self.tts.rails.get(utterance.rail_name)
            if rail:
                rail.state = RailState.GENERATING

        texts = [u.text for u in batch]
        conds_list = [u.conds for u in batch]
        config = batch[0].config

        logger.info(f"Batch[{len(batch)}]: {[u.rail_name for u in batch]}")

        try:
            for chunk_results in self.tts.model.generate_stream_batched(
                texts=texts,
                conds_list=conds_list,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                ramp_schedule=self.tts.ramp_schedule,
            ):
                for result in chunk_results:
                    utterance = batch[result.seq_id]
                    rail = self.tts.rails.get(utterance.rail_name)
                    if rail and not rail.interrupt_flag.is_set():
                        rail._receive_audio(result.audio)
        finally:
            for utterance in batch:
                rail = self.tts.rails.get(utterance.rail_name)
                if rail:
                    rail._generation_complete()


class TTSRails:
    """
    Parallel TTS orchestrator with batched generation.

    Manages multiple concurrent TTS streams ("rails"), each with its own
    voice. Rails generate audio in TRUE PARALLEL through batched inference.

    Args:
        device: Compute device ("cuda", "cpu", "mps")
        max_rails: Maximum concurrent rails (default: 6)
        batch_wait_ms: Time to wait for batching multiple rails (default: 50)
        ramp_schedule: Chunk ramp schedule [(tokens, context, cfm_steps), ...]

    Example:
        tts = TTSRails(device="cuda", max_rails=8)
        tts.register_voice("narrator", "voice.wav")

        rail = tts.rail("main", voice="narrator")
        rail.push("Hello world!")

        while chunk := rail.read(timeout=1.0):
            play(chunk)

        tts.shutdown()

    Custom ramp schedule:
        tts = TTSRails(
            device="cuda",
            ramp_schedule=[
                (4, 0, 1),    # Fast first chunk
                (16, 8, 4),   # Medium quality
                (32, 32, 8),  # Full quality
            ]
        )
    """

    def __init__(
        self,
        device: str = "cuda",
        max_rails: int = DEFAULT_MAX_RAILS,
        batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS,
        ramp_schedule: Optional[RampSchedule] = None,
    ):
        self.device = device
        self.max_rails = max_rails
        self.batch_wait_ms = batch_wait_ms
        self.ramp_schedule = ramp_schedule if ramp_schedule is not None else DEFAULT_RAMP_SCHEDULE

        self.voices: Dict[str, Conditionals] = {}
        self.rails: Dict[str, Rail] = {}
        self.gpu_lock = threading.Lock()

        logger.info(f"Loading TTS model on {device}...")
        self.model = ChatterboxTurboTTS.from_pretrained(device=device)

        logger.info("Compiling model...")
        self.model.compile()

        logger.info("Warming up...")
        for _ in self.model.generate_stream("Hello."):
            pass

        self.coordinator = BatchCoordinator(self, batch_wait_ms, max_rails)

        logger.info(f"TTSRails ready (max_rails={max_rails}, batch_wait={batch_wait_ms}ms)")

    # Voice Management

    def register_voice(self, name: str, wav_path: str, exaggeration: float = 0.5):
        """
        Register a voice from a reference audio file.

        Args:
            name: Voice identifier
            wav_path: Path to reference audio (>5 seconds recommended)
            exaggeration: Emotion exaggeration factor (0.0-1.0)
        """
        logger.info(f"Registering voice '{name}'")
        with self.gpu_lock:
            self.model.prepare_conditionals(wav_path, exaggeration=exaggeration)
            self.voices[name] = self.model.conds
            self.model.conds = None

    def list_voices(self) -> List[str]:
        """List registered voice names."""
        return list(self.voices.keys())

    # Rail Management

    def rail(
        self,
        name: str,
        voice: str,
        exaggeration: float = 0.5,
        temperature: float = 0.3,
        flush_timeout_ms: float = 300.0,
    ) -> Rail:
        """
        Create or get a rail.

        Args:
            name: Rail identifier
            voice: Registered voice name
            exaggeration: Emotion exaggeration (0.0-1.0)
            temperature: Sampling temperature (lower = more deterministic)
            flush_timeout_ms: Max time to buffer text before generating

        Returns:
            Rail instance
        """
        if name in self.rails:
            return self.rails[name]

        if len(self.rails) >= self.max_rails:
            raise RuntimeError(f"Maximum {self.max_rails} rails allowed")

        if voice not in self.voices:
            raise ValueError(f"Voice '{voice}' not registered. Available: {self.list_voices()}")

        config = RailConfig(
            voice=voice,
            exaggeration=exaggeration,
            temperature=temperature,
            flush_timeout_ms=flush_timeout_ms,
        )

        rail = Rail(name, self, config)
        self.rails[name] = rail
        return rail

    def get_rail(self, name: str) -> Optional[Rail]:
        """Get a rail by name."""
        return self.rails.get(name)

    def list_rails(self) -> List[str]:
        """List active rail names."""
        return list(self.rails.keys())

    def close_rail(self, name: str):
        """Close and remove a rail."""
        if name in self.rails:
            self.rails[name].close()
            del self.rails[name]

    # Lifecycle

    def interrupt_all(self):
        """Interrupt all rails."""
        for rail in self.rails.values():
            rail.interrupt()

    def shutdown(self):
        """Shutdown all rails and cleanup."""
        logger.info("Shutting down TTSRails...")
        self.coordinator.shutdown()
        for name in list(self.rails.keys()):
            self.close_rail(name)
        logger.info("TTSRails shutdown complete")

    # Properties

    @property
    def num_rails(self) -> int:
        """Number of active rails."""
        return len(self.rails)

    @property
    def num_voices(self) -> int:
        """Number of registered voices."""
        return len(self.voices)

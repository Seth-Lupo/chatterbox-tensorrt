"""
TTSRails - Parallel Streaming TTS with Voice Cloning

A high-level API for managing multiple concurrent TTS streams ("rails"),
each with its own voice, running in TRUE PARALLEL on a shared GPU.

The BatchCoordinator collects ready utterances from all rails and processes
them together through batched generation, enabling simultaneous output.

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
from typing import Dict, Optional, List, Tuple

import torch

from .tts_turbo import ChatterboxTurboTTS, Conditionals, normalize_text

logger = logging.getLogger(__name__)

MAX_RAILS = 6


class RailState(Enum):
    """Rail state machine states."""
    IDLE = auto()
    PENDING = auto()  # Text ready, waiting for batch
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

        # Worker thread for text collection
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
        submitted for parallel generation with other rails.
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
        self.state = RailState.IDLE

    def flush(self):
        """Force generation of any buffered text immediately."""
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
            and self.audio_queue.empty()
        )

    @property
    def has_audio(self) -> bool:
        """True if there's audio ready to read."""
        return not self.audio_queue.empty()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _run(self):
        """Worker thread main loop - collects text and submits to coordinator."""
        while self.running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Rail '{self.name}' error: {e}")
                time.sleep(0.1)

    def _tick(self):
        """Single iteration of the worker loop."""
        self._collect_text()

        with self._lock:
            text = self.pending_text
            if not text:
                time.sleep(0.01)
                return

            # Check if we should generate now
            sentence_end = text.rstrip().endswith(('.', '!', '?'))
            time_since_push = (time.time() - self.last_push_time) * 1000
            timeout_reached = time_since_push >= self.config.flush_timeout_ms

            if sentence_end or timeout_reached:
                self.pending_text = ""
            else:
                time.sleep(0.01)
                return

        # Submit to coordinator for batched generation
        self._submit_for_generation(text)

    def _collect_text(self):
        """Drain text queue into pending buffer."""
        while True:
            try:
                text = self.text_queue.get_nowait()
                with self._lock:
                    self.pending_text += text
            except Empty:
                break

    def _submit_for_generation(self, text: str):
        """Submit text to the coordinator for batched generation."""
        if self.config.voice in self.tts.voices:
            conds = self.tts.voices[self.config.voice]
        else:
            logger.warning(f"Voice '{self.config.voice}' not found, using default")
            conds = self.tts.model.conds

        utterance = PendingUtterance(
            rail_name=self.name,
            text=text,
            conds=conds,
            config=self.config,
        )

        self.state = RailState.PENDING
        self.tts.coordinator.submit(utterance)

    def _receive_audio(self, chunk: torch.Tensor):
        """Called by coordinator to deliver audio."""
        if not self.interrupt_flag.is_set():
            self.audio_queue.put(chunk)

    def _generation_complete(self):
        """Called by coordinator when generation is done."""
        self.state = RailState.IDLE

    @staticmethod
    def _clear_queue(q: Queue):
        """Clear all items from a queue."""
        while True:
            try:
                q.get_nowait()
            except Empty:
                break


class BatchCoordinator:
    """
    Coordinates batched generation across multiple rails.

    Collects pending utterances from rails and processes them together
    for true parallel generation on the GPU.
    """

    def __init__(self, tts: 'TTSRails'):
        self.tts = tts
        self.pending: Queue[PendingUtterance] = Queue()
        self.running = True

        # Batching parameters
        self.batch_wait_ms = 10  # Wait for more rails before processing
        self.max_batch_size = MAX_RAILS

        # Worker thread
        self.worker = threading.Thread(target=self._run, daemon=True, name="batch-coordinator")
        self.worker.start()

        logger.info("BatchCoordinator started")

    def submit(self, utterance: PendingUtterance):
        """Submit an utterance for batched generation."""
        self.pending.put(utterance)

    def shutdown(self):
        """Shutdown the coordinator."""
        self.running = False
        self.worker.join(timeout=2.0)
        logger.info("BatchCoordinator shutdown")

    def _run(self):
        """Main coordinator loop."""
        while self.running:
            try:
                self._process_batch()
            except Exception as e:
                logger.error(f"BatchCoordinator error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _process_batch(self):
        """Collect and process a batch of utterances."""
        # Wait for first utterance
        try:
            first = self.pending.get(timeout=0.1)
        except Empty:
            return

        batch = [first]

        # Brief wait to collect more utterances for batching
        deadline = time.time() + self.batch_wait_ms / 1000
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                utterance = self.pending.get(timeout=0.001)
                batch.append(utterance)
            except Empty:
                break

        # Process the batch
        self._generate_batch(batch)

    def _generate_batch(self, batch: List[PendingUtterance]):
        """Generate audio for a batch of utterances."""
        if not batch:
            return

        # Mark rails as generating
        for utterance in batch:
            rail = self.tts.rails.get(utterance.rail_name)
            if rail:
                rail.state = RailState.GENERATING

        # Prepare inputs for batched generation
        texts = [u.text for u in batch]
        conds_list = [u.conds for u in batch]

        # Use first utterance's config for shared parameters
        # (temperature, etc. should be similar across rails)
        config = batch[0].config

        logger.debug(f"Generating batch of {len(batch)} utterances: {[u.rail_name for u in batch]}")

        try:
            # Run batched generation
            for chunk_results in self.tts.model.generate_stream_batched(
                texts=texts,
                conds_list=conds_list,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
            ):
                # Distribute chunks to their respective rails
                for result in chunk_results:
                    utterance = batch[result.seq_id]
                    rail = self.tts.rails.get(utterance.rail_name)

                    if rail and not rail.interrupt_flag.is_set():
                        rail._receive_audio(result.audio)

        finally:
            # Mark all rails as idle
            for utterance in batch:
                rail = self.tts.rails.get(utterance.rail_name)
                if rail:
                    rail._generation_complete()


class TTSRails:
    """
    TTS orchestrator managing multiple parallel rails with TRUE BATCHED GENERATION.

    Compiles models once, keeps them hot, and manages voice registration
    and rail allocation. The BatchCoordinator enables simultaneous generation
    across all active rails.

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

        # Start batch coordinator
        self.coordinator = BatchCoordinator(self)

        logger.info("TTSRails ready (batched generation enabled)")

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
        self.coordinator.shutdown()
        for name in list(self.rails.keys()):
            self.close_rail(name)
        logger.info("TTSRails shutdown complete")

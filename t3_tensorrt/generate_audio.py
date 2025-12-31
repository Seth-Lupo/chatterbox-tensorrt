#!/usr/bin/env python3
"""
Generate audio using TensorRT-accelerated T3 transformer.

This replaces the PyTorch transformer with TensorRT for the prefill stage,
while keeping the rest of the pipeline in PyTorch.

Usage:
    python generate_audio.py "Hello, this is a test."
    python generate_audio.py "Hello world" --output hello.wav
    python generate_audio.py "Hello world" --compare  # Compare TRT vs PyTorch
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Add project to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ENGINE_PATH = SCRIPT_DIR / "t3_transformer.engine"


class T3WithTensorRT:
    """
    T3 TTS model with TensorRT-accelerated transformer.

    Uses TensorRT for the transformer forward pass (prefill),
    while keeping embeddings and vocoder in PyTorch.
    """

    def __init__(self, model, trt_engine_path: str):
        """
        Args:
            model: Original ChatterboxTurboTTS model
            trt_engine_path: Path to TensorRT engine
        """
        self.model = model
        self.device = model.device
        self.dtype = model.dtype

        # Load TensorRT engine
        from trt_wrapper import T3TensorRTTransformer
        self.trt_transformer = T3TensorRTTransformer(trt_engine_path)

        # Store original transformer for comparison
        self.pytorch_transformer = model.t3.tfmr

        print(f"T3 with TensorRT initialized")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")

    def generate(
        self,
        text: str,
        audio_prompt_path: str = None,
        use_trt: bool = True,
    ):
        """
        Generate audio from text.

        Args:
            text: Input text
            audio_prompt_path: Optional path to voice reference audio
            use_trt: If True, use TensorRT transformer; if False, use PyTorch

        Returns:
            audio: Generated audio tensor
            sample_rate: Audio sample rate
        """
        # For now, use the streaming API and collect all chunks
        audio_chunks = []

        if use_trt:
            # Temporarily replace transformer with a wrapper that uses TRT
            original_forward = self.model.t3.tfmr.forward

            def trt_forward(inputs_embeds=None, **kwargs):
                # Use TensorRT for transformer
                if inputs_embeds is not None:
                    # Ensure correct format
                    if inputs_embeds.dtype != torch.float16:
                        inputs_embeds = inputs_embeds.half()
                    if not inputs_embeds.is_contiguous():
                        inputs_embeds = inputs_embeds.contiguous()

                    # Run TensorRT
                    hidden_states = self.trt_transformer(inputs_embeds)

                    # Return in expected format
                    class Output:
                        def __init__(self, h):
                            self.last_hidden_state = h
                            self.hidden_states = (h,)
                            self.past_key_values = None
                        def __getitem__(self, idx):
                            return self.last_hidden_state if idx == 0 else None

                    return Output(hidden_states)
                else:
                    return original_forward(inputs_embeds=inputs_embeds, **kwargs)

            # Monkey-patch
            self.model.t3.tfmr.forward = trt_forward

        try:
            # Generate using streaming API
            for chunk, metrics in self.model.generate_stream(
                text=text,
                audio_prompt_path=audio_prompt_path,
            ):
                audio_chunks.append(chunk)
        finally:
            if use_trt:
                # Restore original forward
                self.model.t3.tfmr.forward = original_forward

        # Concatenate all chunks
        audio = torch.cat(audio_chunks, dim=-1)

        return audio, self.model.sr


def generate_audio(
    text: str,
    output_path: Path,
    use_trt: bool = True,
    audio_prompt_path: str = None,
):
    """Generate audio and save to file."""
    print("=" * 60)
    print("T3 TensorRT Audio Generation")
    print("=" * 60)

    # Check engine exists
    if use_trt and not ENGINE_PATH.exists():
        print(f"ERROR: TensorRT engine not found: {ENGINE_PATH}")
        print("Run ./build_engine.sh first, or use --no-trt for PyTorch only")
        sys.exit(1)

    # Load model
    print("\nLoading T3 model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )
    print("Model loaded!")

    if use_trt:
        # Wrap with TensorRT
        trt_model = T3WithTensorRT(model, str(ENGINE_PATH))
    else:
        trt_model = None

    # Generate
    print(f"\nGenerating audio for: \"{text}\"")
    print(f"  Using: {'TensorRT' if use_trt else 'PyTorch'}")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    if use_trt:
        audio, sr = trt_model.generate(text, audio_prompt_path, use_trt=True)
    else:
        # Use original model
        audio_chunks = []
        for chunk, metrics in model.generate_stream(text=text, audio_prompt_path=audio_prompt_path):
            audio_chunks.append(chunk)
        audio = torch.cat(audio_chunks, dim=-1)
        sr = model.sr

    torch.cuda.synchronize()
    gen_time = time.perf_counter() - start_time

    # Calculate stats
    audio_duration = audio.shape[-1] / sr
    rtf = gen_time / audio_duration

    print(f"\nGeneration complete!")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Real-time factor: {rtf:.2f}x")

    # Save audio
    print(f"\nSaving to: {output_path}")

    # Ensure audio is on CPU and correct shape
    audio_cpu = audio.cpu()
    if audio_cpu.dim() == 1:
        audio_cpu = audio_cpu.unsqueeze(0)

    torchaudio.save(str(output_path), audio_cpu, sr)
    print(f"  Saved! ({output_path.stat().st_size / 1024:.1f} KB)")

    return audio, sr, gen_time


def compare_trt_vs_pytorch(text: str, audio_prompt_path: str = None):
    """Compare TensorRT vs PyTorch generation times."""
    print("=" * 60)
    print("TensorRT vs PyTorch Comparison")
    print("=" * 60)

    # Load model
    print("\nLoading T3 model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )

    trt_model = T3WithTensorRT(model, str(ENGINE_PATH))

    print(f"\nGenerating: \"{text}\"")

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        for chunk, _ in model.generate_stream(text=text[:20]):
            pass

    # PyTorch timing
    print("\nPyTorch generation...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    audio_chunks = []
    for chunk, _ in model.generate_stream(text=text, audio_prompt_path=audio_prompt_path):
        audio_chunks.append(chunk)
    pytorch_audio = torch.cat(audio_chunks, dim=-1)

    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start

    # TensorRT timing
    print("TensorRT generation...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    trt_audio, sr = trt_model.generate(text, audio_prompt_path, use_trt=True)

    torch.cuda.synchronize()
    trt_time = time.perf_counter() - start

    # Results
    audio_duration = pytorch_audio.shape[-1] / model.sr

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAudio duration: {audio_duration:.2f}s")
    print(f"\n{'Method':<15} {'Time':<12} {'RTF':<10}")
    print("-" * 37)
    print(f"{'PyTorch':<15} {pytorch_time:.2f}s{'':<6} {pytorch_time/audio_duration:.2f}x")
    print(f"{'TensorRT':<15} {trt_time:.2f}s{'':<6} {trt_time/audio_duration:.2f}x")
    print("-" * 37)

    speedup = pytorch_time / trt_time
    print(f"\nSpeedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} with TensorRT")

    # Save both for comparison
    torchaudio.save("output_pytorch.wav", pytorch_audio.cpu().unsqueeze(0), model.sr)
    torchaudio.save("output_tensorrt.wav", trt_audio.cpu().unsqueeze(0), sr)
    print(f"\nSaved: output_pytorch.wav, output_tensorrt.wav")


def main():
    parser = argparse.ArgumentParser(description="Generate audio with TensorRT-accelerated T3")
    parser.add_argument("text", type=str, nargs="?",
                        default="Hello, this is a test of the TensorRT accelerated text to speech system.",
                        help="Text to synthesize")
    parser.add_argument("--output", "-o", type=Path, default=SCRIPT_DIR / "output.wav",
                        help="Output WAV file path")
    parser.add_argument("--audio-prompt", type=str, default=None,
                        help="Path to voice reference audio")
    parser.add_argument("--no-trt", action="store_true",
                        help="Use PyTorch instead of TensorRT")
    parser.add_argument("--compare", action="store_true",
                        help="Compare TensorRT vs PyTorch")
    args = parser.parse_args()

    if args.compare:
        compare_trt_vs_pytorch(args.text, args.audio_prompt)
    else:
        generate_audio(
            args.text,
            args.output,
            use_trt=not args.no_trt,
            audio_prompt_path=args.audio_prompt,
        )


if __name__ == "__main__":
    main()

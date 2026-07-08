#!/usr/bin/env python3
"""A/B benchmark for the optimized Sortformer and Silero Curator stages."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torchaudio

from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage


def _time_call(fn: Callable[[], Any], iterations: int) -> tuple[list[float], Any]:
    timings: list[float] = []
    result: Any = None
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return timings, result


def _summarize(timings: list[float], audio_seconds: float) -> dict[str, float]:
    median = statistics.median(timings)
    return {
        "median_seconds": median,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "realtime_factor": median / audio_seconds,
        "rtfx": audio_seconds / median,
    }


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_silero(audio_path: str, iterations: int) -> dict[str, Any]:
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_seconds = waveform.shape[-1] / sample_rate
    results: dict[str, Any] = {}

    for backend in ("torch", "onnx"):
        stage = VADSegmentationStage(backend=backend)
        setup_start = time.perf_counter()
        stage.setup()
        setup_seconds = time.perf_counter() - setup_start
        stage._get_vad_segments(waveform, sample_rate)  # warm-up
        timings, segments = _time_call(
            lambda: stage._get_vad_segments(waveform, sample_rate),
            iterations,
        )
        results[backend] = {
            "setup_seconds": setup_seconds,
            "segments": segments,
            **_summarize(timings, audio_seconds),
        }
        stage.teardown()

    torch_segments = results["torch"]["segments"]
    onnx_segments = results["onnx"]["segments"]
    results["parity"] = {
        "exact_segment_match": torch_segments == onnx_segments,
        "torch_segment_count": len(torch_segments),
        "onnx_segment_count": len(onnx_segments),
    }
    results["onnx_speedup_over_torch"] = (
        results["torch"]["median_seconds"] / results["onnx"]["median_seconds"]
    )
    return results


def _sortformer_case(
    audio_paths: list[str],
    audio_seconds: float,
    iterations: int,
    *,
    precision: str,
    avoid_cuda_cache_flush: bool,
) -> dict[str, Any]:
    stage = InferenceSortformerStage(
        precision=precision,
        avoid_cuda_cache_flush=avoid_cuda_cache_flush,
        inference_batch_size=len(audio_paths),
        batch_size=len(audio_paths),
    )
    setup_start = time.perf_counter()
    stage.setup()
    setup_seconds = time.perf_counter() - setup_start

    first_timings, first_segments = _time_call(lambda: stage._diarize(audio_paths), 1)
    timings, segments = _time_call(lambda: stage._diarize(audio_paths), iterations)
    result = {
        "setup_seconds": setup_seconds,
        "first_inference_seconds": first_timings[0],
        "segment_count": sum(len(item) for item in segments),
        "segments": segments,
        **_summarize(timings, audio_seconds),
    }
    del stage
    _release_cuda()
    return result


def benchmark_sortformer(audio_paths: list[str], iterations: int) -> dict[str, Any]:
    audio_seconds = 0.0
    for audio_path in audio_paths:
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_seconds += waveform.shape[-1] / sample_rate
        del waveform
    cases = {
        "eager_fp32": {"precision": "fp32", "avoid_cuda_cache_flush": False},
        "eager_fp32_cache_optimized": {"precision": "fp32", "avoid_cuda_cache_flush": True},
        "eager_fp16_optimized": {"precision": "fp16", "avoid_cuda_cache_flush": True},
        "eager_bf16_optimized": {"precision": "bf16", "avoid_cuda_cache_flush": True},
    }
    results = {
        name: _sortformer_case(audio_paths, audio_seconds, iterations, **config)
        for name, config in cases.items()
    }
    baseline = results["eager_fp32"]["median_seconds"]
    for name, result in results.items():
        result["speedup_over_eager_fp32"] = baseline / result["median_seconds"]
        result["same_segments_as_eager_fp32"] = result["segments"] == results["eager_fp32"]["segments"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--skip-sortformer", action="store_true")
    parser.add_argument("--skip-silero", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    missing = [path for path in args.audio if not path.is_file()]
    if missing:
        parser.error(f"Audio file does not exist: {missing[0]}")

    output: dict[str, Any] = {
        "audio": [str(path) for path in args.audio],
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not args.skip_silero:
        output["silero"] = benchmark_silero(str(args.audio[0]), args.iterations)
    if not args.skip_sortformer:
        output["sortformer"] = benchmark_sortformer([str(path) for path in args.audio], args.iterations)

    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

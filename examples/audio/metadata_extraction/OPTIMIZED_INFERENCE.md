# Optimized Sortformer and Silero inference

The metadata extraction pipeline offers lightweight, open-source acceleration
without changing its output schema or requiring NVIDIA Riva artifacts.

## Recommended starting configuration

```bash
python examples/audio/metadata_extraction/run_metadata_extraction.py \
  --data_config input.yaml \
  --output_dir output \
  --sortformer_model nvidia/diar_streaming_sortformer_4spk-v2 \
  --sortformer_precision fp32 \
  --sortformer_reuse_cuda_cache \
  --sortformer_stage_batch_size 8 \
  --sortformer_batch_size 4 \
  --vad_backend onnx
```

### Sortformer

- `--sortformer_reuse_cuda_cache` keeps PyTorch's CUDA caching allocator warm
  instead of flushing it after each NeMo inference batch. It is beneficial for
  batch size 8 or greater; leave it disabled for single-recording latency.
- Keep `--sortformer_precision fp32` when exact output parity is required.
  `fp16` and `bf16` are optional throughput modes and must be validated on the
  target diarization corpus because threshold-adjacent predictions can move or
  split segments.
- `--sortformer_stage_batch_size` controls the Ray batch delivered to an actor;
  `--sortformer_batch_size` controls NeMo's inference batch. Tune both together.
- Whole-model `torch.compile` is intentionally not enabled. NeMo's streaming
  Python control flow and CPU-originating inputs cause graph breaks and high
  compilation overhead for this model.

### Silero VAD

- `--vad_backend onnx` loads the ONNX model shipped by the official
  `silero-vad` package and executes it with ONNX Runtime on CPU.
- Segment creation still uses `get_speech_timestamps`, so threshold, minimum
  duration, maximum duration, silence interval, and padding semantics are
  shared with the Torch backend.
- `--vad_backend torch` remains available for compatibility and optional CUDA
  execution when the stage is configured with GPU resources.

## Evaluation

Compare against `--sortformer_precision fp32` and `--vad_backend torch`. Measure end-to-end audio
hours per wall-clock hour in addition to individual stage latency. For quality,
compare diarization error rate and speaker counts for Sortformer, and speech
boundary differences plus downstream ASR WER for Silero.

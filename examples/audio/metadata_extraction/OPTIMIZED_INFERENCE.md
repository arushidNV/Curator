# Optimized Sortformer and Silero inference

The metadata extraction pipeline offers lightweight, open-source acceleration
without changing its output schema or requiring NVIDIA Riva artifacts.

## Recommended starting configuration

```bash
python examples/audio/metadata_extraction/run_metadata_extraction.py \
  --data_config input.yaml \
  --output_dir output \
  --sortformer_model nvidia/diar_streaming_sortformer_4spk-v2 \
  --sortformer_precision bf16 \
  --sortformer_compile \
  --sortformer_stage_batch_size 8 \
  --sortformer_batch_size 4 \
  --vad_backend onnx
```

### Sortformer

- `--sortformer_compile` compiles the NeMo model's forward pass with
  `torch.compile(mode="reduce-overhead", dynamic=True)` while retaining NeMo's
  streaming state and timestamp postprocessing.
- `--sortformer_precision bf16` uses CUDA autocast. Use `fp16` on GPUs without
  BF16 support or `fp32` when establishing an accuracy baseline.
- The optimized stage keeps PyTorch's CUDA caching allocator warm instead of
  flushing its cache after each NeMo inference batch.
- `--sortformer_stage_batch_size` controls the Ray batch delivered to an actor;
  `--sortformer_batch_size` controls NeMo's inference batch. Tune both together.
- Compilation is lazy, so exclude the first batch when measuring steady-state
  throughput.

### Silero VAD

- `--vad_backend onnx` loads the ONNX model shipped by the official
  `silero-vad` package and executes it with ONNX Runtime on CPU.
- Segment creation still uses `get_speech_timestamps`, so threshold, minimum
  duration, maximum duration, silence interval, and padding semantics are
  shared with the Torch backend.
- `--vad_backend torch` remains available for compatibility and optional CUDA
  execution when the stage is configured with GPU resources.

## Evaluation

Compare against `--sortformer_precision fp32` without
`--sortformer_compile`, and `--vad_backend torch`. Measure end-to-end audio
hours per wall-clock hour in addition to individual stage latency. For quality,
compare diarization error rate and speaker counts for Sortformer, and speech
boundary differences plus downstream ASR WER for Silero.

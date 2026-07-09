# Optimized Sortformer and Silero inference

The metadata extraction pipeline offers native in-process acceleration without
changing its output schema or calling an NVIDIA Riva/Triton service. TensorRT
engines are built from the exact checkpoints used by the Curator stages.

## TensorRT engine preparation

Build engines on the target GPU and with the same TensorRT version used for
inference. FP32 is the initial parity mode; FP16/BF16 require a separate
quality evaluation.

```bash
python scripts/audio/build_sortformer_tensorrt_engine.py \
  --model /models/diar_streaming_sortformer_4spk-v2.nemo \
  --onnx /models/diar_streaming_sortformer_4spk-v2.streaming.onnx \
  --output /models/diar_streaming_sortformer_4spk-v2.fp32.plan \
  --precision fp32

python scripts/audio/build_silero_tensorrt_engine.py \
  --onnx /models/silero_vad.onnx \
  --output /models/silero_vad.fp32.plan
```

The Sortformer builder writes a JSON sidecar containing SHA-256 hashes for the
checkpoint, ONNX graph, and engine. Do not pair the engine with another model
revision even if its tensor names are identical.

## Recommended TensorRT configuration

```bash
python examples/audio/metadata_extraction/run_metadata_extraction.py \
  --data_config input.yaml \
  --output_dir output \
  --sortformer_model /models/diar_streaming_sortformer_4spk-v2.nemo \
  --sortformer_backend tensorrt \
  --sortformer_tensorrt_engine /models/diar_streaming_sortformer_4spk-v2.fp32.plan \
  --sortformer_precision fp32 \
  --sortformer_stage_batch_size 8 \
  --sortformer_batch_size 8 \
  --vad_backend tensorrt \
  --vad_tensorrt_engine /models/silero_vad.fp32.plan \
  --vad_gpus 1
```

### Sortformer

- `--sortformer_backend tensorrt` leaves NeMo's audio preprocessing, streaming
  cache semantics, and diarization postprocessing in place. The exported
  six-input streaming neural graph runs through one persistent TensorRT engine,
  execution context, and non-default CUDA stream per Curator actor.
- Ray batches are sorted into duration buckets before NeMo micro-batching and
  then restored to their original task order. This reduces padding without
  changing manifest ordering.
- `--sortformer_reuse_cuda_cache` keeps PyTorch's CUDA caching allocator warm
  for the PyTorch backend. It is not needed by the TensorRT backend.
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

- `--vad_backend tensorrt` uses one persistent engine/context/stream and
  shape-aware reusable GPU buffers. Recurrent state and the 64-sample input
  context remain on the GPU across 512-sample windows and reset between files.
- `--vad_backend onnx` loads the ONNX model shipped by the official
  `silero-vad` package and executes it with ONNX Runtime on CPU.
- Segment creation still uses `get_speech_timestamps`, so threshold, minimum
  duration, maximum duration, silence interval, and padding semantics are
  shared with the Torch backend.
- `--vad_backend torch` remains available for compatibility and optional CUDA
  execution when the stage is configured with GPU resources.

## Evaluation

Compare each backend independently before combining them:

1. PyTorch Sortformer + Torch Silero (baseline).
2. TensorRT Sortformer + Torch Silero.
3. PyTorch Sortformer + TensorRT Silero.
4. TensorRT Sortformer + TensorRT Silero.

Measure end-to-end audio hours per wall-clock hour and stage-only GPU time. For
quality, require RTTM/DER parity for Sortformer and boundary/downstream-ASR
parity for Silero. Do not infer a gain from average GPU utilization alone.

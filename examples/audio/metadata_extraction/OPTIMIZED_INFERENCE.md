# Optimized Sortformer and Silero inference

The metadata extraction pipeline offers native in-process acceleration without
changing its output schema or calling an NVIDIA Riva/Triton service. TensorRT
engines are built from the exact checkpoints used by the Curator stages.

## TensorRT engine preparation

Build engines on the target GPU and with the same TensorRT version used for
inference. The production defaults use FP32: this preserves threshold-sensitive
diarization and recurrent VAD behavior while TensorRT batching supplies the
throughput gain.

```bash
python scripts/audio/build_sortformer_tensorrt_engine.py \
  --model /models/diar_streaming_sortformer_4spk-v2.nemo \
  --onnx /models/diar_streaming_sortformer_4spk-v2.streaming.onnx \
  --output /models/diar_streaming_sortformer_4spk-v2.fp32.plan \
  --precision fp32

python scripts/audio/build_silero_tensorrt_engine.py \
  --onnx-output /models/silero_vad.clean.onnx \
  --output /models/silero_vad.fp32.plan \
  --opt-batch 32 \
  --max-batch 64
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
  --sortformer_stage_batch_size 32 \
  --sortformer_batch_size 8 \
  --sortformer_gpus 0.5 \
  --vad_backend tensorrt \
  --vad_tensorrt_engine /models/silero_vad.fp32.plan \
  --vad_stage_batch_size 32 \
  --vad_gpus 0.5 \
  --read_concurrency 4 \
  --writer_concurrency 4
```

### Sortformer

- `--sortformer_backend tensorrt` leaves NeMo's audio preprocessing, streaming
  cache semantics, and diarization postprocessing in place. The exported
  six-input streaming neural graph runs through one persistent TensorRT engine,
  execution context, and non-default CUDA stream per Curator actor.
- The TensorRT backend deliberately uses one persistent Ray actor. GPU
  concurrency comes from model batch size 8 and a duration-sorted stage window
  of 32 recordings, avoiding duplicate model/engine allocations.
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

- `--vad_backend tensorrt` rebuilds the branch-free 16 kHz Silero core from the
  official 6.2.1 TorchScript weights. This avoids the nested `If` nodes in the
  public convenience wrapper that TensorRT cannot compile.
- One persistent engine/context/stream advances many recordings together.
  Recurrent state and the 64-sample input context remain on the GPU across
  512-sample windows; completed recordings are compacted out of later batches.
- The builder checks the rebuilt PyTorch core against the official model,
  checks ONNX Runtime parity, and checks recurrent TensorRT parity before it
  writes a usable engine. TF32 is disabled for this small recurrent graph to
  prevent long-recording state drift.
- `--vad_backend onnx` loads the ONNX model shipped by the official
  `silero-vad` package and executes it with ONNX Runtime on CPU.
- Segment creation implements the official `get_speech_timestamps` 6.2.1
  algorithm over the batched probabilities, so threshold, minimum duration,
  maximum duration, silence interval, and padding semantics remain shared with
  the Torch backend.
- `--vad_backend torch` remains available for compatibility and optional CUDA
  execution when the stage is configured with GPU resources.

## Evaluation

For a delivery benchmark, compare only two immutable images on the same GPU:

1. The untouched original Curator commit using its original PyTorch stages.
2. The final commit using TensorRT for both neural graphs and the recommended
   batching above.

First run two recordings and require successful engine execution markers,
valid RTTM/manifests, and nonempty output. Then report end-to-end audio hours
per wall-clock hour and `baseline_wall / final_wall` on the full dataset.
Build-time numerical preflights establish graph parity; corpus-level RTTM and
speech-duration summaries catch gross end-to-end regressions. Do not infer a
gain from average GPU utilization alone.

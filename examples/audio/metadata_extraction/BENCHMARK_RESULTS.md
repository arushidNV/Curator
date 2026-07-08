# Initial inference benchmark

Measured in an isolated container on one NVIDIA RTX A5000 using real Riva ASR
test recordings. Timings exclude model setup and use warmed inference.

## Silero VAD

Both backends used the same Curator `get_speech_timestamps` postprocessing.

| Audio duration | Torch median | ONNX median | Speedup | Segment parity |
|---:|---:|---:|---:|---|
| 63.5 s | 0.568 s | 0.306 s | 1.86x | Exact |
| 123.7 s | 1.050 s | 0.633 s | 1.66x | Exact |

Recommendation: use the ONNX backend for CPU VAD.

## Streaming Sortformer

| Batch | Mode | Median | RTFx | Speedup | Exact FP32 segment parity |
|---:|---|---:|---:|---:|---|
| 1 | FP32 baseline | 0.180 s | 250x | 1.00x | Yes |
| 1 | FP32, reuse CUDA cache | 0.193 s | 233x | 0.93x | Yes |
| 4 | FP32 baseline | 0.672 s | 357x | 1.00x | Yes |
| 4 | FP32, reuse CUDA cache | 0.664 s | 362x | 1.01x | Yes |
| 4 | BF16, reuse CUDA cache | 0.625 s | 384x | 1.08x | No |
| 8 | FP32 baseline | 1.722 s | 326x | 1.00x | Yes |
| 8 | FP32, reuse CUDA cache | 1.587 s | 354x | 1.09x | Yes |
| 8 | FP16, reuse CUDA cache | 1.551 s | 362x | 1.11x | No |
| 8 | BF16, reuse CUDA cache | 1.511 s | 371x | 1.14x | No |

Whole-model `torch.compile` was rejected: compilation exceeded 2.5 minutes,
could not capture CUDA graphs because NeMo receives CPU-originating inputs, and
failed a TorchDynamo guard in the encoder.

Recommendation: batch eight similarly sized recordings, retain FP32, and reuse
the CUDA allocator. Treat FP16/BF16 as experimental until DER is evaluated.

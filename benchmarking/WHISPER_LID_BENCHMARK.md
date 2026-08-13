# Whisper LID optimization benchmark

Date: 2026-08-06

## Scope

This benchmark compares pull request 57's Whisper LID path with the optimized
GPU preprocessing path in `WhisperLangIDStage`.

These measurements predate the port onto pull request 60. They validate the
optimized inference path, but they are not an end-to-end PR 60/Ray performance
claim; PR 60's batch-size and worker configuration require a separate run.

- GPU: NVIDIA RTX A5000 (24 GB)
- Model: OpenAI Whisper Medium (`medium.pt`)
- Batch size: 8
- Samples: 1,000 real WAV files, 100 per language
- Languages: Arabic, German, English, Spanish, French, Hindi, Italian,
  Japanese, Portuguese, and Russian
- Total audio: 7,607.85 seconds (2.11 hours)
- Duration range: 1.00-24.12 seconds; median 5.76 seconds
- Selection seed: 57
- Audio decode and resampling were completed before timing

The test data came from labeled local ASR evaluation corpora under
`/home/arushid/Seagate/arushid/asr_datasets`. Directory-level locale labels
were used as ground truth.

## Compared paths

1. **PR 57:** CPU log-Mel/STFT, batch-global dynamic-range normalization,
   FP32 Mel input.
2. **Optimized FP32:** GPU log-Mel/STFT, per-sample normalization, FP32 Mel
   input. This isolates the preprocessing and normalization changes.
3. **Optimized FP16:** GPU log-Mel/STFT, per-sample normalization, FP16 Mel
   input. Whisper model parameters remain FP32.

The order of the three paths was rotated for each batch to reduce warm-up and
thermal-order bias. Model loading and audio decoding were excluded from the
timed region.

## Aggregate results

| Path | Accuracy | Files/s | Audio x realtime | Median batch | P95 batch | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| PR 57 | 96.5% (965/1000) | 7.69 | 58.50x | 1047.7 ms | 1059.8 ms | 3.367 GiB |
| Optimized FP32 | 96.7% (967/1000) | 8.13 | 61.86x | 993.2 ms | 997.7 ms | 3.382 GiB |
| Optimized FP16 | 96.6% (966/1000) | 36.72 | 279.36x | 219.0 ms | 221.3 ms | 3.126 GiB |

- Optimized FP32 speedup over PR 57: **1.06x**
- Optimized FP16 speedup over PR 57: **4.78x**
- Optimized FP16 peak-memory reduction: **7.16%**
- PR 57 and optimized FP16 prediction agreement: **99.4%**
- PR 57 95% Wilson accuracy interval: **95.17%-97.47%**
- Optimized FP16 95% Wilson accuracy interval: **95.29%-97.56%**

Of the six predictions changed by optimized FP16, three became correct, two
became incorrect, and one remained incorrect. The net change was therefore
+1 correct result out of 1,000; this evaluation provides no evidence of an
accuracy regression from FP16.

## Accuracy by language

| Language | PR 57 | Optimized FP32 | Optimized FP16 |
|---|---:|---:|---:|
| Arabic (`ar`) | 100% | 100% | 100% |
| German (`de`) | 100% | 100% | 100% |
| English (`en`) | 90% | 90% | 90% |
| Spanish (`es`) | 98% | 98% | 98% |
| French (`fr`) | 100% | 100% | 100% |
| Hindi (`hi`) | 91% | 91% | 90% |
| Italian (`it`) | 100% | 100% | 100% |
| Japanese (`ja`) | 100% | 100% | 100% |
| Portuguese (`pt`) | 87% | 89% | 89% |
| Russian (`ru`) | 99% | 99% | 99% |

## Interpretation

Moving preprocessing to the GPU alone gives a modest 1.06x gain because the
Whisper Medium encoder dominates FP32 runtime. FP16 Mel inputs activate
Whisper's mixed-precision inference path and produce the material 4.78x
speedup.

The per-sample normalization fix improves determinism: PR 57's use of
`whisper.log_mel_spectrogram` on a batch applies one scalar maximum across all
rows, so a clip's features and prediction can vary with its batch neighbors.
The optimized implementation matches independent per-file normalization while
retaining a vectorized STFT. In FP32 it changed five predictions and improved
net accuracy by two samples.

## Limitations

- This is a balanced 10-language evaluation, not all Whisper languages.
- Directory locale is treated as ground truth; code-switching inside a clip is
  not annotated.
- Results cover one A5000 and batch size 8.
- End-to-end Curator/Ray scheduling and audio decode time are not included.

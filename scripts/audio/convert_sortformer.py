#!/usr/bin/env python3
"""Convert streaming Sortformer checkpoints from NeMo to Riva and TensorRT.

Examples:

  # Run in an environment containing the checked-out NeMo dependencies.
  python convert_sortformer.py nemo-to-riva new.nemo new.riva

  # Run on the deployment GPU in the matching Riva NIM image.
  python convert_sortformer.py riva-to-trt new.riva engine/sortformer_unified.trt

  # If one Python environment contains both NeMo and TensorRT:
  python convert_sortformer.py all new.nemo build/new.riva \
      build/sortformer_unified.trt

The generated .riva is an unencrypted Riva model archive. Encryption, RMIR
creation, and deployment are separate Riva ServiceMaker concerns.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import yaml


RIVA_MANIFEST = {
    "artifacts": {
        "model_config.yaml": {
            "artifact_type": "File",
            "conf_path": "model_config.yaml",
            "content_callback": "BinaryContentCallback",
            "description": "",
            "encryption": False,
            "path_type": "TAR_PATH",
        },
        "model_graph.onnx": {
            "artifact_type": "File",
            "content_callback": "BinaryContentCallback",
            "description": "Exported model",
            "encryption": False,
            "onnx": True,
            "onnx_archive_format": 1,
            "runtime": "ONNX",
        },
    },
    "metadata": {
        "description": "Exported Nemo Model",
        "format_version": 3,
        "has_pytorch_checkpoint": False,
        "min_nemo_version": "1.3",
        "obj_cls": "nemo.collections.asr.models.SortformerEncLabelModel",
        "onnx": True,
        "onnx_archive_format": 1,
        "runtime": "ONNX",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tar_member_bytes(archive: Path, accepted_names: set[str]) -> bytes:
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            normalized = member.name.lstrip("./")
            if normalized in accepted_names:
                extracted = bundle.extractfile(member)
                if extracted is None:
                    break
                return extracted.read()
    raise FileNotFoundError(
        f"None of {sorted(accepted_names)} was found in archive {archive}"
    )


def add_bytes(bundle: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    bundle.addfile(info, io.BytesIO(data))


def nemo_to_riva(
    nemo_path: Path,
    riva_path: Path,
    exporter: Path,
    device: str,
    keep_onnx: Path | None,
    no_bf16_roundtrip: bool,
):
    if not nemo_path.is_file():
        raise FileNotFoundError(nemo_path)
    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    riva_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sortformer-export-") as temp_directory:
        temporary_onnx = Path(temp_directory) / "model_graph.onnx"
        command = [
            sys.executable,
            str(exporter),
            str(nemo_path),
            str(temporary_onnx),
            "--device",
            device,
        ]
        if no_bf16_roundtrip:
            command.append("--no-bf16-roundtrip")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        model_config = tar_member_bytes(nemo_path, {"model_config.yaml"})
        manifest = yaml.safe_dump(
            RIVA_MANIFEST, sort_keys=False, default_flow_style=False
        ).encode()
        with tarfile.open(riva_path, "w:gz", compresslevel=9) as bundle:
            add_bytes(bundle, "artifacts/model_config.yaml", model_config)
            bundle.add(temporary_onnx, "artifacts/model_graph.onnx", recursive=False)
            add_bytes(bundle, "manifest.yaml", manifest)

        if keep_onnx is not None:
            keep_onnx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(temporary_onnx, keep_onnx)

    print(
        f"Wrote {riva_path} ({riva_path.stat().st_size / 2**20:.1f} MiB), "
        f"sha256={sha256(riva_path)}"
    )


def nested(config: dict, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def model_parameters(config: dict, args) -> dict:
    speakers = (
        args.num_speakers
        if args.num_speakers is not None
        else nested(config, "sortformer_modules", "num_spks", default=None)
    )
    if speakers is None:
        speakers = config.get("max_num_of_spks", 4)
    spkcache_len = (
        args.spkcache_len
        if args.spkcache_len is not None
        else nested(config, "sortformer_modules", "spkcache_len", default=160)
    )
    emb_dim = nested(config, "model_defaults", "fc_d_model", default=512)
    subsampling = nested(config, "encoder", "subsampling_factor", default=8)
    preprocessor = config.get("preprocessor", {})
    sample_rate = int(preprocessor.get("sample_rate", config.get("sample_rate", 16000)))
    window_size = float(preprocessor.get("window_size", 0.025))
    window_stride = float(preprocessor.get("window_stride", 0.01))
    return {
        "num_speakers": int(speakers),
        "spkcache_len": int(spkcache_len),
        "fifo_len": int(args.fifo_len),
        "chunk_len": int(args.chunk_len),
        "emb_dim": int(emb_dim),
        "subsampling_factor": int(subsampling),
        "spkcache_refresh_rate": int(args.spkcache_refresh_rate),
        "max_batch_size": int(args.max_batch_size),
        "opt_batch_size": int(args.opt_batch_size),
        "sample_rate": sample_rate,
        "n_fft": int(preprocessor.get("n_fft", 512)),
        "win_length": int(round(sample_rate * window_size)),
        "hop_length": int(round(sample_rate * window_stride)),
        "preemphasis": float(preprocessor.get("preemph", 0.97)),
        "log_guard": float(2**-24),
        "center_chunk_frames": int(args.center_chunk_frames),
        "left_context_frames": int(args.left_context_frames),
        "right_context_frames": int(args.right_context_frames),
        "output_step_ms": int(args.output_step_ms),
    }


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    parameters: dict,
    precision: str,
    workspace_gib: int,
    verbose: bool,
):
    import tensorrt as trt

    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib) * (1 << 30)
    )
    if precision == "bf16":
        build_config.set_flag(trt.BuilderFlag.BF16)
        build_config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "fp16":
        build_config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError(f"Unsupported precision: {precision}")

    if precision != "fp32":
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            layer_name = (layer.name or "").lower()
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT):
                continue
            if layer.type == trt.LayerType.SOFTMAX or any(
                token in layer_name for token in ("layernorm", "norm", "ln")
            ):
                layer.precision = trt.float32
                for output_index in range(layer.num_outputs):
                    layer.set_output_type(output_index, trt.float32)

    minimum_batch = 1
    optimum_batch = parameters["opt_batch_size"]
    maximum_batch = parameters["max_batch_size"]
    chunk_len = parameters["chunk_len"]
    cache_len = parameters["spkcache_len"]
    fifo_len = parameters["fifo_len"]
    emb_dim = parameters["emb_dim"]
    profile_shapes = {
        "chunk": (
            (minimum_batch, chunk_len, 128),
            (optimum_batch, chunk_len, 128),
            (maximum_batch, chunk_len, 128),
        ),
        "chunk_lengths": (
            (minimum_batch,),
            (optimum_batch,),
            (maximum_batch,),
        ),
        "spkcache": (
            (minimum_batch, 1, emb_dim),
            (optimum_batch, cache_len, emb_dim),
            (maximum_batch, cache_len, emb_dim),
        ),
        "spkcache_lengths": (
            (minimum_batch,),
            (optimum_batch,),
            (maximum_batch,),
        ),
        "fifo": (
            (minimum_batch, 1, emb_dim),
            (optimum_batch, max(1, fifo_len // 2), emb_dim),
            (maximum_batch, max(1, fifo_len), emb_dim),
        ),
        "fifo_lengths": (
            (minimum_batch,),
            (optimum_batch,),
            (maximum_batch,),
        ),
    }
    profile = builder.create_optimization_profile()
    for name, (minimum, optimum, maximum) in profile_shapes.items():
        # TensorRT 10.x returns None on success, while some older bindings
        # returned a boolean. Profile validity is checked when it is attached.
        profile.set_shape(name, minimum, optimum, maximum)
    build_config.add_optimization_profile(profile)

    print(
        f"Building TensorRT engine: precision={precision}, "
        f"batch={minimum_batch}/{optimum_batch}/{maximum_batch}"
    )
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


def copy_runtime_module(destination: Path, source: Path | None):
    candidates = [
        source,
        Path("/opt/riva/backends/sortformer_modules.py"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            shutil.copyfile(candidate, destination / "sortformer_modules.py")
            return
    raise FileNotFoundError(
        "Riva's sortformer_modules.py was not found. Run inside the matching "
        "Riva NIM image or provide --runtime-modules PATH."
    )


def riva_to_trt(riva_path: Path, engine_path: Path, args):
    if not riva_path.is_file():
        raise FileNotFoundError(riva_path)
    config_bytes = tar_member_bytes(
        riva_path, {"artifacts/model_config.yaml", "model_config.yaml"}
    )
    onnx_bytes = tar_member_bytes(
        riva_path, {"artifacts/model_graph.onnx", "model_graph.onnx"}
    )
    model_config = yaml.safe_load(config_bytes)
    parameters = model_parameters(model_config, args)

    with tempfile.TemporaryDirectory(prefix="sortformer-trt-") as temporary:
        onnx_path = Path(temporary) / "model_graph.onnx"
        onnx_path.write_bytes(onnx_bytes)
        build_trt_engine(
            onnx_path,
            engine_path,
            parameters,
            args.precision,
            args.workspace_gib,
            args.verbose,
        )

    try:
        from librosa.filters import mel as librosa_mel
    except ImportError as error:
        raise RuntimeError("librosa is required to generate the NeMo mel basis") from error
    mel_basis = librosa_mel(
        sr=parameters["sample_rate"],
        n_fft=parameters["n_fft"],
        n_mels=128,
        fmin=0,
        fmax=None,
        norm="slaney",
    ).astype(np.float32)
    mel_path = engine_path.parent / "mel_basis.npy"
    np.save(mel_path, mel_basis)

    copy_runtime_module(engine_path.parent, args.runtime_modules)
    runtime_config = {
        **parameters,
        "precision": args.precision,
        "mel_basis": mel_path.name,
        "source_riva": str(riva_path.resolve()),
        "source_riva_sha256": sha256(riva_path),
        "engine_sha256": sha256(engine_path),
    }
    config_path = engine_path.with_suffix(".json")
    config_path.write_text(json.dumps(runtime_config, indent=2) + "\n")
    print(
        f"Wrote {engine_path} ({engine_path.stat().st_size / 2**20:.1f} MiB)\n"
        f"Wrote {config_path}\n"
        f"Wrote {mel_path}"
    )


def add_trt_arguments(parser):
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--opt-batch-size", type=int, default=128)
    parser.add_argument("--chunk-len", type=int, default=128)
    parser.add_argument("--spkcache-len", type=int)
    parser.add_argument("--fifo-len", type=int, default=80)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--spkcache-refresh-rate", type=int, default=0)
    parser.add_argument("--center-chunk-frames", type=int, default=112)
    parser.add_argument("--left-context-frames", type=int, default=16)
    parser.add_argument("--right-context-frames", type=int, default=0)
    parser.add_argument("--output-step-ms", type=int, default=80)
    parser.add_argument("--workspace-gib", type=int, default=16)
    parser.add_argument("--runtime-modules", type=Path)
    parser.add_argument("--verbose", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    nemo_parser = subparsers.add_parser("nemo-to-riva")
    nemo_parser.add_argument("nemo", type=Path)
    nemo_parser.add_argument("riva", type=Path)
    nemo_parser.add_argument(
        "--exporter",
        type=Path,
        default=Path(__file__).with_name(
            "reverse_engineered_sortformer_onnx_export.py"
        ),
    )
    nemo_parser.add_argument("--device", default="cpu")
    nemo_parser.add_argument("--keep-onnx", type=Path)
    nemo_parser.add_argument("--no-bf16-roundtrip", action="store_true")

    trt_parser = subparsers.add_parser("riva-to-trt")
    trt_parser.add_argument("riva", type=Path)
    trt_parser.add_argument("engine", type=Path)
    add_trt_arguments(trt_parser)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("nemo", type=Path)
    all_parser.add_argument("riva", type=Path)
    all_parser.add_argument("engine", type=Path)
    all_parser.add_argument(
        "--exporter",
        type=Path,
        default=Path(__file__).with_name(
            "reverse_engineered_sortformer_onnx_export.py"
        ),
    )
    all_parser.add_argument("--device", default="cpu")
    all_parser.add_argument("--keep-onnx", type=Path)
    all_parser.add_argument("--no-bf16-roundtrip", action="store_true")
    add_trt_arguments(all_parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command in ("nemo-to-riva", "all"):
        nemo_to_riva(
            args.nemo,
            args.riva,
            args.exporter,
            args.device,
            args.keep_onnx,
            args.no_bf16_roundtrip,
        )
    if args.command in ("riva-to-trt", "all"):
        riva_to_trt(args.riva, args.engine, args)


if __name__ == "__main__":
    main()

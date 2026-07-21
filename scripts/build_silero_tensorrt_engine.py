#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export Silero's exact 16 kHz inference core and build a TensorRT engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _make_silero_core():  # noqa: ANN202
    """Rebuild the branch-free 16 kHz core from the official TorchScript weights."""
    import torch
    from silero_vad import load_silero_vad
    from torch.nn import functional

    class Silero16kInferenceCore(torch.nn.Module):
        """Branch-free 576-sample recurrent inference graph."""

        def __init__(self, source) -> None:  # noqa: ANN001
            super().__init__()
            self.register_buffer("forward_basis", source.stft.forward_basis_buffer.detach().clone())
            self.register_buffer("right_reflect_indices", torch.arange(574, 510, -1, dtype=torch.int64))
            self.encoder = torch.nn.ModuleList()
            for index in range(4):
                source_conv = getattr(source.encoder, str(index)).reparam_conv
                conv = torch.nn.Conv1d(
                    source_conv.in_channels,
                    source_conv.out_channels,
                    source_conv.kernel_size,
                    stride=source_conv.stride,
                    padding=source_conv.padding,
                    dilation=source_conv.dilation,
                )
                conv.load_state_dict(source_conv.state_dict())
                self.encoder.append(conv)

            source_rnn = source.decoder.rnn
            self.rnn_weight_ih = torch.nn.Parameter(source_rnn.weight_ih.detach().clone())
            self.rnn_weight_hh = torch.nn.Parameter(source_rnn.weight_hh.detach().clone())
            self.rnn_bias_ih = torch.nn.Parameter(source_rnn.bias_ih.detach().clone())
            self.rnn_bias_hh = torch.nn.Parameter(source_rnn.bias_hh.detach().clone())
            source_output = getattr(source.decoder.decoder, "2")
            self.output = torch.nn.Conv1d(128, 1, 1)
            self.output.load_state_dict(source_output.state_dict())

        def forward(self, audio: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            reflected = torch.index_select(audio, 1, self.right_reflect_indices)
            padded = torch.cat((audio, reflected), dim=1).unsqueeze(1)
            transform = functional.conv1d(padded, self.forward_basis, stride=128)
            real = transform[:, :129, :]
            imaginary = transform[:, 129:, :]
            encoded = torch.sqrt(real.square() + imaginary.square())
            for conv in self.encoder:
                encoded = functional.relu(conv(encoded))

            recurrent_input = encoded.squeeze(-1)
            gates = functional.linear(recurrent_input, self.rnn_weight_ih, self.rnn_bias_ih)
            gates = gates + functional.linear(state[:, 0, :], self.rnn_weight_hh, self.rnn_bias_hh)
            input_gate = torch.sigmoid(gates[:, 0:128])
            forget_gate = torch.sigmoid(gates[:, 128:256])
            candidate = torch.tanh(gates[:, 256:384])
            output_gate = torch.sigmoid(gates[:, 384:512])
            cell = forget_gate * state[:, 1, :] + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            probability = torch.sigmoid(self.output(functional.relu(hidden).unsqueeze(-1))).squeeze(-1)
            return probability, torch.stack((hidden, cell), dim=1)

    official = load_silero_vad()._model.eval()
    return official, Silero16kInferenceCore(official).eval()


def _validate_pytorch_core(official, core) -> None:  # noqa: ANN001
    """Prove the branch-free graph matches the official 16 kHz submodel."""
    import torch

    generator = torch.Generator().manual_seed(0)
    for batch_size in (1, 4, 17):
        audio = torch.randn((batch_size, 576), dtype=torch.float32, generator=generator)
        state = torch.randn((batch_size, 2, 128), dtype=torch.float32, generator=generator)
        with torch.inference_mode():
            expected_output, expected_state = official(audio, state.transpose(0, 1).contiguous())
            output, next_state = core(audio, state)
        torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            next_state,
            expected_state.transpose(0, 1).contiguous(),
            rtol=1e-6,
            atol=1e-7,
        )
    print("SILERO_PYTORCH_CORE_PARITY_PASSED")


def _export_silero_onnx(output_path: Path):  # noqa: ANN202
    """Export and validate a dynamic-batch, branch-free ONNX graph."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    official, core = _make_silero_core()
    _validate_pytorch_core(official, core)
    audio = torch.zeros((1, 576), dtype=torch.float32)
    state = torch.zeros((1, 2, 128), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            core,
            (audio, state),
            str(output_path),
            input_names=["input", "state"],
            output_names=["output", "stateN"],
            dynamic_axes={
                "input": {0: "batch"},
                "state": {0: "batch"},
                "output": {0: "batch"},
                "stateN": {0: "batch"},
            },
            opset_version=16,
            do_constant_folding=True,
            dynamo=False,
        )

    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    control_flow = [node.name for node in exported.graph.node if node.op_type in {"If", "Loop", "Scan"}]
    if control_flow:
        message = f"Silero TensorRT graph contains control flow: {control_flow}"
        raise RuntimeError(message)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(1)
    for batch_size in (1, 8):
        audio = torch.randn((batch_size, 576), dtype=torch.float32, generator=generator)
        state = torch.randn((batch_size, 2, 128), dtype=torch.float32, generator=generator)
        with torch.inference_mode():
            expected_output, expected_state = core(audio, state)
        output, next_state = session.run(None, {"input": audio.numpy(), "state": state.numpy()})
        np.testing.assert_allclose(output, expected_output.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(next_state, expected_state.numpy(), rtol=1e-4, atol=1e-5)
    print(f"SILERO_ONNX_EXPORT_PARITY_PASSED path={output_path}")
    return core


def _shape_for_batch(
    network_shape: Sequence[int],
    batch_size: int,
) -> tuple[int, ...]:
    """Resolve Silero's dynamic batch dimension for one profile point."""
    return tuple(batch_size if dimension == -1 else dimension for dimension in network_shape)


def _validate_tensorrt_engine(engine_path: Path, core, *, fp16: bool = False) -> None:  # noqa: ANN001
    """Execute recurrent TensorRT inference and compare it with the exact core."""
    import torch

    from nemo_curator.stages.audio.segmentation.silero_tensorrt import TensorRTSession

    rtol, atol = (5e-2, 5e-3) if fp16 else (3e-4, 3e-5)
    generator = torch.Generator().manual_seed(2)
    session = TensorRTSession(engine_path)
    try:
        for batch_size in (1, 8):
            expected_state = torch.zeros((batch_size, 2, 128), dtype=torch.float32)
            actual_state = expected_state.to(session.device)
            for _ in range(3):
                audio = torch.randn((batch_size, 576), dtype=torch.float32, generator=generator)
                with torch.inference_mode():
                    expected_output, expected_state = core(audio, expected_state)
                outputs = session.infer(
                    {
                        "input": audio.to(session.device),
                        "state": actual_state,
                    }
                )
                actual_output = outputs["output"].cpu()
                actual_state = outputs["stateN"]
                torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)
                torch.testing.assert_close(actual_state.cpu(), expected_state, rtol=rtol, atol=atol)
    finally:
        session.close()
    print("SILERO_TENSORRT_RECURRENT_PREFLIGHT_PASSED batches=1,8 steps=3")


def build_engine(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915
    try:
        import tensorrt as trt
    except ImportError as error:
        message = "TensorRT Python bindings are required to build the Silero engine"
        raise RuntimeError(message) from error

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.onnx_output) if args.onnx_output else output_path.with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    core = _export_silero_onnx(onnx_path)

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        message = f"Failed to parse {onnx_path}:\n{errors}"
        raise RuntimeError(message)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
    # TF32 recurrent error accumulates over long recordings and can shift VAD decisions.
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            message = "This GPU does not provide fast FP16 TensorRT kernels"
            raise RuntimeError(message)
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    dynamic_inputs = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        network_shape = tuple(tensor.shape)
        if -1 not in network_shape:
            continue
        minimum = _shape_for_batch(network_shape, args.min_batch)
        optimum = _shape_for_batch(network_shape, args.opt_batch)
        maximum = _shape_for_batch(network_shape, args.max_batch)
        shape_status = profile.set_shape(tensor.name, minimum, optimum, maximum)
        if shape_status is False:
            message = f"Could not set TensorRT profile for {tensor.name}: {minimum}/{optimum}/{maximum}"
            raise RuntimeError(message)
        dynamic_inputs.append((tensor.name, minimum, optimum, maximum))
    if dynamic_inputs:
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        message = "TensorRT failed to build the Silero engine"
        raise RuntimeError(message)
    output_path.write_bytes(serialized_engine)

    _validate_tensorrt_engine(output_path, core, fp16=args.fp16)

    precision = "fp16" if args.fp16 else "fp32"
    print(f"Wrote {precision} Silero TensorRT engine to {output_path}")
    for name, minimum, optimum, maximum in dynamic_inputs:
        print(f"  {name}: min={minimum} opt={optimum} max={maximum}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Destination .plan/.engine file")
    parser.add_argument(
        "--onnx-output",
        default=None,
        help="Destination for the generated branch-free ONNX graph (default: beside the engine)",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Build FP16 instead of the parity-oriented FP32 default")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    return args


if __name__ == "__main__":
    build_engine(parse_args())

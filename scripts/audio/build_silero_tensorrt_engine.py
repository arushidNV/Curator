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

"""Build a dynamic-batch TensorRT engine from the official Silero VAD ONNX graph."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _specialize_16khz_model(model_bytes: bytes) -> bytes:
    """Inline the official model's ``sr == 16000`` branch.

    Riva's TensorRT Silero engine is fixed at 16 kHz and therefore has only
    ``input`` and ``state`` inputs. Removing the top-level ONNX ``If`` avoids
    TensorRT control-flow parsing and eliminates a scalar host input per call.
    """
    import onnx
    from onnx import helper

    model = onnx.load_model_from_string(model_bytes)
    if_nodes = [node for node in model.graph.node if node.op_type == "If"]
    input_names = {value.name for value in model.graph.input}
    if len(if_nodes) != 1 or "sr" not in input_names:
        message = "Expected the official Silero graph with one sample-rate If node"
        raise ValueError(message)

    branch = next(attribute.g for attribute in if_nodes[0].attribute if attribute.name == "then_branch")
    outputs = list(model.graph.output)
    if len(branch.output) != len(outputs):
        message = "Silero 16 kHz branch output count does not match the wrapper graph"
        raise ValueError(message)

    nodes = [copy.deepcopy(node) for node in branch.node]
    for branch_output, graph_output in zip(branch.output, outputs, strict=True):
        nodes.append(
            helper.make_node(
                "Identity",
                inputs=[branch_output.name],
                outputs=[graph_output.name],
                name=f"Expose_{graph_output.name}",
            )
        )

    graph = helper.make_graph(
        nodes,
        "silero_vad_16khz_tensorrt",
        [copy.deepcopy(value) for value in model.graph.input if value.name != "sr"],
        [copy.deepcopy(value) for value in outputs],
        initializer=[copy.deepcopy(value) for value in branch.initializer],
        value_info=[copy.deepcopy(value) for value in branch.value_info],
    )
    specialized = helper.make_model(
        graph,
        opset_imports=[copy.deepcopy(value) for value in model.opset_import],
        producer_name="nemo-curator-silero-tensorrt",
    )
    specialized.ir_version = model.ir_version
    onnx.checker.check_model(specialized)
    print("SILERO_16KHZ_SPECIALIZATION_PASSED inputs=input,state")
    return specialized.SerializeToString()


def _shape_for_batch(
    input_name: str,
    network_shape: Sequence[int],
    batch_size: int,
    input_samples: int,
) -> tuple[int, ...]:
    """Resolve Silero's dynamic audio/state dimensions for one profile point."""
    shape = list(network_shape)
    for index, dimension in enumerate(shape):
        if dimension != -1:
            continue
        if (input_name == "state" and index == 1) or index == 0:
            shape[index] = batch_size
        else:
            shape[index] = input_samples
    return tuple(shape)


def build_engine(args: argparse.Namespace) -> None:  # noqa: C901
    try:
        import tensorrt as trt
    except ImportError as error:
        message = "TensorRT Python bindings are required to build the Silero engine"
        raise RuntimeError(message) from error

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        message = f"Silero ONNX model not found: {onnx_path}"
        raise FileNotFoundError(message)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    specialized_model = _specialize_16khz_model(onnx_path.read_bytes())
    if not parser.parse(specialized_model):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        message = f"Failed to parse {onnx_path}:\n{errors}"
        raise RuntimeError(message)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
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
        minimum = _shape_for_batch(tensor.name, network_shape, args.min_batch, args.input_samples)
        optimum = _shape_for_batch(tensor.name, network_shape, args.opt_batch, args.input_samples)
        maximum = _shape_for_batch(tensor.name, network_shape, args.max_batch, args.input_samples)
        if not profile.set_shape(tensor.name, minimum, optimum, maximum):
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

    precision = "fp16" if args.fp16 else "fp32"
    print(f"Wrote {precision} Silero TensorRT engine to {output_path}")
    for name, minimum, optimum, maximum in dynamic_inputs:
        print(f"  {name}: min={minimum} opt={optimum} max={maximum}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Path to the official Silero VAD ONNX model")
    parser.add_argument("--output", required=True, help="Destination .plan/.engine file")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument(
        "--input-samples",
        type=int,
        default=576,
        help="512-sample Silero window plus its 64-sample recurrent context",
    )
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Build FP16 instead of the parity-oriented FP32 default")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    return args


if __name__ == "__main__":
    build_engine(parse_args())

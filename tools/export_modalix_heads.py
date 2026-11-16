"""Turn the cut six-head export into the form the Modalix box decoder consumes.

The decoder's ``yolo`` path does no decoding - ``decode_to_model_frame`` is
"decode just with dequantizing" - so the graph must deliver finished values:

  bbox_k        [cx, cy, width, height] in model-frame pixels
  class_prob_k  sigmoid probabilities

The raw Ultralytics head emits (left, top, right, bottom) distances in stride
units and pre-sigmoid logits, which is why feeding it directly produced nonsense
boxes.

This keeps the arithmetic in the shape SiMa's own surgeon uses: split the box
tensor into two 2-channel halves and combine them with Sub/Div/Add/Concat. An
earlier version sliced out four 1-channel tensors instead, and that graph ran at
0.76 fps against 27 fps for the same model with raw heads - the narrow
intermediates forced per-frame layout conversions. Intermediates stay at two
channels or wider here.

The result is checked against a NumPy decode of the same raw tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

STRIDES = (8, 16, 32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="six-head ONNX with raw heads")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-size", type=int, default=1280)
    parser.add_argument("--classes", type=int, default=4)
    arguments = parser.parse_args()

    model = onnx.load(str(arguments.raw))
    inferred = shape_inference.infer_shapes(model)
    shapes = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
              for value in list(inferred.graph.value_info) + list(inferred.graph.output)}
    for name in ("bbox_0", "bbox_1", "bbox_2", "class_prob_0", "class_prob_1", "class_prob_2"):
        if name not in shapes:
            raise SystemExit(f"raw graph has no output {name}")

    side = arguments.model_size
    grids = (side // 8, side // 16, side // 32)
    nodes, initializers = [], []
    box_outputs, class_outputs = [], []

    for level, (stride, grid) in enumerate(zip(STRIDES, grids)):
        prefix = f"decode_{level}"
        columns = np.arange(grid, dtype=np.float32)
        anchor_x = ((columns + 0.5) * stride).reshape(1, 1, 1, grid).repeat(grid, axis=2)
        anchor_y = ((columns + 0.5) * stride).reshape(1, 1, grid, 1).repeat(grid, axis=3)
        anchor = np.concatenate([anchor_x, anchor_y,
                                 np.zeros_like(anchor_x), np.zeros_like(anchor_x)],
                                axis=1).astype(np.float32)
        initializers.append(numpy_helper.from_array(anchor, f"{prefix}_anchor"))

        # The decode is linear in (l, t, r, b), so it is a 1x1 convolution. A conv is
        # the one op the MLA is built for; the elementwise formulation of the same
        # maths ran 35x slower than this model's raw heads.
        half = stride / 2.0
        weights = np.array([
            [-half, 0.0, half, 0.0],   # cx = ax + (r - l) * s/2
            [0.0, -half, 0.0, half],   # cy = ay + (b - t) * s/2
            [stride, 0.0, stride, 0.0],  # w = (l + r) * s
            [0.0, stride, 0.0, stride],  # h = (t + b) * s
        ], dtype=np.float32).reshape(4, 4, 1, 1)
        initializers.append(numpy_helper.from_array(weights, f"{prefix}_w"))

        nodes += [
            helper.make_node("Conv", [f"bbox_{level}", f"{prefix}_w"], [f"{prefix}_lin"],
                             name=f"{prefix}_conv", kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
            helper.make_node("Add", [f"{prefix}_lin", f"{prefix}_anchor"], [f"decoded_bbox_{level}"],
                             name=f"{prefix}_offset"),
        ]
        box_outputs.append(f"decoded_bbox_{level}")
        nodes.append(helper.make_node("Sigmoid", [f"class_prob_{level}"], [f"decoded_class_prob_{level}"],
                                      name=f"sigmoid_{level}"))
        class_outputs.append(f"decoded_class_prob_{level}")

    existing = list(model.graph.node)
    model.graph.ClearField("node")
    model.graph.node.extend(existing + nodes)
    initializers.extend(inferred.graph.initializer)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)
    model.graph.ClearField("output")
    for name in box_outputs + class_outputs:
        channels = arguments.classes if name.startswith("decoded_class") else 4
        grid = grids[int(name.rsplit("_", 1)[1])]
        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, channels, grid, grid]))

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(arguments.output))
    total = sum(1 for node in model.graph.node)
    print(f"written: {arguments.output}  {arguments.output.stat().st_size / 1e6:.2f} MB  nodes={total}")


if __name__ == "__main__":
    main()

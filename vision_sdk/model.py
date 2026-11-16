"""Turn a trained YOLO26s ONNX graph into the six heads used by Modalix."""

from __future__ import annotations

from pathlib import Path


class ModelExportError(RuntimeError):
    pass


def _discover_head(graph, shapes, token: str, channels: int, side: int) -> str | None:
    """Find a head conv by name token and output shape.

    The exact node path is pinned first; this fallback survives graph renames
    and layer moves as long as the cv2/cv3 token and the shape survive.
    """
    for node in graph.graph.node:
        if node.op_type != "Conv" or len(node.output) != 1 or token not in node.name:
            continue
        if shapes.get(node.output[0]) == [1, channels, side, side]:
            return node.output[0]
    return None


def export_six_heads(source: Path, destination: Path, class_count: int, box_channels: int = 4) -> Path:
    """Expose the three box and three class heads without the host-side decoder.

    The graph family is checked by node names and output shapes. An incompatible
    Ultralytics export fails explicitly instead of producing a bad board model.
    """
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as error:
        raise ModelExportError("Install the training extras to export ONNX models") from error
    graph = onnx.load(str(source))
    inferred = shape_inference.infer_shapes(graph)
    shapes = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        shapes[value.name] = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
    # The grids follow from the exported input side, so a 640 export and a 1280
    # export both work without a second hardcoded set of strides.
    input_shape = [dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim]
    input_side = input_shape[2] if len(input_shape) == 4 and input_shape[2] > 0 else 640
    sides = (input_side // 8, input_side // 16, input_side // 32)
    selected = []
    for kind, prefix, channels in (("bbox", "cv2", box_channels), ("class_prob", "cv3", class_count)):
        for level, side in enumerate(sides):
            name = f"/model.23/{prefix}.{level}/{prefix}.{level}.2/Conv"
            matches = [node for node in graph.graph.node if node.name == name]
            output_name = matches[0].output[0] if len(matches) == 1 else None
            if output_name is None:
                output_name = _discover_head(graph, shapes, prefix, channels, side)
            if output_name is None:
                raise ModelExportError(
                    f"graph does not expose {kind} level {level} ({side}x{side}, "
                    f"{channels} channels); export format changed"
                )
            if shapes.get(output_name) != [1, channels, side, side]:
                raise ModelExportError(f"Unexpected {kind} shape: {shapes.get(output_name)}")
            selected.append((output_name, f"{kind}_{level}"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".extract.onnx")
    onnx.utils.extract_model(str(source), str(temporary), [graph.graph.input[0].name], [name for name, _ in selected])
    extracted = onnx.load(str(temporary))
    for current, desired in selected:
        for node in extracted.graph.node:
            node.input[:] = [desired if name == current else name for name in node.input]
            node.output[:] = [desired if name == current else name for name in node.output]
        for value in extracted.graph.output:
            if value.name == current:
                value.name = desired
    onnx.checker.check_model(extracted)
    onnx.save(extracted, str(destination))
    temporary.unlink()
    return destination

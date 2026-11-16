"""Export the trained checkpoint to ONNX and inspect what the graph actually exposes."""

from pathlib import Path

import onnx
from onnx import shape_inference
from ultralytics import YOLO
from analysis_paths import DATA

RUN = DATA / "runs" / "spike-v3-prod-e120" / "weights" / "best.pt"
OUT = DATA / "export"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(RUN))
    exported = model.export(format="onnx", imgsz=640, dynamic=False, simplify=False, opset=17)
    source = Path(exported)
    target = OUT / "model.onnx"
    target.write_bytes(source.read_bytes())
    print("exported:", source)
    print("size    :", f"{target.stat().st_size / 1e6:.1f} MB")

    graph = onnx.load(str(target))
    inferred = shape_inference.infer_shapes(graph)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    print("\ninputs:", [(i.name, shapes.get(i.name)) for i in graph.graph.input])
    print("outputs:", [(o.name, shapes.get(o.name)) for o in graph.graph.output])
    print("node count:", len(graph.graph.node), "| opset:", graph.opset_import[0].version)
    print("\nhead-like convs (cv2/cv3 tokens or 80/40/20 grids):")
    for node in graph.graph.node:
        if node.op_type != "Conv":
            continue
        shape = shapes.get(node.output[0])
        if not shape or len(shape) != 4:
            continue
        if any(token in node.name for token in ("cv2", "cv3", "one2one", "cv4", "cv5")) or shape[2] in (80, 40, 20, 84):
            print(f"  {node.name:<45} -> {node.output[0]:<30} {shape}")
    print("\nlast 12 nodes:")
    for node in graph.graph.node[-12:]:
        print(f"  {node.op_type:<12} {node.name:<42} out={list(node.output)}")


if __name__ == "__main__":
    main()

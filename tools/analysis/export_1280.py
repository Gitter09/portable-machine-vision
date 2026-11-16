"""Export the 1280 checkpoint to ONNX and split it into the six Modalix heads."""

import sys
from pathlib import Path

import onnx
from ultralytics import YOLO

TRAIN_ROOT = DATA
REPO = Path(r"C:\Users\DELL\Desktop\d-meter\demeter-systems-mv-full")
sys.path.insert(0, str(REPO))

from vision_sdk.model import export_six_heads  # noqa: E402
from analysis_paths import DATA


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "spike-v4-1280-e120"
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    weights = TRAIN_ROOT / "runs" / run / "weights" / "best.pt"
    out = TRAIN_ROOT / "export-1280"
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False,
                                 simplify=False, opset=17))
    fused = out / "model_1280.onnx"
    fused.write_bytes(exported.read_bytes())
    six = export_six_heads(fused, out / "model_1280_sima_6out.onnx", 4)

    for path in (fused, six):
        graph = onnx.load(str(path))
        shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                  for v in graph.graph.output}
        print(f"\n{path.name}  {path.stat().st_size / 1e6:.2f} MB")
        print("  input :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                            for i in graph.graph.input])
        for name in sorted(shapes):
            print(f"  {name:<16} {shapes[name]}")


if __name__ == "__main__":
    main()

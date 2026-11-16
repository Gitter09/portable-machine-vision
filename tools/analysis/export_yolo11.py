"""Export the trained YOLO11 checkpoint as the six heads the Modalix decoder reads.

The DFL box branch already matches what the decoder's ``yolov8`` path expects: 64
channels (4 x reg_max 16) at 80/40/20, decoded by the hardware. The class branch
must arrive already sigmoided, because that path hard-codes
``sigmoid_on_probabilities = false`` and the flag is not a config key.

Ultralytics applies its sigmoid after concatenating all three levels, so
per-level post-sigmoid tensors do not exist in the export. Three small Sigmoid
nodes are appended instead - far less than a full decode, and the only way to
meet the contract.
"""

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from ultralytics import YOLO

TRAIN_ROOT = DATA
REPO = Path(r"C:\Users\DELL\Desktop\d-meter\demeter-systems-mv-full")
sys.path.insert(0, str(REPO))

from vision_sdk.model import export_six_heads  # noqa: E402
from analysis_paths import DATA


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "train-yolo11n-1280"
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    classes = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    out = TRAIN_ROOT / "export-yolo11"
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(TRAIN_ROOT / "runs" / run / "weights" / "best.pt"))
    exported = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False,
                                 simplify=False, opset=17))
    fused = out / "yolo11n_fused.onnx"
    fused.write_bytes(exported.read_bytes())

    six = export_six_heads(fused, out / "yolo11n_six.onnx", classes, box_channels=64)
    graph = onnx.load(str(six))
    for level in range(3):
        graph.graph.node.append(helper.make_node(
            "Sigmoid", [f"class_prob_{level}"], [f"class_sigmoid_{level}"],
            name=f"sigmoid_{level}"))
    names = {output.name for output in graph.graph.output}
    for output in graph.graph.output:
        if output.name.startswith("class_prob_"):
            output.name = output.name.replace("class_prob_", "class_sigmoid_")
    final = out / "yolo11n_modalix_6out.onnx"
    onnx.checker.check_model(graph)
    onnx.save(graph, str(final))

    check = onnx.load(str(final))
    print("input :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in check.graph.input])
    for output in check.graph.output:
        print("  ", output.name, [d.dim_value for d in output.type.tensor_type.shape.dim])
    print("written:", final, f"{final.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

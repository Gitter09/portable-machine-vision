"""Feasibility spike: can yolo26n learn these parts at a given transform scale?

Not an acceptance run. ~84 training images, no augmentation, the validation split
taken from held-out blocks of the same two runs. The only question it answers is
whether the four-class contract is representable at this pixel size.
"""

import sys
from pathlib import Path

from ultralytics import YOLO
from analysis_paths import DATA

TRAIN_ROOT = DATA
WEIGHTS = Path(r"C:\Users\DELL\Desktop\d-meter\demeter-systems-mv-full\weights\yolo26n.pt")


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dataset = sys.argv[2] if len(sys.argv) > 2 else "v1-crop"
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    name = f"spike-{dataset}-e{epochs}"
    data = str(TRAIN_ROOT / "dataset" / dataset / "data.yaml")

    model = YOLO(str(WEIGHTS))
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=0,
        seed=42,
        deterministic=True,
        project=str(TRAIN_ROOT / "runs"),
        name=name,
        exist_ok=True,
        # The plan's baseline: capture real rotations, synthesise none of them.
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.1,
        hsv_v=0.15,
        amp=True,
        plots=True,
        verbose=True,
    )

    trained = YOLO(str(TRAIN_ROOT / "runs" / name / "weights" / "best.pt"))
    metrics = trained.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=0,
        plots=True,
        project=str(TRAIN_ROOT / "runs"),
        name=name + "-val",
        exist_ok=True,
    )
    print("\n=== per-class validation at the framework default operating point ===")
    for index, class_name in enumerate(metrics.names.values()):
        if index >= len(metrics.box.p):
            print(f"{class_name:<15} no examples in validation")
            continue
        precision, recall, ap50, ap = (float(value) for value in metrics.box.class_result(index))
        print(f"{class_name:<15} P={precision:.3f} R={recall:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")
    print(f"\noverall mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()

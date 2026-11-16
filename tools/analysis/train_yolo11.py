"""Train yolo11n at native 1280 resolution on the IC/terminal dataset."""

from pathlib import Path

from ultralytics import YOLO
from analysis_paths import DATA

TRAIN_ROOT = DATA


def main() -> None:
    model = YOLO(str(TRAIN_ROOT / "yolo11n.pt"))
    model.train(
        data=str(TRAIN_ROOT / "dataset" / "v4-1280" / "data.yaml"),
        epochs=150,
        patience=30,
        imgsz=1280,
        batch=8,
        device=0,
        workers=0,
        seed=42,
        deterministic=True,
        project=str(TRAIN_ROOT / "runs"),
        name="train-yolo11n-1280",
        exist_ok=True,
        # Real captured rotations only; no synthetic geometry.
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

    best = TRAIN_ROOT / "runs" / "train-yolo11n-1280" / "weights" / "best.pt"
    trained = YOLO(str(best))
    metrics = trained.val(
        data=str(TRAIN_ROOT / "dataset" / "v4-1280" / "data.yaml"),
        imgsz=1280, batch=8, device=0, workers=0, plots=True,
        project=str(TRAIN_ROOT / "runs"), name="train-yolo11n-1280-val", exist_ok=True,
    )
    print("\n=== per-class validation ===")
    for index, class_name in enumerate(metrics.names.values()):
        if index >= len(metrics.box.p):
            print(f"{class_name:<15} no examples")
            continue
        precision, recall, ap50, ap = (float(v) for v in metrics.box.class_result(index))
        print(f"{class_name:<15} P={precision:.3f} R={recall:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")
    print(f"\noverall mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()

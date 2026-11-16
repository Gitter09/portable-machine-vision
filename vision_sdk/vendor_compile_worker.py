"""Runs inside the provisioned SiMa ModelSDK container, not on the host PC."""

from pathlib import Path
import os
import tarfile

import cv2
import numpy as np
from afe.apis.defines import BiasCorrectionType, QuantizationParams, default_calibration, gen2_target, quantization_scheme
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.defines import InputName
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from sima_utils.data.data_generator import DataGenerator


ROOT = Path("/work")
MODEL = ROOT / "model_sima_6out.onnx"
CALIBRATION = ROOT / "calibration"
OUTPUT = ROOT / "compiled"

#: Square model input side. 640 was the only size this worker supported; the
#: caller sets it so a higher-resolution build does not need a second script.
INPUT_SIDE = int(os.environ.get("SIMA_INPUT_SIDE", "640"))


def read_input(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (INPUT_SIDE, INPUT_SIDE, 3):
        raise ValueError(f"Calibration image must be {INPUT_SIDE} × {INPUT_SIDE}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)


def main():
    paths = sorted(CALIBRATION.glob("*.png"))
    if len(paths) < 50:
        raise ValueError(f"At least 50 prepared calibration images are needed; found {len(paths)}")
    name = InputName("images")
    generator = DataGenerator({name: [str(path) for path in paths]})
    generator.map({name: read_input})
    calibration = convert_data_generator_to_iterable(generator, length_limit=len(paths))
    source = onnx_source(str(MODEL), {"images": (1, 3, INPUT_SIDE, INPUT_SIDE)}, {"images": ScalarType.float32})
    model = load_model(source, target=gen2_target)
    settings = QuantizationParams(
        calibration_method=default_calibration(),
        activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
        weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        requantization_mode="sima", node_names={""}, custom_quantization_configs=None,
        biascorr_type=BiasCorrectionType.NONE, channel_equalization=False, smooth_quant=False,
    )
    network = model.quantize(calibration_data=calibration, quantization_config=settings,
                             model_name="yolov8", arm_only=False, any_shape_on_mla=True,
                             automatic_layout_conversion=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    network.compile(output_path=str(OUTPUT), batch_size=1, compress=False)
    for archive in OUTPUT.glob("*_mpk.tar.gz"):
        with tarfile.open(archive, "r:gz") as package:
            for member in package.getmembers():
                name = Path(member.name).name
                if member.isfile() and name in {"yolov8_stage1_mla.elf", "yolov8_mpk.json"}:
                    with package.extractfile(member) as source:
                        (OUTPUT / name).write_bytes(source.read())
    if not (OUTPUT / "yolov8_stage1_mla.elf").is_file() or not (OUTPUT / "yolov8_mpk.json").is_file():
        raise RuntimeError("SiMa compiler did not produce the expected board artifacts")


if __name__ == "__main__":
    main()

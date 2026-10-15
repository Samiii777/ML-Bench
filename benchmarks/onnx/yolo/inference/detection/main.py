#!/usr/bin/env python3
"""ONNX Runtime object detection with Ultralytics YOLO (exported on first use).

Exported with NMS inside the graph (output: [batch, 300, 6] = x1,y1,x2,y2,score,class),
so an iteration covers the same work as the PyTorch predict benchmark.
"""

import shutil
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "pytorch" / "yolo" / "inference" / "detection"))

import numpy as np

from core.args import add_onnx_args, build_base_parser
from core.harness import OnnxHarness, run_main
from main import COCO_DOG, IMGSZ, load_yolo, sample_batch  # shared with the PyTorch YOLO benchmark


class OnnxYolo(OnnxHarness):
    use_case = "detection"

    def get_onnx_model_path(self):
        path = ROOT / "models" / "onnx" / f"{self.args.model}_{self.precision}.onnx"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            half = self.precision == "fp16"
            exported = load_yolo(self.args.model).export(format="onnx", imgsz=IMGSZ, dynamic=True, nms=True,
                                                         half=half, device=0 if half else "cpu")
            shutil.move(exported, path)
        return str(path)

    def prepare_numpy_inputs(self):
        batch = sample_batch(self.batch_size)
        return {self._model.get_inputs()[0].name: batch.astype(np.float16 if self.precision == "fp16" else np.float32)}

    def validate_result(self, session, inputs, outputs, validator):
        dets = outputs[0][0]  # image 0: (300, 6)
        dog = dets[(dets[:, 5].astype(int) == COCO_DOG)]
        conf = float(dog[:, 4].max()) if len(dog) else 0.0
        print(f"Best dog confidence: {conf:.2f}")
        validator.expect_greater_than("dog_confidence", conf, 0.5)

    def get_config(self):
        return {**super().get_config(), "nms": "in-graph"}


if __name__ == "__main__":
    parser = build_base_parser("ONNX Runtime YOLO detection")
    add_onnx_args(parser)
    parser.set_defaults(model="yolo11s", num_warmup=20, num_runs=50)  # Ultralytics/cuDNN settle after ~10 calls
    run_main(OnnxYolo, parser)

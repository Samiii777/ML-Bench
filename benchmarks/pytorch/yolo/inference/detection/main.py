#!/usr/bin/env python3
"""Object detection inference with Ultralytics YOLO (YOLO11 by default).

One iteration = Ultralytics predict on a 640x640 batch: network + NMS +
result objects, i.e. what an application actually waits for. Input is the
real data/dog.jpg, and a dog must be detected for the run to PASS.
"""

import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from core.args import build_base_parser
from core.harness import InferenceHarness, SkipBenchmark, run_main
from core.schema import MetricEntry
from utils.download import get_sample_image_path

IMGSZ = 640
COCO_DOG = 16


def load_yolo(name: str):
    from ultralytics import YOLO
    weights = ROOT / "models" / f"{name}.pt"
    weights.parent.mkdir(exist_ok=True)
    return YOLO(str(weights))  # Ultralytics downloads official weights to this path if missing


def sample_batch(batch_size: int) -> np.ndarray:
    """(B, 3, 640, 640) float32 in [0, 1] from data/dog.jpg."""
    img = Image.open(get_sample_image_path()).convert("RGB").resize((IMGSZ, IMGSZ))
    chw = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(np.repeat(chw[None], batch_size, axis=0))


class YoloDetection(InferenceHarness):
    use_case = "detection"

    def load_model(self):
        if self.precision not in ("fp32", "fp16"):
            raise SkipBenchmark("Ultralytics predict supports fp32 and fp16 only")
        return load_yolo(self.args.model)

    def prepare_inputs(self):
        return torch.from_numpy(sample_batch(self.batch_size)).to(self.device)

    def run_step(self, model, inputs):
        return model.predict(inputs, imgsz=IMGSZ, half=self.precision == "fp16", device=self.device,
                             verbose=False)

    def get_extra_metrics(self, model, inputs, outputs):
        boxes = outputs[0].boxes
        self._dog_conf = max((float(c) for c, k in zip(boxes.conf, boxes.cls) if int(k) == COCO_DOG), default=0.0)
        print(f"Detections in image 0: {len(boxes)}, best dog confidence {self._dog_conf:.2f}")
        return [MetricEntry("detections_per_image", float(len(boxes)), "count", "higher_is_better")]

    def validate_result(self, model, inputs, outputs, validator):
        validator.expect_greater_than("dog_confidence", self._dog_conf, 0.5)

    def get_config(self):
        return {"imgsz": IMGSZ, "pipeline": "ultralytics predict (incl. NMS)"}


if __name__ == "__main__":
    parser = build_base_parser("Ultralytics YOLO detection inference")
    parser.set_defaults(model="yolo11s", num_warmup=20, num_runs=50)  # Ultralytics/cuDNN settle after ~10 calls
    run_main(YoloDetection, parser)

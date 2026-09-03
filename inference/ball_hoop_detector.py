#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Standalone inference wrapper around the fine-tuned ball/hoop YOLOX-s
checkpoint. Deliberately self-contained (only needs YOLOX + torch + opencv +
numpy) so it can be copied into the main app's pipeline once a checkpoint is
finalized -- this repo only trains, the app runs elsewhere.

Unlike YOLOX's own trainer (CUDA-only), this wrapper is device-generic: it
picks CUDA > MPS > CPU automatically, so the exact same code runs on this
training box and on the Mac where the rest of the pipeline lives.

Usage:
    from ball_hoop_detector import BallHoopDetector

    detector = BallHoopDetector(ckpt_path="YOLOX_outputs/yolox_s_ball_hoop_full/best_ckpt.pth")
    detections = detector.detect(image)  # image: BGR np.ndarray (as from cv2.imread)
    for d in detections:
        print(d.class_name, d.confidence, d.bbox)

CLI smoke test:
    venv/bin/python inference/ball_hoop_detector.py --ckpt <path> --image <path>
"""

import argparse
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "YOLOX"))

from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

DEFAULT_EXP_FILE = os.path.join(
    os.path.dirname(__file__), "..", "training", "yolox_s_ball_hoop_full.py"
)

# Category id -> name, matching the Roboflow export's COCO categories
# (id range 0-6, same order the model's num_classes=7 head was trained with).
CLASS_NAMES = ["player", "Ball", "FG Attempt", "FG Made", "Hoop", "Player", "Ref"]


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in original image pixel coordinates


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BallHoopDetector:
    def __init__(
        self,
        ckpt_path: str,
        exp_file: str = DEFAULT_EXP_FILE,
        device: torch.device = None,
        conf_thresh: float = None,
        nms_thresh: float = None,
    ):
        self.device = device or pick_device()

        exp = get_exp(exp_file)
        self.num_classes = exp.num_classes
        self.test_size = exp.test_size
        self.conf_thresh = conf_thresh if conf_thresh is not None else exp.test_conf
        self.nms_thresh = nms_thresh if nms_thresh is not None else exp.nmsthre

        model = exp.get_model()
        # weights_only=False: these are our own self-generated checkpoints (not
        # third-party downloads), and post-eval checkpoints embed a numpy AP
        # scalar that torch's default weights_only=True unpickler rejects.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model.to(self.device)

        self.preproc = ValTransform(legacy=False)

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> list:
        """Run detection on a single BGR image (as returned by cv2.imread)."""
        height, width = image.shape[:2]
        ratio = min(self.test_size[0] / height, self.test_size[1] / width)

        img, _ = self.preproc(image, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        outputs = self.model(img)
        outputs = postprocess(
            outputs, self.num_classes, self.conf_thresh, self.nms_thresh, class_agnostic=True
        )[0]

        if outputs is None:
            return []

        outputs = outputs.cpu()
        boxes = outputs[:, 0:4] / ratio
        scores = outputs[:, 4] * outputs[:, 5]
        classes = outputs[:, 6].long()

        return [
            Detection(
                class_name=CLASS_NAMES[cls],
                confidence=float(score),
                bbox=tuple(float(v) for v in box),
            )
            for box, score, cls in zip(boxes.tolist(), scores.tolist(), classes.tolist())
        ]


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--exp-file", default=DEFAULT_EXP_FILE)
    parser.add_argument("--conf", type=float, default=None)
    args = parser.parse_args()

    detector = BallHoopDetector(args.ckpt, exp_file=args.exp_file, conf_thresh=args.conf)
    print(f"device: {detector.device}")

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    detections = detector.detect(image)
    print(f"{len(detections)} detections:")
    for d in detections:
        print(f"  {d.class_name:12s} conf={d.confidence:.3f} bbox={d.bbox}")


if __name__ == "__main__":
    _main()

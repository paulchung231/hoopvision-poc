#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Run BallHoopDetector over every frame of a video and write an annotated copy
with detection boxes drawn, for visual review.

Usage:
    venv/bin/python inference/annotate_video.py \
        --video /path/to/clip.mp4 --ckpt <ckpt.pth> --out /path/to/out.mp4
"""

import argparse
import time

import cv2
import torch

from ball_hoop_detector import BallHoopDetector

COLORS = {
    "Ball": (0, 165, 255),
    "Hoop": (255, 0, 0),
    "player": (0, 255, 0),
    "Player": (0, 255, 0),
    "Ref": (255, 255, 0),
    "FG Attempt": (0, 0, 255),
    "FG Made": (255, 0, 255),
}


def draw(image, detections):
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = COLORS.get(d.class_name, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true", help="force CPU (avoid competing with training on GPU)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else None
    detector = BallHoopDetector(args.ckpt, device=device, conf_thresh=args.conf)
    print(f"device: {detector.device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {total_frames} frames @ {fps:.1f}fps, {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_idx = 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        writer.write(draw(frame, detections))
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total_frames} frames processed")

    cap.release()
    writer.release()
    elapsed = time.time() - t_start
    print(f"done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps) -> {args.out}")


if __name__ == "__main__":
    main()

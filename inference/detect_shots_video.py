#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Run BallHoopDetector + ShotDetector over a full video: draws detection boxes
plus a "SHOT MADE" flash whenever the geometric shot detector fires, and
prints a timestamped event log.

Usage:
    venv/bin/python inference/detect_shots_video.py \
        --video /path/to/clip.mp4 --ckpt <ckpt.pth> --out /path/to/out.mp4
"""

import argparse
import time

import cv2
import torch

from ball_hoop_detector import BallHoopDetector
from shot_detector import ShotDetector

COLORS = {
    "Ball": (0, 165, 255),
    "Hoop": (255, 0, 0),
    "player": (0, 255, 0),
    "Player": (0, 255, 0),
    "Ref": (255, 255, 0),
    "FG Attempt": (0, 0, 255),
    "FG Made": (255, 0, 255),
}

FLASH_FRAMES = 15  # how long the "SHOT MADE" banner stays on screen


def draw_detections(image, detections):
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = COLORS.get(d.class_name, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_made_banner(image):
    h, w = image.shape[:2]
    text = "SHOT MADE"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
    x, y = (w - tw) // 2, 80
    cv2.rectangle(image, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0, 200, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else None
    detector = BallHoopDetector(args.ckpt, device=device, conf_thresh=args.conf)
    shot_detector = ShotDetector()
    print(f"device: {detector.device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {fps:.1f}fps, {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_idx = 0
    flash_until = -1
    events = []
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        new_events = shot_detector.update(frame_idx, detections)
        for e in new_events:
            events.append(e)
            flash_until = frame_idx + FLASH_FRAMES
            ts = frame_idx / fps
            print(f"  MADE at frame {frame_idx} (t={ts:.2f}s), hoop center={e.hoop_center}")

        annotated = draw_detections(frame, detections)
        if frame_idx <= flash_until:
            draw_made_banner(annotated)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx} frames processed")

    cap.release()
    writer.release()
    elapsed = time.time() - t_start
    print(f"\ndone: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps) -> {args.out}")
    print(f"total shots detected: {len(events)}")


if __name__ == "__main__":
    main()

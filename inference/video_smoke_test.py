#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Quick pipeline smoke test: run BallHoopDetector over sampled frames of a real
video and dump annotated frames + timing, so we can eyeball whether the
video -> frame -> detect -> draw plumbing works end to end (not a quality
benchmark -- the checkpoint used here is still mid-training).

Usage:
    venv/bin/python inference/video_smoke_test.py \
        --video /path/to/clip.mp4 --ckpt <ckpt.pth> --out-dir <dir> \
        --every-n-frames 15 --max-samples 12
"""

import argparse
import os
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
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--every-n-frames", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true", help="force CPU (avoid competing with training on GPU)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu") if args.cpu else None
    detector = BallHoopDetector(args.ckpt, device=device, conf_thresh=args.conf)
    print(f"device: {detector.device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {total_frames} frames @ {fps:.1f}fps")

    frame_idx = 0
    saved = 0
    infer_times = []
    class_counts = {}

    while saved < args.max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every_n_frames == 0:
            t0 = time.time()
            detections = detector.detect(frame)
            infer_times.append(time.time() - t0)

            for d in detections:
                class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

            annotated = draw(frame, detections)
            out_path = os.path.join(args.out_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"frame {frame_idx}: {len(detections)} detections -> {out_path}")
            saved += 1
        frame_idx += 1

    cap.release()

    if infer_times:
        avg = sum(infer_times) / len(infer_times)
        print(f"\navg inference time: {avg:.3f}s/frame ({1/avg:.1f} fps) on {detector.device}")
    print(f"class counts across sampled frames: {class_counts}")


if __name__ == "__main__":
    main()

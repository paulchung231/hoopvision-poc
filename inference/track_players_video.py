#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Run BallHoopDetector + PlayerTracker over a full video: draws a stable,
color-coded box + name/track-ID per player (so identity persistence through
occlusion is visible by eye), and saves one representative crop per track
into --gallery-dir for manual tagging.

Manual-tagging workflow:
  1. Run once without --tags. Look at <gallery-dir>/tags_template.json
     (one entry per discovered track, value "") and the crop images next to
     it (track_NNN.jpg) to see who's who.
  2. Fill in names, save as e.g. tags.json.
  3. Re-run with --tags tags.json -- the output video now labels each player
     by name instead of by track number.

Usage:
    venv/bin/python inference/track_players_video.py \
        --video /path/to/clip.mp4 --ckpt <ckpt.pth> --out /path/to/out.mp4 \
        --gallery-dir /path/to/gallery [--tags /path/to/tags.json]
"""

import argparse
import json
import os
import time

import cv2
import torch

from ball_hoop_detector import BallHoopDetector
from player_tracker import PlayerTracker

TRACK_COLOR_PALETTE = [
    (66, 135, 245), (245, 66, 135), (66, 245, 156), (245, 191, 66),
    (156, 66, 245), (66, 245, 245), (245, 66, 66), (144, 238, 144),
]


def color_for(track_id):
    return TRACK_COLOR_PALETTE[track_id % len(TRACK_COLOR_PALETTE)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gallery-dir", required=True)
    parser.add_argument("--tags", default=None, help="JSON file mapping track_id (string) -> player name")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    tags = {}
    if args.tags:
        with open(args.tags) as f:
            tags = json.load(f)

    os.makedirs(args.gallery_dir, exist_ok=True)
    device = torch.device("cpu") if args.cpu else None
    detector = BallHoopDetector(args.ckpt, device=device, conf_thresh=args.conf)
    tracker = PlayerTracker()
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

    seen_track_ids = set()
    track_frame_counts = {}
    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        tracked = tracker.update(frame_idx, detections, frame)

        annotated = frame.copy()
        for t in tracked:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            color = color_for(t.track_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = tags.get(str(t.track_id)) or f"#{t.track_id}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            track_frame_counts[t.track_id] = track_frame_counts.get(t.track_id, 0) + 1
            # Save a gallery crop the first time we see a track, and again
            # once more mid-track (frame 15) to give a slightly better sample
            # to tag from than a possibly-partial first appearance.
            if t.track_id not in seen_track_ids or track_frame_counts[t.track_id] == 15:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(args.gallery_dir, f"track_{t.track_id:03d}.jpg"), crop)
                seen_track_ids.add(t.track_id)

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx} frames processed, {len(seen_track_ids)} tracks so far")

    cap.release()
    writer.release()
    elapsed = time.time() - t_start
    print(f"\ndone: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps) -> {args.out}")
    print(f"total distinct tracks: {len(seen_track_ids)}")
    print(f"gallery crops saved to: {args.gallery_dir}")

    template_path = os.path.join(args.gallery_dir, "tags_template.json")
    template = {str(tid): tags.get(str(tid), "") for tid in sorted(seen_track_ids)}
    with open(template_path, "w") as f:
        json.dump(template, f, indent=2)
    print(f"tag template written to: {template_path} -- fill in names, pass back via --tags")


if __name__ == "__main__":
    main()

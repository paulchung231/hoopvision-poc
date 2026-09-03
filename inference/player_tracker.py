#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Multi-player tracking with identity survival through brief occlusion.
Tracking-by-detection over BallHoopDetector's per-frame Player boxes -- no
extra model, no training. Two-stage matching per frame, mirroring how
ByteTrack/DeepSORT split the problem:

  Stage 1 (cheap, handles the common case): match currently-active tracks to
  this frame's detections by IoU against a simple constant-velocity position
  prediction. Players don't teleport frame-to-frame, so this alone resolves
  almost every match.

  Stage 2 (handles real occlusion): whatever's left over -- tracks that have
  gone missing for a few frames, and detections that didn't match anything
  in stage 1 -- gets matched by appearance similarity (an HSV color
  histogram of the box crop, roughly "what does this person's clothing look
  like") instead of position, since a track that's been occluded for a while
  has an unreliable position prediction but usually still looks the same.

Assumes a static/fixed camera (mounted, not handheld) -- the IoU position
prediction has no way to distinguish a player's own motion from camera pan.
Tested against a moving/panning clip and found this assumption genuinely
matters: the hoop's on-screen position swung hundreds of pixels within
seconds, and every player's predicted position was wrong just as often,
causing heavy track-ID churn. Rather than add camera-motion compensation
(optical flow to estimate and subtract out camera movement -- doable, but
real added complexity, and this system's own hoop-tracking in
shot_detector.py would need the same treatment), the simpler fix is a
capture requirement: mount the camera. Validate against
single_player_side_view.mp4 (confirmed static) rather than
pickup_moving_side_view.mp4 (confirmed panning -- see PROGRESS.md).

A track surviving the whole grace period unmatched is dropped. No global
identity (name) is assigned here -- that's a separate, manual step (see
`assign_player_tags` below) precisely because it's a project decision
(pickup games here have no jersey numbers to read automatically).
"""

from dataclasses import dataclass

import cv2
import numpy as np


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _appearance_hist(frame, bbox):
    """HSV hue/saturation histogram of a box crop -- ignores value (brightness)
    so it's reasonably lighting-invariant. A cheap stand-in for a learned
    re-ID embedding, good enough to distinguish players by clothing color."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _appearance_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return 0.0
    return max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _greedy_match(cost_matrix, threshold):
    """Repeatedly take the lowest-cost pair under threshold. Not optimal like
    the Hungarian algorithm, but simple, dependency-free, and fine for the
    handful of players typically in frame."""
    matches = []
    if cost_matrix.size == 0:
        return matches, list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    remaining = cost_matrix.copy()
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    while True:
        i, j = np.unravel_index(np.argmin(remaining), remaining.shape)
        cost = remaining[i, j]
        if cost > threshold or not np.isfinite(cost):
            break
        matches.append((i, j))
        unmatched_rows.discard(i)
        unmatched_cols.discard(j)
        remaining[i, :] = np.inf
        remaining[:, j] = np.inf

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


@dataclass
class TrackedPlayer:
    track_id: int
    bbox: tuple
    confidence: float


class _Track:
    def __init__(self, track_id, bbox, confidence, hist, frame_idx):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.velocity = (0.0, 0.0)
        self.hist = hist
        self.last_seen_frame = frame_idx
        self.hits = 1

    def predicted_bbox(self, frame_idx):
        dt = frame_idx - self.last_seen_frame
        x1, y1, x2, y2 = self.bbox
        vx, vy = self.velocity
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)

    def update(self, bbox, confidence, hist, frame_idx):
        dt = max(1, frame_idx - self.last_seen_frame)
        old_cx = (self.bbox[0] + self.bbox[2]) / 2
        old_cy = (self.bbox[1] + self.bbox[3]) / 2
        new_cx = (bbox[0] + bbox[2]) / 2
        new_cy = (bbox[1] + bbox[3]) / 2
        self.velocity = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)
        self.bbox = bbox
        self.confidence = confidence
        if hist is not None:
            # EMA so appearance adapts slowly (angle/lighting changes) without
            # forgetting the player entirely from one odd frame.
            self.hist = hist if self.hist is None else 0.7 * self.hist + 0.3 * hist
        self.last_seen_frame = frame_idx
        self.hits += 1


class PlayerTracker:
    IOU_MATCH_THRESHOLD = 0.3       # stage 1: min IoU to count as the same track
    IOU_STAGE_MAX_GAP = 10          # stage 1 eligibility: position prediction stays trustworthy this many frames
    APPEARANCE_MATCH_THRESHOLD = 0.5  # stage 2: min histogram correlation to re-acquire
    MAX_LOST_FRAMES = 45            # ~1.5s @ 30fps grace period before a track is dropped
    DUPLICATE_IOU_THRESHOLD = 0.6   # tracks/detections overlapping this much are the same physical person

    def __init__(self):
        self._tracks: list = []
        self._next_id = 1

    def update(self, frame_idx, detections, frame_image) -> list:
        """detections: Detection list from BallHoopDetector (any class named
        'player'/'Player' is treated as a player). frame_image: the BGR frame,
        needed to compute appearance histograms."""
        player_dets = [d for d in detections if d.class_name in ("player", "Player")]
        boxes = [d.bbox for d in player_dets]
        hists = [_appearance_hist(frame_image, b) for b in boxes]

        unmatched_det_idx = list(range(len(player_dets)))
        matched_track_ids = set()

        # Stage 1: IoU against predicted position, active (recently-seen) tracks only.
        active = [t for t in self._tracks if frame_idx - t.last_seen_frame <= self.IOU_STAGE_MAX_GAP]
        if active and unmatched_det_idx:
            cost = np.ones((len(active), len(unmatched_det_idx)))
            for i, t in enumerate(active):
                pred = t.predicted_bbox(frame_idx)
                for jj, j in enumerate(unmatched_det_idx):
                    cost[i, jj] = 1 - _iou(pred, boxes[j])
            matches, _, unmatched_cols = _greedy_match(cost, 1 - self.IOU_MATCH_THRESHOLD)
            still_unmatched = []
            consumed = set()
            for i, jj in matches:
                j = unmatched_det_idx[jj]
                active[i].update(boxes[j], player_dets[j].confidence, hists[j], frame_idx)
                matched_track_ids.add(active[i].track_id)
                consumed.add(j)
            unmatched_det_idx = [j for j in unmatched_det_idx if j not in consumed]

        # Stage 2: appearance match, remaining detections against lost/stale tracks.
        lost = [
            t for t in self._tracks
            if t.track_id not in matched_track_ids and frame_idx - t.last_seen_frame <= self.MAX_LOST_FRAMES
        ]
        if lost and unmatched_det_idx:
            cost = np.ones((len(lost), len(unmatched_det_idx)))
            for i, t in enumerate(lost):
                for jj, j in enumerate(unmatched_det_idx):
                    cost[i, jj] = 1 - _appearance_similarity(t.hist, hists[j])
            matches, _, _ = _greedy_match(cost, 1 - self.APPEARANCE_MATCH_THRESHOLD)
            consumed = set()
            for i, jj in matches:
                j = unmatched_det_idx[jj]
                lost[i].update(boxes[j], player_dets[j].confidence, hists[j], frame_idx)
                matched_track_ids.add(lost[i].track_id)
                consumed.add(j)
            unmatched_det_idx = [j for j in unmatched_det_idx if j not in consumed]

        # Remaining detections: new tracks -- unless a detection heavily
        # overlaps a track already matched *this same frame*, in which case
        # it's almost certainly a duplicate detection of the same physical
        # person slipping past the model's NMS (seen in practice: two
        # near-identical boxes for one person in one frame), not a second
        # person. Spawning a track for it creates a "zombie" that then
        # flickers for control with the real track on later frames. Drop it
        # instead of tracking it.
        matched_this_frame = [t for t in self._tracks if t.track_id in matched_track_ids]
        for j in unmatched_det_idx:
            if any(_iou(boxes[j], t.bbox) > self.DUPLICATE_IOU_THRESHOLD for t in matched_this_frame):
                continue
            track = _Track(self._next_id, boxes[j], player_dets[j].confidence, hists[j], frame_idx)
            self._tracks.append(track)
            matched_track_ids.add(track.track_id)
            matched_this_frame.append(track)
            self._next_id += 1

        # Safety net: merge any tracks that end up highly overlapping anyway
        # (e.g. two tracks that both survived separate occlusions and drifted
        # back together). Keep whichever has more hits (more established).
        self._tracks.sort(key=lambda t: -t.hits)
        deduped = []
        for t in self._tracks:
            if any(_iou(t.bbox, kept.bbox) > self.DUPLICATE_IOU_THRESHOLD for kept in deduped):
                continue
            deduped.append(t)
        self._tracks = deduped

        # Drop tracks that have exceeded the grace period.
        self._tracks = [t for t in self._tracks if frame_idx - t.last_seen_frame <= self.MAX_LOST_FRAMES]

        return [
            TrackedPlayer(track_id=t.track_id, bbox=t.bbox, confidence=t.confidence)
            for t in self._tracks
            if t.last_seen_frame == frame_idx
        ]

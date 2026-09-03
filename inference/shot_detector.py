#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Geometric shot-made detection: tracks the Ball detection's position over time
and checks whether it passes down through a Hoop detection's scoring zone.
Pure post-processing over BallHoopDetector output -- no extra model, no
training. Deliberately doesn't use the FG Attempt/FG Made classes (see
PROGRESS.md for why those are the weakest, sparsest labels in the dataset).

Handles multiple hoops in frame (e.g. a near hoop and a far hoop both
visible) by tracking one independent state machine per hoop -- a shot only
counts against whichever hoop's zone the ball actually falls through.
"""

from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class ShotEvent:
    frame_idx: int
    hoop_center: tuple


class _State(Enum):
    IDLE = auto()
    ENTERING = auto()


class _HoopTracker:
    """One state machine per physical hoop, tracking ball descent through its zone."""

    # Tunable geometry, expressed relative to the hoop bbox's own height/width
    # so it scales with distance-from-camera automatically.
    EXIT_MARGIN_RATIO = 0.8    # ball must fall this far below the rim (net depth) to count as made
    X_MARGIN_RATIO = 0.2       # horizontal tolerance around the rim's width
    JITTER_TOL_PX = 5          # allow small upward bbox-center noise without resetting
    TIMEOUT_FRAMES = 20        # abandon an in-progress shot if it stalls this long
    COOLDOWN_FRAMES = 30       # ignore re-arming for a bit after a make (net sway can look like a second descent)

    def __init__(self, bbox, smoothing=0.3):
        self.bbox = bbox
        self.smoothing = smoothing
        self.state = _State.IDLE
        self.state_frame = 0
        self.last_y = None
        self.last_seen_frame = 0
        self.cooldown_until = -1

    def update_zone(self, bbox):
        a = self.smoothing
        self.bbox = tuple(a * n + (1 - a) * o for n, o in zip(bbox, self.bbox))

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _zone(self):
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        exit_y = y2 + self.EXIT_MARGIN_RATIO * h
        margin = self.X_MARGIN_RATIO * w
        return exit_y, x1 - margin, x2 + margin

    def observe_ball(self, frame_idx, point):
        """point: (cx, cy) or None if the ball wasn't detected this frame."""
        exit_y, x_lo, x_hi = self._zone()

        if point is None:
            if self.state != _State.IDLE and frame_idx - self.last_seen_frame > self.TIMEOUT_FRAMES:
                self.state = _State.IDLE
            return None

        cx, cy = point
        in_x = x_lo <= cx <= x_hi
        self.last_seen_frame = frame_idx

        if self.state == _State.IDLE:
            # Arm as soon as the ball is seen in-x above the exit line -- not
            # gated on having witnessed it above entry_y first, since real
            # footage often occludes the ball (net, rim, motion blur) right
            # through that part of the descent. entry_y isn't load-bearing
            # here; the actual make/no-make discrimination happens below via
            # monotonic-descent + the exit_y crossing.
            if in_x and cy < exit_y and frame_idx >= self.cooldown_until:
                self.state = _State.ENTERING
                self.state_frame = frame_idx
                self.last_y = cy
            return None

        # ENTERING: ball must keep moving downward, staying inside the hoop's x-range
        if not in_x or cy < self.last_y - self.JITTER_TOL_PX:
            self.state = _State.IDLE
            return None

        self.last_y = cy
        if frame_idx - self.state_frame > self.TIMEOUT_FRAMES:
            self.state = _State.IDLE
            return None

        if cy > exit_y:
            self.state = _State.IDLE
            self.cooldown_until = frame_idx + self.COOLDOWN_FRAMES
            return ShotEvent(frame_idx=frame_idx, hoop_center=self.center)

        return None


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


class ShotDetector:
    """Feed it per-frame Detection lists (from BallHoopDetector); get ShotEvents back."""

    # IoU rather than center-distance: robust to a jittery hoop bbox (which
    # was fragmenting one physical hoop into multiple duplicate trackers,
    # each independently firing a MADE event for the same real shot).
    MATCH_IOU_THRESHOLD = 0.2

    # Ball association tuning: a video can have multiple "Ball" detections per
    # frame (e.g. a stationary false positive -- a floor marking, a shoe --
    # that sometimes outscores the real, motion-blurred ball in flight).
    # Picking highest confidence each frame is wrong; instead track the ball
    # by proximity to its last known position, like a minimal single-object
    # tracker. MAX_JUMP_PX_PER_FRAME caps how far the ball may plausibly move
    # per frame gap, so a stale/false detection can't get "adopted" just for
    # being the only candidate while the real ball is briefly occluded.
    MAX_GAP_FRAMES = 20
    MAX_JUMP_PX_PER_FRAME = 120

    # A real ball in play moves; a static false positive (a pictured ball in
    # wall art/signage, a floor marking) doesn't. Nearest-to-last-known-point
    # tracking has a failure mode where it locks onto exactly that kind of
    # static object and never lets go, because the object's own constant
    # presence keeps "refreshing" MAX_GAP_FRAMES every frame even though
    # it's never actually the ball in play (confirmed on real footage: a
    # basketball pictured in a wall poster). If the tracked point hasn't
    # moved in STATIC_ANCHOR_FRAMES, treat it as untrustworthy and let the
    # next pick re-bootstrap by confidence across all candidates instead of
    # staying anchored to it.
    STATIC_ANCHOR_FRAMES = 15
    STATIC_MOVE_TOL_PX = 4

    def __init__(self):
        self._hoops: list = []
        self._last_ball_point = None
        self._last_ball_frame = None
        self._still_frames = 0

    def _note_pick(self, point):
        if self._last_ball_point is not None:
            moved = ((point[0] - self._last_ball_point[0]) ** 2 + (point[1] - self._last_ball_point[1]) ** 2) ** 0.5
            self._still_frames = 0 if moved > self.STATIC_MOVE_TOL_PX else self._still_frames + 1

    def _match_hoop(self, bbox):
        best, best_iou = None, 0.0
        for tracker in self._hoops:
            iou = _iou(bbox, tracker.bbox)
            if iou > self.MATCH_IOU_THRESHOLD and iou > best_iou:
                best, best_iou = tracker, iou
        return best

    def _pick_ball_point(self, frame_idx, ball_dets):
        if not ball_dets:
            return None
        candidates = [((d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2, d.confidence) for d in ball_dets]

        have_recent_prior = (
            self._last_ball_point is not None
            and frame_idx - self._last_ball_frame <= self.MAX_GAP_FRAMES
            and self._still_frames < self.STATIC_ANCHOR_FRAMES
        )
        if not have_recent_prior:
            cx, cy, _ = max(candidates, key=lambda c: c[2])
            self._note_pick((cx, cy))
            self._last_ball_point = (cx, cy)
            self._last_ball_frame = frame_idx
            return (cx, cy)

        lx, ly = self._last_ball_point
        gap = frame_idx - self._last_ball_frame
        cx, cy, _ = min(candidates, key=lambda c: (c[0] - lx) ** 2 + (c[1] - ly) ** 2)
        if ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5 > self.MAX_JUMP_PX_PER_FRAME * gap:
            return None  # nothing plausible this frame; treat as occluded, don't move last-known point

        self._note_pick((cx, cy))
        self._last_ball_point = (cx, cy)
        self._last_ball_frame = frame_idx
        return (cx, cy)

    def update(self, frame_idx, detections) -> list:
        hoop_dets = [d for d in detections if d.class_name == "Hoop"]
        ball_dets = [d for d in detections if d.class_name == "Ball"]

        for d in hoop_dets:
            tracker = self._match_hoop(d.bbox)
            if tracker is None:
                self._hoops.append(_HoopTracker(d.bbox))
            else:
                tracker.update_zone(d.bbox)

        ball_point = self._pick_ball_point(frame_idx, ball_dets)

        events = []
        for tracker in self._hoops:
            event = tracker.observe_ball(frame_idx, ball_point)
            if event is not None:
                events.append(event)
        return events

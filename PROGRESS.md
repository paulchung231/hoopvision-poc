# HoopVision POC — Setup Progress

Goal: local fine-tuning benchmark run (YOLOX-s, 1 epoch) on ball/hoop dataset,
RTX 3080 (10GB VRAM), WSL2, to validate the training pipeline before a full run.

## Environment
- GPU: RTX 3080, driver 591.86, CUDA 13.1 (host) — confirmed via `nvidia-smi`.
- Python: system default is 3.14 (too new for PyTorch/YOLOX chain) → using
  pyenv 2.8.4 with Python 3.12.14, pinned via `.python-version` in this repo.
- venv: `~/hoopvision-poc/venv` (rebuilt once — first attempt was accidentally
  built against system Python 3.14, deleted and recreated with 3.12.14).
- Git: repo initialized at `~/hoopvision-poc`, `.gitignore` excludes
  `venv/`, `data/`, `checkpoints/`, `*.zip`, `YOLOX/`, outputs, caches.

## Done
- [x] Build deps for pyenv's Python compile installed via apt (had to split into
  two rounds — `libncursesw5-dev` doesn't exist on Ubuntu 26.04, used
  `libncurses-dev` instead).
- [x] pyenv install 3.12.14, set as local version for this repo.
- [x] venv created with 3.12.14, pip upgraded.
- [x] git repo initialized, first commit made (training config + gitignore).
- [x] `ball_hoop_training.zip` copied from Windows Downloads
  (`/mnt/c/Users/Paul Chung/Downloads/`) and extracted. Layout verified matches
  what YOLOX's COCODataset loader expects:
  - `training/yolox_s_ball_hoop.py`
  - `data/ball_hoop_dataset/subset/annotations/instances_{train,val}.json`
  - `data/ball_hoop_dataset/subset/train2017/` (200 images)
  - `data/ball_hoop_dataset/subset/val2017/` (50 images)
  - `checkpoints/yolox_s.pth` (72MB)
- [x] CUDA-enabled PyTorch installed: 2.13.0+cu126. `torch.cuda.is_available()`
  → True, `torch.cuda.get_device_name(0)` → "NVIDIA GeForce RTX 3080".
- [x] YOLOX cloned and installed editable (`pip install -v -e . --no-build-isolation`),
  ONNX export deps stripped from requirements.txt first — installed cleanly,
  no setuptools issues. pycocotools/loguru/thop/tabulate all pulled in
  automatically, no manual installs needed.
- [x] **1-epoch benchmark run completed successfully**, no errors. See below.

## Benchmark run results (2026-08-19 02:06)
- Command: `python YOLOX/tools/train.py -f training/yolox_s_ball_hoop.py -d 1 -b 16 -c checkpoints/yolox_s.pth`
- Full run (checkpoint load → train → eval → save): ~18s wall clock.
- 13 iterations/epoch (200 train images / batch 16). Logged iter_time (avg to
  iter 10): **0.857s/iter**, data_time 0.008s/iter (data loading is not the
  bottleneck).
- GPU memory: **8171MB / 10240MB (~80%)** at batch 16, 640x640 — this is close
  to the ceiling for the 3080's 10GB. Batch size is unlikely to go much higher
  without OOM; if a bigger effective batch is wanted later, use gradient
  accumulation rather than raising `-b` directly.
- Eval on val (50 images): AP@0.5:0.95 = 0.081, AP@0.5 = 0.193 (expected to be
  low — only 1 epoch on a 200-image subset, this run is a timing benchmark,
  not a real training result).
- Only warnings: `torch.cuda.amp.*` deprecation warnings (torch 2.13 wants
  `torch.amp.*` instead) — cosmetic, YOLOX's own code, not fixed since it
  doesn't affect correctness or speed.
- **Caveat for full-run timing estimates**: YOLOX disables mosaic/mixup
  augmentation for the last `no_aug_epochs` epochs (default 15) before
  `max_epoch`. Since this benchmark used `max_epoch=1`, it fell inside that
  window and mosaic was OFF the whole run ("--->No mosaic aug now!" in the
  log). Real training epochs early in a full run *will* have mosaic/mixup on,
  which does more per-image CPU-side augmentation and will likely push
  iter_time higher than 0.857s. Worth a second benchmark with mosaic forced
  on (or `max_epoch` > `no_aug_epochs`) before finalizing full-run time
  estimates.

## Full dataset pull & full run launch (2026-08-19)
- Full dataset (8,138 train / 203 valid images, all 7 classes, same category
  IDs as the subset — verified by diffing `categories` in both annotation
  JSONs) pulled directly from Roboflow into
  `data/ball_hoop_dataset_full/{train2017,val2017,annotations}`, same layout
  convention as the subset. `unzip` still isn't installed on this box —
  extracted via Python `zipfile` again.
- New exp config: `training/yolox_s_ball_hoop_full.py` — same
  depth/width/num_classes as the subset config, `data_dir` pointed at the
  full dataset, `max_epoch = 100`, `eval_interval = 10` (was `1`, too
  expensive to eval every epoch over 100).
- Launched:
  `venv/bin/python YOLOX/tools/train.py -f training/yolox_s_ball_hoop_full.py -d 1 -b 16 -c checkpoints/yolox_s.pth`
  as a background process (task id `bn289i43n`), output logged to
  `YOLOX_outputs/yolox_s_ball_hoop_full/train_log.txt` (and mirrored to the
  background-task output file). Confirmed at launch: `mosaic_prob 1.0`,
  `mixup_prob 1.0`, `no_aug_epochs 15` — so this run resolves the mosaic-off
  caveat from the benchmark by construction (mosaic/mixup will be on for
  epochs 1–85, off for the last 15).
- Epoch count (100) and "go straight to full dataset, skip a separate
  mosaic-only re-benchmark" were explicit user decisions — full run's own
  early iterations serve as the real mosaic-on timing measurement instead of
  a throwaway benchmark run.
- A persistent monitor (task `bcbfs03t7`) tails the run's output for epoch
  boundaries, per-epoch AP (every 10th epoch, per `eval_interval`), and error
  signatures (traceback/OOM/killed) — deliberately not tailing every
  `iter_time` line, that fires ~50x/epoch and would flood notifications over
  a ~20h run.
- First real mosaic-on iter_time readings (epoch 1, batch 16): 0.412–0.808s,
  noticeably noisier than the subset benchmark's steady 0.857s — YOLOX's
  multiscale training is resizing the input each iter (480 up to 704+ seen
  already, base 640 ± multiscale_range·32) on top of mosaic/mixup, both add
  variance. GPU mem climbed 8171→8712MB across the first 40 iters as scale
  increased. Early ETA estimates from the trainer itself: ~9-11h for 100
  epochs.

## Run stalled at iter 60/509 — killed and fixed (2026-08-19 02:25)
- Run hit iter 60, GPU mem climbed to 10015/10240MB (98%), then stopped
  logging entirely for 6+ minutes while GPU util stayed at 100% and the
  process stayed alive at ~99.7% CPU — not a crash, but effectively stalled
  (prior iters logged every 5-10s). Root cause: multiscale training
  (`multiscale_range=5` default) resizes input up to 800x800 (640±5*32)
  every 10 iters; at batch 16 that's right at the 3080's 10GB ceiling, and
  the CUDA allocator thrashes/retries near the limit instead of cleanly
  OOM-ing.
- Fix: killed the run (background task `bn289i43n`, plus its monitor/watcher
  tasks), added `self.multiscale_range = 0` to
  `training/yolox_s_ball_hoop_full.py` to pin input size at the fixed
  640x640 already proven safe in the subset benchmark (8171MB, steady
  0.857s/iter). Confirmed GPU memory fully freed (505MB baseline) before
  relaunching.
- Relaunched as background task `bhe0lb14i`. Trade-off accepted: fixed-scale
  training loses some of multiscale's robustness-to-input-size benefit, but
  given this is a single unattended ~8-20h run on a VRAM-constrained card,
  reliability wins. Revisit multiscale later (smaller range, e.g. ±2, or
  batch 12) if wanted for a future run.
- **Confirmed fixed**: GPU mem now pinned steady at 8171MB (80%, matches the
  subset benchmark exactly), no more climbing. iter_time settled to ~0.26s
  after warmup — faster than the multiscale run since there's no more
  large-scale iterations. New trainer ETA: **~5.4h** for 100 epochs, down
  from the ~9-11h estimate under multiscale (that estimate was inflated by
  the stall risk, not a fair comparison, but the fixed-scale run is also
  legitimately faster per-iter on average).
- New persistent monitor for this run: task `b9x8eskix` (same filter as
  before: epoch boundaries, eval AP every 10 epochs, error/OOM signatures).
  Old monitor/watcher tasks for the killed run (`bcbfs03t7`, `bj2o90hdb`)
  were stopped.

## Inference wrapper + video smoke test (2026-08-19 02:40-02:47)
- `inference/ball_hoop_detector.py` — standalone `BallHoopDetector` class,
  device-generic (CUDA > MPS > CPU auto-select), loads a checkpoint + exp
  config, `detect(image)` returns `Detection(class_name, confidence, bbox)`
  list. Deliberately self-contained (only YOLOX + torch + cv2 + numpy) so it
  can be copied into the Mac-side app pipeline later — this repo only
  trains, doesn't run the app.
- `inference/video_smoke_test.py` — samples frames from a video, runs the
  detector, draws boxes, dumps annotated jpgs + timing. Tested against
  `single_player_side_view.mp4` (from Windows Downloads) using the
  in-progress full-run checkpoint (~epoch 9): all 8 sampled frames got
  detections, ~9.5 fps on CPU (forced CPU deliberately to not compete with
  the training job's GPU memory).
- Visual check on a sample frame: correctly boxed both hoops (0.84, 0.75
  conf), the ball in-hand (0.78), and the player (0.87) — tight, accurate
  boxes, genuinely good for only ~9 epochs into a 100-epoch run.
- Design decision (not yet acted on): the model's own `player`/`Player`
  classes could eventually replace the existing separate player detector
  (single forward pass instead of two), but defer that until this model's
  Player-class AP is validated — for now, plan is to run this model
  ball/hoop-only alongside the existing player detector.

## Repo cleanup (2026-08-19 02:46)
- Deleted `ball_hoop_training.zip` (83MB, already fully extracted) and the
  4 superseded 1-epoch subset-benchmark checkpoints in
  `YOLOX_outputs/yolox_s_ball_hoop/*.pth` (275MB, results already recorded
  above) and `training/__pycache__/`. ~358MB freed. Nothing in the active
  full run touched. Repo's git side was already clean -- everything bulky
  is gitignored.

## Epoch 10 eval results (2026-08-19 02:49)
Overall AP@0.5:0.95 = **0.335**, AP@0.5 = **0.566** (vs. 0.081/0.193 for the
old 1-epoch subset benchmark) — 10% through the run.

| Class | AP | AR |
|---|---|---|
| Ball | 51.9 | 58.9 |
| Hoop | 56.8 | 67.2 |
| FG Attempt | 47.8 | 58.9 |
| FG Made | 18.4 | 66.7 |
| Player | 26.1 | 47.1 |
| Ref | 0.0 | 0.0 |

Ball/Hoop (what we actually need) are already the strongest classes. `Ref`
at 0 and lowercase `player` at `nan` need watching at the next eval (epoch
20) — may just be low/zero instance counts in val, not necessarily a
problem.

## Shot-made detection prototype (2026-08-19 02:56-03:11)
Built the actual downstream logic discussed above: geometric ball-through-hoop
detection over BallHoopDetector's per-frame output, not reliant on the
FG Made class (see rationale above -- sparsest, fuzziest label in the
dataset).

- `inference/shot_detector.py` -- `ShotDetector` class. One `_HoopTracker`
  state machine per physical hoop (handles multiple hoops in frame). Tracks
  the ball's bbox-center trajectory; arms when the ball is seen in the
  hoop's x-range above the exit line, confirms MADE when it crosses below
  the exit line (rim y2 + 0.8x hoop height, approximating net depth) without
  a big upward reversal in between (which would mean a miss/bounce-out, not
  a make).
- `inference/detect_shots_video.py` -- runs BallHoopDetector + ShotDetector
  over a full video, draws boxes + a "SHOT MADE" banner, prints a
  timestamped event log.
- Fixed along the way, all against real footage (2 clips from Windows
  Downloads: `single_player_side_view.mp4`, `pickup_moving_side_view.mp4`):
  1. **torch.load failure on post-eval checkpoints** -- once eval started
     saving a numpy AP scalar into the checkpoint dict (epoch 10+), PyTorch
     2.13's default `weights_only=True` rejected it. Fixed by passing
     `weights_only=False` in `ball_hoop_detector.py` (safe -- these are our
     own self-generated checkpoints, not third-party downloads).
  2. **Wrong ball picked when multiple "Ball" detections exist in a frame**
     -- naively picking highest-confidence-per-frame grabbed a stationary
     false positive (a floor marking, consistently ~0.3-0.5 conf) instead of
     the real, motion-blurred ball in flight, right at the critical moment
     it entered the hoop zone. Fixed with a minimal single-object tracker:
     pick the candidate nearest the last known ball position, and treat an
     implausibly large jump as "not detected this frame" (ball occluded)
     rather than adopting a wrong point.
  3. **Arming gated on seeing the ball above the "entry" line first --
     too fragile.** Real footage frequently occludes the ball (net, rim,
     motion blur) for a dozen-plus frames right through that part of the
     descent, so the ball would reappear already past the gate and never
     arm. Fixed by arming on any in-x sighting above the exit line instead
     -- the actual make/no-make discrimination is the monotonic-descent +
     exit-line-crossing check, not the entry gate.
  4. **Duplicate MADE events for one real shot** -- (a) no cooldown, so net
     sway after a make could look like a second descent and re-fire within
     ~0.4s; (b) hoop-tracker matching was center-distance-based and too
     strict, so bbox jitter fragmented one physical hoop into multiple
     tracker instances that all fired near-simultaneously. Fixed with a
     30-frame post-make cooldown per hoop, and switched hoop matching to
     IoU overlap (much more robust to jitter than center distance).
- **Validated against real video, visually confirmed frame-by-frame**: both
  test clips now report exactly 1 made shot each, and manually inspecting
  the flagged frame in each case shows a real ball-in-net moment (frame 103
  in `single_player_side_view.mp4`, frame 319 in `pickup_moving_side_view.mp4`
  -- pulled that raw frame and the ball is visibly right at the net). A
  pre-fix run had additionally flagged a phantom event at frame 113 of the
  pickup clip; checked that frame directly and there's no hoop anywhere near
  the claimed location (empty ceiling/lighting area) -- confirms the fixes
  removed a real false positive, not a real shot.
- Annotated output videos (`single_player_side_view_shots.mp4`, `pickup_moving_side_view_shots.mp4`,
  plus the earlier plain box-annotated `*_annotated.mp4`) copied to Windows
  Downloads for review.
- Not yet handled / known gaps for a future pass: no "miss" detection (only
  MADE events are emitted right now), single-ball-only tracking (would need
  extending for multi-ball scenes), and all of this is still running against
  a mid-training checkpoint (~epoch 19 as of this note) -- worth rerunning
  against `best_ckpt.pth` once the full 100-epoch run finishes.

## Epoch 20 eval results (2026-08-19 03:12)
Overall AP@0.5:0.95 = **0.418** (up from 0.335 at epoch 10), AP small =
0.278. Steady improvement, no red flags. Full per-class breakdown not
pulled this time (Ball/Hoop trend positive based on overall AP); pull it
again at epoch 30/40.

## Player tracking + manual tagging (2026-08-19 03:14-03:26)
Built the multi-player half of "tag players and track points per player" --
identity tracking with occlusion survival, plus a manual name-tagging
workflow (chosen over jersey-OCR/face-recognition since the test footage is
plain-clothes pickup ball with no jersey numbers -- OCR wouldn't work on it
anyway).

- `inference/player_tracker.py` -- `PlayerTracker`, tracking-by-detection
  over the Player boxes. Two-stage per-frame matching (mirrors
  ByteTrack/DeepSORT's structure): stage 1 matches active tracks to
  detections by IoU against a constant-velocity position prediction (cheap,
  handles the common case); stage 2 matches whatever's left -- tracks that
  have been missing for a few frames, i.e. real occlusion -- by appearance
  similarity (HSV color-histogram of the box crop) instead of position,
  since a long-lost track's position prediction is unreliable but it
  usually still looks the same. No learned re-ID model, no new dependency.
- `inference/track_players_video.py` -- runs detector + tracker over a
  video, draws stable color-coded boxes + track ID (or a name, if tags are
  supplied), and saves gallery crops per track for manual tagging.
- **Manual tagging loop, validated end-to-end**: first run (no `--tags`)
  writes `<gallery-dir>/tags_template.json` (`{"1": "", "2": "", ...}`) next
  to one representative crop per track (`track_NNN.jpg`); fill in names,
  pass the file back via `--tags`, and the output video labels players by
  name instead of track number. Confirmed working on
  `single_player_side_view.mp4` (tagged track 1 as "Alex", rendered
  correctly).
- **Found and fixed a real problem, not just a tuning issue**: first test
  was against `pickup_moving_side_view.mp4` and produced ~40 distinct
  tracks for what's visibly ~9 real people (confirmed by eye on a sample
  frame -- the *current-frame* boxes were accurate 1:1, so the churn was
  entirely about tracks losing and regaining identity over time, not bad
  detections). Root cause, confirmed by checking the Hoop detection's
  on-screen position over time: **that clip's camera pans hundreds of
  pixels within ~1 second** (confirmed: hoop center x went 1406→173 in 40
  frames). A per-player constant-velocity model can't tell "the player
  moved" from "the camera panned," so position predictions are wrong for
  every player at once whenever the camera pans, forcing constant fallback
  to the weaker appearance-matching stage.
  - Considered adding camera-motion compensation (optical flow to estimate
    and subtract camera movement before predicting player positions --
    standard technique, e.g. BoT-SORT's GMC module). Decided against it for
    now: real added complexity, and `shot_detector.py`'s hoop-tracking has
    the same unstated static-camera assumption and would need the same fix
    to fully benefit. User's call: simpler to make **static/mounted camera
    a capture requirement** for this system rather than engineer around
    handheld panning. Revisit camera-motion compensation only if a real
    use case requires handheld footage.
  - Re-validated against `single_player_side_view.mp4`, confirmed static
    (hoop position barely moves frame to frame): **1 stable track for all
    186 frames**, exactly matching the video's real content (one player).
    This is the clip to use for future player-tracking validation;
    `pickup_moving_side_view.mp4` remains useful for shot detection (where
    the hoop-vs-camera math tolerated it better) but not for tracking demos.
- Panning magnitude that actually breaks tracking, for future reference when
  deciding on mounting: roughly >20-40px/frame of *camera-induced* on-screen
  motion (given typical 60-150px player-box widths here and IoU match
  threshold 0.3) is enough to break stage-1 matching. Ordinary handheld
  unsteadiness should be fine; deliberate whip-pans to follow a fast break
  are not.
- Gotcha worth remembering: comparing two runs against `latest_ckpt.pth`
  while training is still active isn't apples-to-apples -- the file is
  being overwritten epoch to epoch, so the underlying model genuinely
  changes between calls (observed directly: 1 vs. 2 tracks on identical
  input, explained by this, not a tracker bug). Snapshot/copy a checkpoint
  for stable side-by-side comparisons in the future.
- Not yet done: attributing a made shot (from `shot_detector.py`) to a
  specific tracked/named player -- next logical step once both pieces are
  solid, via "which tracked player was nearest to / had the ball right
  before release."
- Demo videos copied to Windows Downloads: `pickup_moving_side_view_tracked.mp4` (numeric
  IDs, panning clip, shows the pre-fix churn -- kept as a reference example
  of the camera-pan problem), `single_player_side_view_named.mp4` (name-tagged, static clip,
  clean result).

## Duplicate-detection track bug (player tracker) + third real video (2026-08-19 03:28-03:33)
- New clip added: `clipped_pickup_back_view.mp4` (confirmed static camera --
  hoop center stays within a couple px of (945,417) for the whole clip).
- User caught a real bug: `single_player_side_view_named.mp4` showed 2 players when there's
  only 1. Traced it -- at frame 60, the detector produced two heavily
  overlapping "Player" boxes for the same physical person (an NMS near-miss,
  not a tracker logic error), which spawned a genuine duplicate track. Once
  both tracks existed, the matcher had no way to know they were the same
  person, so it flip-flopped between them frame to frame on tiny position
  noise.
- Fixed in `player_tracker.py`: (1) before spawning a new track from a
  leftover detection, check it against tracks already matched *this same
  frame* -- if it heavily overlaps one (IoU > 0.6), it's a duplicate
  detection of an already-tracked person, drop it, don't track it; (2) a
  general dedup safety-net pass each frame merges any tracks that end up
  highly overlapping anyway (keeps the more-established one, by hit count).
  Re-verified against `single_player_side_view.mp4`: exactly 1 track for
  the whole clip now.
- Snapshotted a checkpoint for stable comparisons going forward:
  `checkpoints/snapshots/epoch28_snapshot.pth` (copied from the live
  `latest_ckpt.pth`, which the training job keeps overwriting -- confirmed
  directly this matters: got different track counts on identical input
  across two calls purely because the live file had changed underneath us
  between them).

## Player tracker on the new static clip: real, distinct failure mode (2026-08-19 03:33-03:40)
- `clipped_pickup_back_view.mp4` (defensive slide/agility drill, players far
  from camera, boxes only 40-100px wide) still produced heavy track churn
  (37 tracks for ~6 real people) *despite* the camera being confirmed static
  -- ruling out the panning explanation this time.
- User pushed back on my first theory (fast/erratic motion) and pointed out
  a more likely one directly: players walking out of camera frame and back
  in. That's a distinct failure mode from occlusion -- a full re-acquisition
  after a real gap, which leans entirely on the weaker appearance-matching
  stage (color histogram), not the reliable IoU stage. User's call: don't
  engineer around this now, send current outputs for evaluation first.
- Sent `clipped_pickup_back_view_shots.mp4` and `clipped_pickup_back_view_tracked.mp4` to
  Downloads for review before deciding whether/how to fix.

## Shot detector missed a real shot -- found and fixed a distinct bug (2026-08-19 03:44-03:52)
- User reported the shot detector missed an actual made shot in
  `clipped_pickup_back_view.mp4`. Traced the real ball trajectory directly
  (frames 297-331: ball approaches the hoop, brief low-confidence gap, then
  a near-dead-center sighting at the rim (frame 325, 7px from hoop center),
  then a clean downward fall through frame 331 -- textbook make pattern)
  and compared against what `ShotDetector` actually tracked: **the ball
  association was frozen on `(1198, 414)` for the entire window and never
  moved.** Visually confirmed what's there: a promotional wall poster
  showing a person holding a basketball -- the *pictured* ball got detected
  as a real "Ball" and, being permanently present, kept re-satisfying the
  recency check every frame, so the tracker never went stale and never
  looked anywhere else. Different bug from the earlier "wrong confidence
  pick" fix -- this was a stale anchor that never expired because its own
  constant presence kept refreshing its own freshness.
- Separately verified the user's other hypothesis (camera-angle/domain gap)
  with real evidence rather than guessing: pulled 5 random training images
  (`dataset_sample_montage.jpg`, sent to Downloads) -- all close-up,
  side-on, eye-level broadcast/action shots, nothing resembling this clip's
  far-away, elevated, whole-court angle. Real domain gap, plausibly explains
  the generally lower confidence scores in this clip (0.3-0.7 vs 0.8+ in
  the closer clips) -- but confirmed separately that it did *not* cause
  this specific miss; the sticky-anchor bug did, independent of confidence
  level.
- Fixed in `shot_detector.py`: track how long the ball-association's picked
  point has gone without moving more than a few px
  (`STATIC_ANCHOR_FRAMES`/`STATIC_MOVE_TOL_PX`). A real ball in play moves;
  something that hasn't moved in 15 frames is probably scenery, not the
  ball -- once that threshold is hit, the next pick re-bootstraps by
  confidence across *all* candidates instead of staying anchored. Re-ran on
  `clipped_pickup_back_view.mp4`: now correctly detects the make at frame
  331 (t=11.03s), matching the manually-traced trajectory.
- Regression-tested against the other two clips. `single_player_side_view`
  still correct (1 shot). `pickup_moving_side_view` dropped from 1 to 0
  detected shots -- **investigated and confirmed this is not caused by the
  fix** (disabling the new logic entirely, same checkpoint, still finds 0 --
  isolates the cause to the checkpoint having changed, not the code).
  Root cause on this clip: another sticky-decoy-shaped object, but this
  time in a *panning* clip, where a world-static decoy still drifts
  on-screen right along with the pan -- which defeats the new
  near-zero-movement check, since it never looks "stuck." This is the
  panning problem showing up a third way (after hoop-tracking and
  player-tracking) -- not a new bug to chase, the same already-accepted
  limitation. `pickup_moving_side_view.mp4` isn't a reliable test case for
  shot detection either, for the same reason it wasn't for player tracking:
  use the two static clips for validation going forward.

## Video organization (2026-08-19 03:59)
- Moved all source clips and generated result videos out of Windows
  Downloads into the repo, so they're browsable/playable directly in
  VS Code: `videos/input/` (source clips: `single_player_side_view.mp4`,
  `pickup_moving_side_view.mp4`, `clipped_pickup_back_view.mp4`, and the
  raw uncut `pickup_back_view.mp4` it was clipped from) and `videos/output/`
  (everything `inference/*.py` has generated: `*_shots.mp4`, `*_tracked.mp4`,
  `*_named.mp4`, plus `dataset_sample_montage.jpg`).
- Added `videos/` to `.gitignore` (same treatment as `data/`,
  `checkpoints/`, `YOLOX_outputs/` — large binaries, not meant to be
  git-tracked).
- Going forward: write new inference outputs directly to `videos/output/`
  instead of the scratchpad + copy-to-Downloads pattern used earlier in
  this session.
- **Playback note**: VS Code's built-in video preview (Chromium-based, only
  decodes H.264/VP8/VP9/AV1) won't play these files. Confirmed why: this
  OpenCV build can't encode H.264 here (tested directly -- `avc1`/`h264`/
  `H264`/`X264` fourccs all fail, no hardware encoder in WSL, no libx264
  compiled in; only `mp4v`/MPEG-4 Part 2 actually works), and there's no
  system `ffmpeg` CLI to transcode afterward either. User's call: use VLC
  (or another real media player) to view these files rather than installing
  ffmpeg + switching the pipeline to H.264. No code changes made.

## Full training run complete (2026-08-19 06:24)
- 100/100 epochs finished cleanly, no crashes, no restarts needed after the
  iter-60 stall fix on epoch 1. Total wall clock roughly 4h (02:26 start to
  06:24 finish) -- well under the original ETA estimates, since the
  multiscale-disabled config turned out faster per-iter than expected on
  top of being more reliable.
- **Best AP@0.5:0.95 = 50.63%**, achieved at **epoch 96** --
  `YOLOX_outputs/yolox_s_ball_hoop_full/best_ckpt.pth`. This is the
  checkpoint to use going forward, not `latest_ckpt.pth`/`last_epoch_ckpt.pth`
  (epoch 100, marginally worse per the eval trend below).
- Final epoch (100) full eval: AP@0.5:0.95 0.496, AP@0.5 0.727, AP@0.75
  0.591, AP small/medium/large 0.304/0.403/0.570, AR (max100) 0.558.
- Final per-class AP: **Ball 61.1**, **Hoop 68.0** (the two classes that
  actually matter for shot detection -- both strong), FG Attempt 60.4,
  FG Made 69.7 (surprisingly high given this was flagged early on as the
  sparsest/fuzziest label -- worth another look, but not blocking since we
  built shot detection around geometry instead of this class anyway),
  Player 38.3 (present but well behind Ball/Hoop -- confirms the earlier
  decision to keep the existing separate player detector for now rather
  than merging), Ref 0.0 (stayed at zero the entire run -- likely too few
  training instances for that class to ever be learned, not a bug).
- Eval trend across the run for reference: 0.081 (1-epoch subset) -> 0.335
  (ep10) -> 0.418 (ep20) -> 0.447 (ep30) -> 0.464 (ep40) -> 0.473 (ep50) ->
  0.476 (ep60) -> 0.470 (ep70, dip) -> 0.475 (ep80) -> 0.477 (ep85, mosaic
  off) -> climbed through the no-aug tail to peak 0.506 (ep96) -> settled
  to 0.496 (ep100). The no-mosaic tail (last 15 epochs) clearly helped,
  consistent with the plateau observed at epoch 50-80 under mosaic.
- Every epoch checkpoint was kept (`epoch_N_ckpt.pth` for N=10,20,...,100,
  plus 85-99 individually since eval ran every epoch during the no-aug
  tail) -- ~2GB total in `YOLOX_outputs/yolox_s_ball_hoop_full/`, all
  gitignored.

## Next steps
1. Re-run the shot detector and player tracker against `best_ckpt.pth`
   (snapshot it to `checkpoints/snapshots/` first, matching the
   epoch28_snapshot.pth pattern already used) instead of the mid-training
   checkpoints used for prototyping so far.
2. Decide on the player-detection merge question (single-pass model vs. two
   separate detectors) -- now has a real answer: Player AP (38.3) trails
   Ball/Hoop (61-68) enough that keeping the existing separate player
   detector remains the right call, at least for now.
3. Build shot-to-player attribution (nearest tracked player to the ball at
   release, credited on a MADE event).
4. Decide capture requirements for real use (static/mounted camera, per the
   panning findings -- confirmed to affect hoop-tracking, player-tracking,
   *and* ball-tracking) and document them for whoever films footage.
   Validate against static clips only (`single_player_side_view.mp4`,
   `clipped_pickup_back_view.mp4`) going forward -- `pickup_moving_side_view.mp4`
   is not a reliable test case for any of the three systems.
5. Awaiting user evaluation of `clipped_pickup_back_view_tracked.mp4`
   (player-tracker churn on the drill clip) before deciding whether the
   leaves-frame-and-returns re-acquisition problem needs a fix.
6. Eventually port `inference/` into the Mac-side app pipeline.
7. Consider whether the training dataset's apparent lean toward close-up
   side-view/broadcast imagery (see `videos/output/dataset_sample_montage.jpg`)
   needs addressing if far-away/elevated camera angles are a real
   deployment scenario -- would mean sourcing/adding more images at that
   framing.
8. Decide whether 100 epochs was enough or another run is worth it: AP
   peaked at epoch 96 and had already settled/dipped slightly by epoch 100
   -- more epochs alone probably won't help much further without also
   addressing the mosaic-era plateau (epochs 50-80) or adding more/better
   data (e.g. the camera-angle diversity gap from point 7).

## Issues hit & fixes
- `sudo apt-get install ...` needs an interactive terminal — Bash tool can't
  supply a password prompt, so these were run manually by the user.
- `libncursesw5-dev` package doesn't exist on Ubuntu 26.04 → used
  `libncurses-dev` instead.
- `unzip` not installed on the system → used Python's `zipfile` module instead.
- First venv was built against system Python 3.14 by mistake (before pyenv
  local was set) → deleted and rebuilt against 3.12.14.
- Git commit failed with "Author identity unknown" → user set
  `git config user.name` / `user.email` locally (not global, per policy of not
  touching git config directly).

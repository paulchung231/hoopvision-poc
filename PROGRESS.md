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

## Next steps
1. Decide full training run parameters (epochs, whether to use the full
   8,138-image export vs. subset) now that the pipeline is verified working.
2. Optionally re-benchmark with mosaic/mixup enabled to get a realistic
   iter_time for the bulk of a full run (see caveat above).
3. Launch the full training run.

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

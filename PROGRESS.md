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
- [ ] **In progress**: installing CUDA-enabled PyTorch (2.13.0, cu126 build) into
  the venv. Chose cu126 index over cu129/cu130 (also available) — mature/
  well-tested, and RTX 3080 driver (CUDA 13.1) is backward-compatible with it.

## Next steps
1. Verify `torch.cuda.is_available()` is True and reports the 3080.
2. Clone YOLOX from source, strip onnx-simplifier/onnx>= lines from its
   requirements.txt (legacy setup.py breaks under modern setuptools; those
   deps are only needed for ONNX export), `pip install -v -e . --no-build-isolation`.
3. Run the 1-epoch benchmark:
   `python YOLOX/tools/train.py -f training/yolox_s_ball_hoop.py -d 1 -b 16 -c checkpoints/yolox_s.pth`
   Install missing deps as needed (pycocotools, loguru, thop, tabulate).
4. Report: did it complete, per-iteration/epoch timing, errors + fixes.

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

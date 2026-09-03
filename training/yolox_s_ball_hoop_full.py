#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Full fine-tune of YOLOX-s for ball/hoop detection on the complete 8,138-image
train / 203-image val export, starting from the COCO-pretrained checkpoint.
Run from the project root (paths below are relative to it) on a machine with
a real CUDA GPU -- YOLOX's stock trainer is CUDA-only.

    python YOLOX/tools/train.py -f training/yolox_s_ball_hoop_full.py -d 1 -b 16 \
        -c checkpoints/yolox_s.pth

This follows the 1-epoch subset benchmark (training/yolox_s_ball_hoop.py),
which validated the pipeline end-to-end (checkpoint loading, data layout,
num_classes, training loop, eval) at 0.857s/iter with mosaic/mixup disabled.
"""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "data/ball_hoop_dataset_full"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"

        # categories from the Roboflow export: player(0), Ball(1), FG Attempt(2),
        # FG Made(3), Hoop(4), Player(5), Ref(6) -- id range is 0-6, so 7 classes.
        self.num_classes = 7

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 10

        # Multiscale training (default multiscale_range=5) resizes the input
        # up to 800x800 (640 +/- 5*32) every 10 iters. At batch 16 that pushed
        # VRAM to ~98% (10015/10240MB) and the run stalled for minutes at a
        # time -- the CUDA allocator thrashing near the ceiling rather than a
        # clean OOM. Pin input size to the fixed 640x640 already proven safe
        # in the subset benchmark (8171MB, steady 0.857s/iter, no stalls).
        self.multiscale_range = 0

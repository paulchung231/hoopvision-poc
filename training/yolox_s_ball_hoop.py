#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fine-tune YOLOX-s for ball/hoop detection, starting from the COCO-pretrained
checkpoint. Run from the project root (paths below are relative to it) on a
machine with a real CUDA GPU -- YOLOX's stock trainer is CUDA-only.

    python YOLOX/tools/train.py -f training/yolox_s_ball_hoop.py -d 1 -b 16 \
        -c checkpoints/yolox_s.pth

Quick-benchmark run first: data_dir points at the 200/50-image subset
(data/ball_hoop_dataset/subset), not the full 8,138-image export, so we can
time it before committing to a full run.
"""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "data/ball_hoop_dataset/subset"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"

        # categories from the Roboflow export: player(0), Ball(1), FG Attempt(2),
        # FG Made(3), Hoop(4), Player(5), Ref(6) -- id range is 0-6, so 7 classes
        # even though we filtered images down to ones containing Ball/Hoop.
        self.num_classes = 7

        self.max_epoch = 1  # quick timing benchmark -- raise once we have a number
        self.data_num_workers = 4
        self.eval_interval = 1

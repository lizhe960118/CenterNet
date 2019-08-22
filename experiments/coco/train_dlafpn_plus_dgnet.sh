#!/bin/bash
python tools/train.py \
 configs/centernet_coco/dla_fpn_plus_dgnet.py \
 --work_dir /root/train/trainset/1/dla_fpn_plus_cache \
 --gpus 3

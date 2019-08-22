#!/bin/bash
python tools/train.py \
 configs/centernet_coco/dla_fpn_plus.py \
 --work_dir /data/lizhe/model/dla_fpn_plus_cache \
 --gpus 2
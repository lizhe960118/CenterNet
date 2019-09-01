#!/bin/bash
python tools/train.py \
 configs/centernet_coco/dla_concat_fpn.py \
 --work_dir /data/lizhe/model/dla_concat_fpn_cache \
 --gpus 2
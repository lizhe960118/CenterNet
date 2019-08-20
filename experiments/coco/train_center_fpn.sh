#!/bin/bash
python tools/train.py \
 configs/centernet_coco/centernet_fpn_coco.py \
 --work_dir /data/lizhe/model/centernet_fpn_cache \
 --gpus 8
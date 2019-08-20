#!/bin/bash
python tools/train.py \
 configs/centernet_coco/centernet_dla_fpn_coco.py \
 --work_dir /data/lizhe/model/centernet_dla_fpn_cache \
 --gpus 8
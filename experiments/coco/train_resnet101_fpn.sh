#!/bin/bash
python tools/train.py configs/centernet_coco/resnet101_fpn.py \
    --work_dir /data/lizhe/model/resnet101_fpn_cache/ \
    --gpus 2
 
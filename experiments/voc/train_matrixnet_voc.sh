#!/bin/bash
python tools/train.py \
    configs/centernet_voc/matrixnet.py \
    --work_dir /data/lizhe/matrixnet_cache --gpus 2
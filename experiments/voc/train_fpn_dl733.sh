#!/bin/bash
python tools/train_1.py \
    configs/centernet_fpn_voc_dl733.py \
    --work_dir /hdd/lizhe/centernet_fpn_cache --gpus 1
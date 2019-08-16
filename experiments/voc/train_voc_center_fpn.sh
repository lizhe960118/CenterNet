#!/bin/bash
python tools/train.py \
 configs/centernet_fpn_voc.py \
 --work_dir /data/lizhe/model/voc_centernet_fpn_cache \
 --gpus 8
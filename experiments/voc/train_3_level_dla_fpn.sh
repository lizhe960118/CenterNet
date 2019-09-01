#!/bin/bash
python tools/train.py  configs/centernet_voc/3_level_dla_fpn.py \
  --work_dir /data/lizhe/model/3_level_dla_cache_2 --gpus 1

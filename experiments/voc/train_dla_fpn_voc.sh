#!/bin/bash
python tools/train.py  configs/centernet_voc/dla_512.py \
  --work_dir /data/lizhe/model/dla_voc_cache_2 --gpus 1

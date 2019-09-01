#!/bin/bash
python tools/train.py \
 configs/centernet_coco/hgfpn.py \
 --work_dir /data/lizhe/model/hgfpn_cache --gpus 4

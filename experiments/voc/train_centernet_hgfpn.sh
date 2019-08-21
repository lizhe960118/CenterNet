#!/bin/bash
python tools/train.py \
 configs/centernet_voc/centernet_hgfpn_voc.py \
 --work_dir /data/lizhe/model/centernet_hgfpn_cache \
 --gpus 2
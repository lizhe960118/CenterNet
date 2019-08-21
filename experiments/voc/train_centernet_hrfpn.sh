#!/bin/bash
python tools/train.py \
 configs/centernet_voc/centernet_hrfpn_voc.py \
 --work_dir /data/lizhe/model/centernet_hrfpn_cache \
 --gpus 2

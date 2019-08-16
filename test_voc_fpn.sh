#!/bin/bash
python tools/test.py configs/centernet_fpn_voc.py \
     /data/lizhe/model/voc_centernet_fpn_cache/latest.pth \
    --out results.pkl \
    --eval bbox
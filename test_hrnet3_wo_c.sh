#!/bin/bash
python tools/test.py configs/voc_centernet_hrnet3.py \
    /data/lizhe/model/hr3_cache_1/latest.pth \
    --out results.pkl \
    --eval bbox
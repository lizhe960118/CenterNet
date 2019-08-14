#!/bin/bash
python tools/test.py configs/voc_centernet_hrnet3_32.py \
    /data/lizhe/model/hr32_cache/latest.pth \
    --out results.pkl \
    --eval bbox
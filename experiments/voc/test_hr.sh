#!/bin/bash
python tools/test.py configs/voc_centernet_hrnet3.py \
    /data/lizhe/model/hr_cache/latest.pth \
    --out results.pkl \
    --eval bbox

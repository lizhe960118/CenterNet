#!/bin/bash
python tools/test.py configs/voc_centernet_hourglass.py \
    /data/lizhe/model/hourglass_cache/latest.pth \
    --out results.pkl \
    --eval bbox

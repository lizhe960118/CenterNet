#!/bin/bash
python tools/test.py configs/voc_centernet_dla.py \
 /data/lizhe/model/dla_cache_2/latest.pth \
    --out results.pkl --eval bbox
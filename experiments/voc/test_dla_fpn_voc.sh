#!/bin/bash
python tools/test.py configs/centernet_voc/dla_512.py \
    /data/lizhe/model/dla_voc_cache/latest.pth \
    --out dla_voc_cache.pkl \
    --eval bbox
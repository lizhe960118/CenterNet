#!/bin/bash
python tools/test.py configs/centernet_voc/matrixnet.py \
    /data/lizhe/matrixnet_cache/latest.pth \
    --out matrixnets.pkl \
    --eval bbox
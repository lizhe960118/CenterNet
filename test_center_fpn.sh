#!/bin/bash
python tools/test.py configs/centernet_fpn_coco.py \
    /data/lizhe/model/centernet_fpn_cache/epoch_1.pth \
    --out results.pkl \
    --eval bbox
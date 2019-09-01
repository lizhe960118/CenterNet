#!/bin/bash
python tools/test.py configs/centernet_coco/dla_fpn_plus.py \
    /data/lizhe/model/dla_fpn_plus_cache/latest.pth \
    --out dla_plus_fpn_results.pkl \
    --eval bbox

#!/bin/bash
python tools/test.py configs/fcos_plus_coco.py \
    /data/lizhe/model/fcos_plus_2_cache/latest.pth \
    --out fcos_plus_2x_results.pkl \
    --eval bbox
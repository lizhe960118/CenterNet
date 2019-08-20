#!/bin/bash
python tools/test.py configs/fcos_plus_coco.py \
    /data/lizhe/model/fcos_plus_cache_2/latest.pth \
    --out fcos_plus_2_results.pkl \
    --eval bbox
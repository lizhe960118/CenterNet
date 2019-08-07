#!/bin/bash
python tools/visdrone_test.py configs/visdrone_centernet_hrnet.py \
    hr_cache/latest.pth \
    --out results.pkl \
    --iou_thr 0.5

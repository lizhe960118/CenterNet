#!/bin/bash
python tools/visdrone_test_1.py configs/visdrone_centernet_dla.py \
    dla_cache/epoch_90.pth \
    --out results.pkl \
    --iou_thr 0.5

#!/bin/bash
python tools/visdrone_test.py configs/visdrone_centernet_hrnet3.py \
    hr3_cache/epoch_70.pth \
    --out results.pkl \
    --iou_thr 0.5

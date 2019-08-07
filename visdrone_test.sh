#!/bin/bash
python tools/visdrone_test.py configs/visdrone_faster_rcnn_r50_fpn_1x.py \
    visdrone_cache/latest.pth \
    --out results.pkl \
    --iou_thr 0.5

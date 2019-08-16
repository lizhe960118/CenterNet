#!/bin/bash
python tools/test.py configs/test_faster_rcnn_r50_fpn_1x.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
    --out results.pkl --eval bbox 

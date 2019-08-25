#!/bin/bash
python tools/test.py configs/centernet_coco/resnet101.py \
    /home/lizhe/CenterNet/ctdet_coco_resdcn101.pth \
    --out resnet101_results.pkl \
    --eval bbox
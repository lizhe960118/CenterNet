#!/bin/bash
python tools/test.py configs/centernet_coco/resnet101_fpn.py \
    /data/lizhe/model/resnet101_fpn_cache/latest.pth \
    --out resnet101_fpn_results.pkl \
    --eval bbox
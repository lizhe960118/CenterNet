#!/bin/bash
python tools/test.py configs/centernet_coco/centernet_fpn_coco.py \
    /data/lizhe/model/centernet_fpn_2_cache/latest.pth \
    --out results.pkl \
    --eval bbox

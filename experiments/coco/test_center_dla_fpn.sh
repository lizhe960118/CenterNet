#!/bin/bash
python tools/test.py configs/centernet_coco/centernet_dla_fpn_coco.py \
    /data/lizhe/model/centernet_dla_fpn_cache/latest.pth \
    --out dla_fpn_results.pkl \
    --eval bbox
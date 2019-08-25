#!/bin/bash
python tools/test.py configs/centernet_coco/centernet_dla.py \
    /home/lizhe/CenterNet/ctdet_coco_dla_2x.pth \
    --out dla_results.pkl \
    --eval bbox
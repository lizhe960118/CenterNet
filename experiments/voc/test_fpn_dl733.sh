#!/bin/bash
python tools/test.py configs/centernet_fpn_voc_dl733.py \
    /hdd/lizhe/centernet_fpn_cache/latest.pth \
    --out voc_center_dl733.pkl \
    --eval bbox
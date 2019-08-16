#!/bin/bash
python tools/voc_test.py \
    configs/centernet_fpn_voc_dl733.py \
   /hdd/lizhe/centernet_fpn_cache/epoch_2.pth \
   --out center_result.pkl

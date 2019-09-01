#!/bin/bash
python tools/test.py \
   configs/centernet_voc/dla_weight_center_head.py \
   /data/lizhe/model/weight_dla_cache/latest.pth \
   --out weight_dla_result.pkl \
   --eval bbox

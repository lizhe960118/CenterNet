#!/bin/bash
python tools/test.py \
   configs/centernet_voc/3_level_dla_fpn.py \
   /data/lizhe/model/3_level_dla_cache_2/latest.pth \
   --out 3_level_dla_result.pkl \
   --eval bbox
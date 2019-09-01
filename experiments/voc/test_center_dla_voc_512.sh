#!/bin/bash
python tools/test.py \
   configs/centernet_voc/voc_centernet_dla.py \
   /home/lizhe/CenterNet/ctdet_pascal_dla_512.pth \
   --out dla_result.pkl \
   --eval bbox
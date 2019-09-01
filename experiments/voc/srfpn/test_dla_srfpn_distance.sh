#!/bin/bash
python tools/test_file_dir/test_1.py \
   configs/centernet_voc/dla_srfpn_distance.py \
   /data/lizhe/srfpn/dla_srfpn_distance_cache/latest.pth \
   --out dla_srfpn_distance_result.pkl \
   --eval bbox
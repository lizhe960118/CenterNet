#!/bin/bash
python tools/test_file_dir/test_1.py \
   configs/centernet_voc/dla_srfpn_area_64.py \
   /data/lizhe/srfpn/dla_srfpn_area_64_cache/latest.pth \
   --out dla_srfpn_area_64_result.pkl \
   --eval bbox
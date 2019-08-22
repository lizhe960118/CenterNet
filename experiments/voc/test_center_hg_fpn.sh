#!/bin/bash
python tools/test.py configs/centernet_voc/centernet_hgfpn_voc.py \
    /data/lizhe/model/centernet_hgfpn_cache/latest.pth \
    --out hg_fpn_results.pkl \
    --eval bbox
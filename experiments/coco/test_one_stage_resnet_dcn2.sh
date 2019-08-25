#!/bin/bash
python tools/test.py configs/centernet_coco/one_stage_resnet_dcn2.py \
    /data/lizhe/model/one_stage_resnetdcn2_cache/latest.pth \
    --out one_stage_resnet_dcn_results.pkl \
    --eval bbox
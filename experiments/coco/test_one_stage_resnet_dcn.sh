#!/bin/bash
python tools/test.py configs/centernet_coco/one_stage_resnet_dcn.py \
    /data/lizhe/model/one_stage_resnetdcn_cache/latest.pth \
    --out one_stage_resnet_dcn_results.pkl \
    --eval bbox

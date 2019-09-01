#!/bin/bash
python tools/test.py configs/centernet_coco/two_stage_resnet_dcn2.py \
    /data/lizhe/model/two_stage_resnetdcn2_cache/latest.pth \
    --out two_stage_resnet_results.pkl \
    --eval bbox
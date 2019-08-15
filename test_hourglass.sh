#!/bin/bash
python tools/visdrone_test_1.py configs/visdrone_centernet_hourglass.py \
    hourglass_cache/epoch_100.pth \
    --out results.pkl \
    --eval bbox

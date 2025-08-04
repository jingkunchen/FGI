#!/bin/bash

EXP_PATH="./"
ROOT="./data/NCI-ISBI-2013"

python run.py \
    -m gaze_sup \
    --data prostate \
    --model unet \
    -bs 8 \
    --exp_path $EXP_PATH \
    --root $ROOT \
    --spatial_size 224 \
    --in_channels 1 \
    --opt sgd \
    --lr 1e-2 \
    --lr_min 1e-4 \
    --lr_scheduler cos \
    --max_ite 15000 \
    --num_levels 2 \
    --cons_mode prop \
    --cons_weight 3 \
    --data_size_rate 1 \
    --device 2 \
    --seed 0

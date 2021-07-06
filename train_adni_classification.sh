#!/bin/sh

DATA_DIR=""  # <- Your Data Directory

python train.py \
    --train_data "${DATA_DIR}/0-train.h5" \
    --val_data "${DATA_DIR}/0-valid.h5" \
    --test_data "${DATA_DIR}/0-test.h5" \
    --discriminator_net "daft" \
    --learning_rate "0.013" \
    --decay_rate "0.0" \
    --film_location "3" \
    --bottleneck_factor "7.0" \
    --n_basefilters "4" \
    --num_classes "3" \
    --dataset "longitudinal" \
    --normalize_image "minmax" \
    --normalize_tabular \
    --task "clf" \
    --batchsize 256 \
    --epoch 30 \
    --workers 4


#!/bin/sh

DATA_DIR=""  # <- Your Data Directory

python train.py \
    --train_data "${DATA_DIR}/0-train.h5" \
    --val_data "${DATA_DIR}/0-valid.h5" \
    --test_data "${DATA_DIR}/0-test.h5" \
    --discriminator_net "daft" \
    --task "surv" \
    --workers 4 \
    --epoch 80 \
    --learning_rate "0.0055" \
    --decay_rate "0.01" \
    --film_location 3 \
    --bottleneck_factor "7.0" \
    --n_basefilters 4 \
    --num_classes 1 \
    --batchsize 256 \
    --normalize_image "minmax" \
    --normalize_tabular


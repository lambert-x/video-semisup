#!/usr/bin/env bash
CONFIG_NAME=r3d_r18_8x8x1_180e_ucf101_rgb_offlinetg_20percent_align0123_L1_1clip_no_contrast_precisebn_ptv
CONFIG=configs/ablation_alignment_loss/$CONFIG_NAME.py
GPUS=8
SAVE_DIR1="./work_dirs/${CONFIG_NAME}_e1/"
SAVE_DIR2="./work_dirs/${CONFIG_NAME}_e2/"
MODEL_NAME='latest.pth'

./tools/dist_test.sh $CONFIG "$SAVE_DIR1$MODEL_NAME" $GPUS --eval top_k_accuracy

wait
sleep 10s
./tools/dist_test.sh $CONFIG "$SAVE_DIR2$MODEL_NAME" $GPUS --eval top_k_accuracy

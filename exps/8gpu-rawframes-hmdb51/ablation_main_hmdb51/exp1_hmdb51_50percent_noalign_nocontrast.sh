#!/usr/bin/env bash
CONFIG_NAME=r3d_r18_8x8x1_360e_hmdb51_rgb_offlinetg_50percent_noalign_no_contrast_syncbn
CONFIG=configs/ablation_main_hmdb51_50percent/$CONFIG_NAME.py
GPUS=8
SAVE_DIR1="./work_dirs/${CONFIG_NAME}_e1/"
MODEL_NAME='latest.pth'


./tools/dist_train_semi.sh $CONFIG $GPUS --validate \
  --work-dir=$SAVE_DIR1
wait
sleep 10s
./tools/dist_test.sh $CONFIG "$SAVE_DIR1$MODEL_NAME" $GPUS --eval top_k_accuracy

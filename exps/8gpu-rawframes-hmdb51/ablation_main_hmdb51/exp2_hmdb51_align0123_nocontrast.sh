#!/usr/bin/env bash
CONFIG_NAME=r3d_r18_8x8x1_360e_hmdb51_rgb_offlinetg_50percent_align0123_no_contrast
CONFIG=configs/hmdb51_ablation_main_hmdb51_50percent/$CONFIG_NAME.py
GPUS=8
SAVE_DIR1="./work_dirs/${CONFIG_NAME}_rawframes_e1/"
SAVE_DIR2="./work_dirs/${CONFIG_NAME}_rawframes_e2/"
MODEL_NAME='latest.pth'


bash ./tools/dist_train_semi.sh $CONFIG $GPUS --validate \
  --work-dir=$SAVE_DIR1
wait
bash ./tools/dist_test.sh $CONFIG "$SAVE_DIR1$MODEL_NAME" $GPUS --eval top_k_accuracy


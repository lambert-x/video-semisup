#!/usr/bin/env bash
CONFIG_NAME=r3d_r18_8x8x1_180e_ucf101_rgb_offlinetg_20percent_align0123_1clip_cosine_weak_sameclip_precisebn_ptv_new_loss_half
CONFIG=configs/ucf101_main/$CONFIG_NAME.py
GPUS=8
SAVE_DIR1="./work_dirs/${CONFIG_NAME}_rawframes_e1/"
SAVE_DIR2="./work_dirs/${CONFIG_NAME}_rawframes_e2/"
MODEL_NAME='latest.pth'


bash ./tools/dist_train_semi.sh $CONFIG $GPUS --validate \
  --work-dir=$SAVE_DIR1
wait
bash ./tools/dist_test.sh $CONFIG "$SAVE_DIR1$MODEL_NAME" $GPUS --eval top_k_accuracy


#!/usr/bin/env bash
CONFIG_NAME=r3d_r18_8x8x1_90e_k400_rgb_offlinetg_10percent_align0123_1clip_cosine_weak_sameclip_precisebn_ptv_new_loss_half
CONFIG=configs/k400-fixmatch-tg-alignment-videos-ptv-simclr/8gpu/$CONFIG_NAME.py
GPUS=8
SAVE_DIR1="./work_dirs/${CONFIG_NAME}_e1/"
SAVE_DIR2="./work_dirs/${CONFIG_NAME}_e2/"
MODEL_NAME='latest.pth'


bash ./tools/dist_train_semi.sh $CONFIG $GPUS --validate \
  --work-dir=$SAVE_DIR1
wait
bash ./tools/dist_test.sh $CONFIG "$SAVE_DIR1$MODEL_NAME" $GPUS --eval top_k_accuracy


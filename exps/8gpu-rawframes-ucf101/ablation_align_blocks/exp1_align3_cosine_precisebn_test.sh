#!/usr/bin/env bash
CONFIG_NAME=r3d_r18_8x8x1_180e_ucf101_rgb_offlinetg_20percent_align3_cosine_no_contrast
CONFIG=configs/semi_tempsup_fixmatch_crossclip_contrast/8gpu/$CONFIG_NAME.py
GPUS=8
SAVE_DIR1="./work_dirs/${CONFIG_NAME}_e1/"
SAVE_DIR2="./work_dirs/${CONFIG_NAME}_e2/"
MODEL_NAME='latest.pth'
PRECISE_BN_CONFIG="configs/semi_tempsup_fixmatch_crossclip_contrast/8gpu/${CONFIG_NAME}_precisebn.py"

./tools/dist_test.sh $PRECISE_BN_CONFIG "$SAVE_DIR1$MODEL_NAME" $GPUS --eval top_k_accuracy

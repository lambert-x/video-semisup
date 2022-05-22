#!/usr/bin/env bash
K400_TOOLS_DIR="./tools/data/kinetics"
DATASET='kinetics400'
cd ${K400_TOOLS_DIR}

bash download_annotations.sh$ {DATASET}
wait
bash rename_classnames.sh ${DATASET}
wait
python ../resize_video.py ../../../data/${DATASET}/train/ ../../../data/${DATASET}/videos_train \
--dense --level 2 --num-worker $1 --ext mp4
wait
python ../resize_video.py ../../../data/${DATASET}/train/ ../../../data/${DATASET}/videos_train \
--dense --level 2 --num-worker $1 --to-mp4 --ext mkv
wait

python ../resize_video.py ../../../data/${DATASET}/val/ ../../../data/${DATASET}/videos_val \
--dense --level 2 --num-worker $1 --ext mp4
wait
python ../resize_video.py ../../../data/${DATASET}/val/ ../../../data/${DATASET}/videos_val \
--dense --level 2 --num-worker $1 --to-mp4 --ext mkv
wait

bash generate_videos_filelist.sh ${DATASET}

# bash extract_rgb_frames_opencv.sh $1
# wait
# bash extract_temporal_gradient_frames.sh $1
# wait
# bash generate_rawframes_filelist.sh






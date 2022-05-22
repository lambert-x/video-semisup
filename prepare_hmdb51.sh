#!/usr/bin/env bash

HMDB51_TOOLS_DIR="./tools/data/hmdb51"

cd ${HMDB51_TOOLS_DIR}

bash download_videos.sh
wait
bash unzip_videos.sh
wait

# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
# mkdir /mnt/SSD/ucf101_extracted/
# ln -s /mnt/SSD/ucf101_extracted/ ../../../data/ucf101/rawframes

bash extract_rgb_frames_opencv.sh $1
wait
bash extract_temporal_gradient_frames.sh $1






#!/usr/bin/env bash

UCF101_TOOLS_DIR="./tools/data/ucf101"

cd ${UCF101_TOOLS_DIR}

bash download_annotations.sh
wait
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
wait
bash generate_rawframes_filelist.sh






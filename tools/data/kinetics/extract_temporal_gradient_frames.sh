#!/usr/bin/env bash

cd ../
python build_rawframes_temporal_gradient.py ../../data/kinetics400/rawframes_val/ ../../data/kinetics400/rawframes_val/ \
      --task temporal_grad --level 2 --num-worker $1 --input-frames
echo "Genearte Temporal Gradient frames with RGB raw frames"

cd kinetics/

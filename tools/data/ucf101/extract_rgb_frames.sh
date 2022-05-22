#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/ucf101/videos/ ../../data/ucf101/rawframes/ --task rgb --level 2  --ext avi --num-worker $1
echo "Genearte raw frames (RGB only)"

cd ucf101/

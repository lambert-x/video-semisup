#!/usr/bin/env bash

DATA_DIR="../../../data/ucf101/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

sudo apt-get install unrar
unrar x UCF101.rar
mv ./UCF-101 ./videos
rm UCF101.rar
cd "../../tools/data/ucf101"

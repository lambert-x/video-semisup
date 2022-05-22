#!/usr/bin/env bash

CONFIG=$1

./tools/dist_train_semi.sh $CONFIG 4 --validate ${@:2}


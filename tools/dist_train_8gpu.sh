#!/usr/bin/env bash

CONFIG=$1

./tools/dist_train.sh $CONFIG 8 --validate ${@:2}


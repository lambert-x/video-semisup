#!/usr/bin/env bash

# Rename classname for convenience
DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

cd ../../../data/${DATASET}/
ls ./train | while read class; do \
  newclass=`echo $class | tr " " "_" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "train/${class}" "train/${newclass}";
  fi
done

ls ./val | while read class; do \
  newclass=`echo $class | tr " " "_" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "val/${class}" "val/${newclass}";
  fi
done

cd ../../tools/data/kinetics/

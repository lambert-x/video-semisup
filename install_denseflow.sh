#!/usr/bin/env bash

# ZZROOT is the root dir of all the installation
# you may put these lines into your .bashrc/.zshrc/etc.
export ZZROOT=$HOME/app
export PATH=$ZZROOT/bin:$PATH
export LD_LIBRARY_PATH=$ZZROOT/lib:$ZZROOT/lib64:$LD_LIBRARY_PATH

# fetch install scripts
git clone https://github.com/lambert-x/setup.git
cd setup

# opencv depends on ffmpeg for video decoding
# ffmpeg depends on nasm, yasm, libx264, libx265, libvpx


./zznasm.sh
wait
./zzyasm.sh
wait
./zzlibx264.sh
wait
./zzlibx265.sh
wait
./zzlibvpx.sh
# finally install ffmpeg
wait
./zzffmpeg.sh
wait
# install opencv 4.5.2
./zzopencv.sh
# you may put this line into your .bashrc
export OpenCV_DIR=$ZZROOT
# install boost
wait
./zzboost.sh
# you may put this line into your .bashrc
export BOOST_ROOT=$ZZROOT
wait
# finally, install denseflow
./zzdenseflow.sh




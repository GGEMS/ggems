#!/bin/bash

GGEMS_FOLDER="/home/dbenoit/data/Dropbox/GGEMS"
BUILD_FOLDER="/home/dbenoit/data/Build/GGEMS"
INSTALL_FOLDER="/home/dbenoit"

echo 'GGEMS folder: '$GGEMS_FOLDER
echo 'GGEMS build folder: '$BUILD_FOLDER

echo 'Removing CMAKE cache...'
if test -f "$BUILD_FOLDER/CMakeCache.txt"; then
  rm $BUILD_FOLDER/CMakeCache.txt
fi

echo 'Installing GGEMS...'
cmake -DCMAKE_BUILD_TYPE=Release -DGGEMSHOME=$INSTALL_FOLDER -DFAST_MATH=ON -DDOUBLE_PRECISION=ON -DCOMPUTE_CAPABILITY_61=ON -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_FOLDER -S $GGEMS_FOLDER -B $BUILD_FOLDER

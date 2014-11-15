#! /bin/bash
export GGEMSHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../ && pwd )"
export GGEMSLIB=$GGEMSHOME/lib 
export GGEMSINC=$GGEMSHOME/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GGEMS/lib

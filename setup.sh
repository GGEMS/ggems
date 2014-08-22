export GGEMS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export GGEMSLIB=$GGEMS/lib 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GGEMS/lib

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

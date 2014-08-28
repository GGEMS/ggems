#! /bin/bash
export GGEMS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export GGEMSLIB=$GGEMS/lib 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GGEMS/lib

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

export SOFTWARE=/home/software
source $SOFTWARE/geant4/geant4.9.6.p03-install/bin/geant4.sh
export G4HOME=$SOFTWARE/geant4/geant4.9.6.p03-install

#*******CLHEP****
export PATH=$PATH:$SOFTWARE/geant4/CLHEP/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SOFTWARE/geant4/geant4.9.6.p03-install/lib
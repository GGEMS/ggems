// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef GLOBAL_CU
#define GLOBAL_CU
#include "global.cuh"
#include "../processes/structures.cuh"
// Set a GPU device
void set_gpu_device(int deviceChoice, float minversion) {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("[\033[31;03mWARNING\033[00m] There is no device supporting CUDA\n");
        exit(EXIT_FAILURE);
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceChoice%deviceCount);

    if(prop.major<minversion) {
        printf("[\033[31;03mWARNING\033[00m] Your device is not compatible with %1.1f version\n",minversion);    
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(deviceChoice%deviceCount);
    printf("[\033[32;01mok\033[00m] \e[1m%s\e[21m found\n",prop.name);
    
}

// Reset the GPU
void reset_gpu_device() {
    printf("[\033[32;01mok\033[00m] Reset device .. \n");
    cudaDeviceReset();
}



#endif

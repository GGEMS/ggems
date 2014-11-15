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
/*
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
*/

// Print out for error
void print_error(std::string msg) {
    printf("\033[31;03m[ERROR] - %s\033[00m", msg.c_str());
}

// Print out for warning
void print_warning(std::string msg) {
    printf("\033[33;03m[WARNING] - %s\033[00m", msg.c_str());
}

// Exit the soft
// Abort the current simulation
void exit_simulation() {
    printf("[\033[31;03mSimulation aborded\033[00m]\n");
    exit(EXIT_FAILURE);
}

//////// Operation on C-Array //////////////////////////////////////////////////

// Equivalent to std::vector.push_back for malloc array (unsigned int version)
void array_push_back(unsigned int **vector, unsigned int &dim, unsigned int val) {

    // If first allocation
    if (dim == 0) {
        (*vector) = (unsigned int*)malloc(sizeof(unsigned int));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (unsigned int*)realloc((*vector), (dim+1)*sizeof(unsigned int));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    }

    (*vector)[dim] = val;
    (dim)++;
}

// Equivalent to std::vector.push_back for malloc array (float version)
void array_push_back(float **vector, unsigned int &dim, float val) {

    // If first allocation
    if (dim == 0) {
        (*vector) = (float*)malloc(sizeof(float));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (float*)realloc((*vector), (dim+1)*sizeof(float));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    }

    (*vector)[dim] = val;
    (dim)++;

}

// Equivalent to std::vector.insert for malloc array (unsigned int version)
void array_insert(unsigned int **vector, unsigned int &dim, unsigned int pos, unsigned int val) {

    // Check the pos value
    if (pos > dim) {
        print_error("Position out of range from array_insert!!!\n");
        exit(EXIT_FAILURE);
    }

    // If first allocation
    if (dim == 0) {
        (*vector) = (unsigned int*)malloc(sizeof(unsigned int));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (unsigned int*)realloc((*vector), (dim+1)*sizeof(unsigned int));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
        // Move data in memory for the insertion
        memmove((*vector)+pos, (*vector)+pos+1, (dim-pos)*sizeof(unsigned int));
    }

    (*vector)[pos] = val;
    (dim)++;

}

// Equivalent to std::vector.insert for malloc array (unsigned int version)
void array_insert(float **vector, unsigned int &dim, unsigned int pos, float val) {

    // Check the pos value
    if (pos > dim) {
        print_error("Position out of range from array_insert!!!\n");
        exit(EXIT_FAILURE);
    }

    // If first allocation
    if (dim == 0) {
        (*vector) = (float*)malloc(sizeof(float));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (float*)realloc((*vector), (dim+1)*sizeof(float));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
        // Move data in memory for the insertion
        memmove((*vector)+pos, (*vector)+pos+1, (dim-pos)*sizeof(float));
    }

    (*vector)[pos] = val;
    (dim)++;

}

// Append an array to another one (float version)
void array_append_array(float **vector, unsigned int &dim, float **an_array, unsigned int a_dim) {

    // If first allocation
    if (dim == 0) {
        (*vector) = (float*)malloc(sizeof(float));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_append_array!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (float*)realloc((*vector), (dim+a_dim)*sizeof(float));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_append_array!!!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Copy data
    memcpy((*vector)+dim, (*an_array), a_dim*sizeof(float));
    (dim)+=a_dim;
}

// Create a color
Color make_color(float r, float g, float b) {
    Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    return c;
}












#endif

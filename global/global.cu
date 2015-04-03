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

// Some usefull functions

// Reset the GPU
void reset_gpu_device() {
    printf("[\033[32;01mok\033[00m] Reset device .. \n");
    cudaDeviceReset();
}

// comes from "cuda by example" book
void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// comes from "cuda programming" book
__host__ void cuda_error_check (const char * prefix, const char * postfix) {
    if(cudaPeekAtLastError() != cudaSuccess ) {
        printf("\n%s%s%s\n",prefix, cudaGetErrorString(cudaGetLastError()),postfix);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

}

// Set a GPU device
void set_gpu_device(int deviceChoice, f32 minversion) {

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
    printf("[\033[32;01mok\033[00m] \e[1m%s\e[21m found\n", prop.name);
    
}

// Print out for error
void print_error(std::string msg) {
    printf("\033[31;03m[ERROR] - %s\033[00m", msg.c_str());
}

// Print out for warning
void print_warning(std::string msg) {
    printf("\033[33;03m[WARNING] - %s\033[00m", msg.c_str());
}

// Print out run time
void print_time(std::string txt, f64 t) {

    f64 res;
    ui32 time_h = (ui32)(t / 3600.0);
    res = t - (time_h*3600.0);
    ui32 time_m = (ui32)(res / 60.0);
    res -= (time_m * 60.0);
    ui32 time_s = (ui32)(res);
    res -= time_s;
    ui32 time_ms = (ui32)(res*1000.0);

    printf("[\033[32;01mRun time\033[00m] %s: ", txt.c_str());

    if (time_h != 0) printf("%i h ", time_h);
    if (time_m != 0) printf("%i m ", time_m);
    if (time_s != 0) printf("%i s ", time_s);
    printf("%i ms\n", time_ms);

}

// Print out memory usage
void print_memory(std::string txt, ui32 t) {

    std::vector<std::string> pref;
    pref.push_back("B");
    pref.push_back("kB");
    pref.push_back("MB");
    pref.push_back("GB");

    ui32 iemem = (ui32)(log(t) / log(1000));
    f32 mem = f32(t) / (pow(1000, iemem));

    printf("[\033[34;01mMemory usage\033[00m] %s: %5.2f %s\n", txt.c_str(), mem, pref[iemem].c_str());

}

// Abort the current simulation
void exit_simulation() {
    printf("[\033[31;03mSimulation aborded\033[00m]\n");
    exit(EXIT_FAILURE);
}

//////// Operation on C-Array //////////////////////////////////////////////////

// Equivalent to std::vector.push_back for malloc array (ui32 version)
void array_push_back(ui32 **vector, ui32 &dim, ui32 val) {

    // If first allocation
    if (dim == 0) {
        (*vector) = (ui32*)malloc(sizeof(ui32));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (ui32*)realloc((*vector), (dim+1)*sizeof(ui32));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    }

    (*vector)[dim] = val;
    (dim)++;
}

// Equivalent to std::vector.push_back for malloc array (f32 version)
void array_push_back(f32 **vector, ui32 &dim, f32 val) {

    // If first allocation
    if (dim == 0) {
        (*vector) = (f32*)malloc(sizeof(f32));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (f32*)realloc((*vector), (dim+1)*sizeof(f32));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_push_back!!!\n");
            exit(EXIT_FAILURE);
        }
    }

    (*vector)[dim] = val;
    (dim)++;

}


// Equivalent to std::vector.insert for malloc array (ui32 version)
void array_insert(ui32 **vector, ui32 &dim, ui32 pos, ui32 val) {

    // Check the pos value
    if (pos > dim) {
        print_error("Position out of range from array_insert!!!\n");
        exit(EXIT_FAILURE);
    }

    // If first allocation
    if (dim == 0) {
        (*vector) = (ui32*)malloc(sizeof(ui32));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (ui32*)realloc((*vector), (dim+1)*sizeof(ui32));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
        // Move data in memory for the insertion
        memmove((*vector)+pos+1, (*vector)+pos, (dim-pos)*sizeof(ui32));
    }
    printf("pos %d val %d \n", pos, val);
    (*vector)[pos] = val;
    (dim)++;

}

// Equivalent to std::vector.insert for malloc array (ui32 version)
void array_insert(f32 **vector, ui32 &dim, ui32 pos, f32 val) {

    // Check the pos value
    if (pos > dim) {
        print_error("Position out of range from array_insert!!!\n");
        exit(EXIT_FAILURE);
    }

    // If first allocation
    if (dim == 0) {
        (*vector) = (f32*)malloc(sizeof(f32));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (f32*)realloc((*vector), (dim+1)*sizeof(f32));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_insert!!!\n");
            exit(EXIT_FAILURE);
        }
        // Move data in memory for the insertion
        memmove((*vector)+pos+1, (*vector)+pos, (dim-pos)*sizeof(f32));
    }
    (*vector)[pos] = val;
    (dim)++;

}

// Append an array to another one (f32 version)
void array_append_array(f32 **vector, ui32 &dim, f32 **an_array, ui32 a_dim) {

    // If first allocation
    if (dim == 0) {
        (*vector) = (f32*)malloc(sizeof(f32));
        if ((*vector) == NULL) {
            print_error("Memory allocation from array_append_array!!!\n");
            exit(EXIT_FAILURE);
        }
    // else reallocation
    } else {
        (*vector) = (f32*)realloc((*vector), (dim+a_dim)*sizeof(f32));
        if ((*vector) == NULL) {
            print_error("Memory reallocation from array_append_array!!!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Copy data
    memcpy((*vector)+dim, (*an_array), a_dim*sizeof(f32));
    (dim)+=a_dim;
}

// Create a color
Color make_color(f32 r, f32 g, f32 b) {
    Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    return c;
}

// Get time
f64 get_time() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}







#endif

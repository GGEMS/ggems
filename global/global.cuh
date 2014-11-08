// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef GLOBAL_CUH
#define GLOBAL_CUH

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include "constants.cuh"
#include "vector.cuh"

//void set_gpu_device(int deviceChoice,float minversion=3.0);
//void reset_gpu_device();

void print_error(std::string msg);
void print_warning(std::string msg);

void exit_simulation();

// Operation on C-Array
void array_push_back(unsigned int **vector, unsigned int &dim, unsigned int val);
void array_push_back(float **vector, unsigned int &dim, float val);
void array_insert(unsigned int **vector, unsigned int &dim, unsigned int pos, unsigned int val);
void array_insert(float **vector, unsigned int &dim, unsigned int pos, float val);
void array_append_array(float **vector, unsigned int &dim, float **an_array, unsigned int a_dim);

// Global simulation parameters
struct GlobalSimulationParameters {
    char physics_list[NB_PROCESSES];
    char secondaries_list[NB_PARTICLES];
    char record_dose_flag;

    unsigned int nb_of_particles;
    unsigned int nb_iterations;

    float time;
    unsigned int seed;

    // To build cross sections table
    unsigned int cs_table_nbins;
    float cs_table_min_E;
    float cs_table_max_E;

};

// Struct that handle colors
struct Color {
    float r, g, b;
};


#endif

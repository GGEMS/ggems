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

#ifndef PARTICLES_CUH
#define PARTICLES_CUH

#include "../global/global.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>

// Stack of particles, format data is defined as SoA
struct ParticleStack{
    // property
    float* E;
    float* dx;
    float* dy;
    float* dz;
    float* px;
    float* py;
    float* pz;
    float* tof;
    // PRNG
    unsigned int* prng_state_1;
    unsigned int* prng_state_2;
    unsigned int* prng_state_3;
    unsigned int* prng_state_4;
    unsigned int* prng_state_5;
    // Navigation
    unsigned int* geometry_id; // current geometry crossed by the particle
    // simulation
    unsigned char* endsimu;
    unsigned char* level;
    unsigned char* pname; // particle name (photon, electron, etc)
    // stack size
    unsigned int size;
}; //

// Helper to handle secondaries particles
struct SecParticle {
    float3 dir;
    float E;
    unsigned char pname;
    unsigned char endsimu;
};

// Helper to handle history of particles
struct OneParticleStep {
    float3 pos;
    float3 dir;
    float E;
};

// History class
class HistoryBuilder {
    public:
        HistoryBuilder();

        void cpu_new_particle_track(unsigned int a_pname);
        void cpu_record_a_step(ParticleStack particles, unsigned int id_part);

        std::vector<unsigned char> pname;
        std::vector<unsigned int> nb_steps;
        std::vector< std::vector<OneParticleStep> > history_data;

        unsigned char record_flag;      // Record or not
        unsigned int max_nb_particles;  // Max nb of particles keep in the history
        unsigned int cur_iter;          // Current number of iterations
        unsigned int stack_size;        // Size fo the particle stack

    private:
        unsigned int current_particle_id;
        unsigned char type_of_particles;
};


// Particles class
class ParticleBuilder {
    public:
        ParticleBuilder();
        void set_stack_size(unsigned int nb);
        void set_seed(unsigned int val_seed);
        void cpu_malloc_stack();
        void init_stack_seed();
        void cpu_print_stack(unsigned int nlim);

        ParticleStack stack;
        unsigned int seed;

    private:


};

#endif
